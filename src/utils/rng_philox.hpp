#ifndef __RNG_PHILOX_H__
#define __RNG_PHILOX_H__

#include <cmath>
#include <cstdint>
#include <vector>

#include "rng.hpp"

// RNG imitiating torch cuda randn on CPU.
// Port from: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/5ef669de080814067961f28357256e8fe27544f4/modules/rng_philox.py
//
// torch's CUDA randn (curand_normal4 over a Philox4x32-10 state) does NOT fill a
// flat tensor sequentially. It launches a grid-stride kernel whose grid is
//   grid_x = min(SM_count * (maxThreadsPerMultiProcessor / 256), ceil(n / 256))
// and unrolls 4 values per thread. Element `i` therefore maps to
//   idx = i % GB,  ii = i / GB   (GB = grid_x * 256)
//   value = [sin, cos, sin, cos][ii] of the two Box-Muller pairs drawn from
//   Philox(counter = (offset, 0, idx, 0), key = seed).
// On a GPU with few enough SMs (e.g. RTX 3090: 82 SMs x 6 blocks = 492), GB < n
// so the mapping wraps. The original A1111 port (and this class before the fix)
// simplified this to "counter[2] = i", which only matches the first GB elements.
//
// Values match torch's CUDA randn structurally but may differ by a few float
// ULPs: cuRAND computes Box-Muller with CUDA device sinf/logf, whose results
// differ from the host <cmath> functions used here. The residual difference
// (max ~2e-6 on N(0,1)) is far below fp16/bf16 resolution, so downstream
// outputs are unaffected.
class PhiloxRNG : public RNG {
private:
    uint64_t seed;
    uint32_t offset;

    // GPU launch config used by torch's calc_execution_policy. 0 = unset, which
    // disables the grid clamp (grid_x = ceil(n/256), i.e. the "no wrap" case).
    uint32_t sm_count_           = 0;
    uint32_t max_threads_per_sm_ = 0;

    static constexpr uint32_t philox_m0 = 0xD2511F53;
    static constexpr uint32_t philox_m1 = 0xCD9E8D57;
    static constexpr uint32_t philox_w0 = 0x9E3779B9;
    static constexpr uint32_t philox_w1 = 0xBB67AE85;
    static constexpr float    two_pow32_inv     = 2.3283064e-10f;
    static constexpr float    two_pow32_inv_2pi = 2.3283064e-10f * 6.2831855f;

    // A single round of the Philox 4x32 random number generator (scalar).
    static inline void philox4_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                     uint32_t k0, uint32_t k1) {
        const uint64_t v1  = static_cast<uint64_t>(c0) * philox_m0;
        const uint64_t v2  = static_cast<uint64_t>(c2) * philox_m1;
        const uint32_t hi1 = static_cast<uint32_t>(v1 >> 32);
        const uint32_t lo1 = static_cast<uint32_t>(v1);
        const uint32_t hi2 = static_cast<uint32_t>(v2 >> 32);
        const uint32_t lo2 = static_cast<uint32_t>(v2);
        c0 = hi2 ^ c1 ^ k0;
        c1 = lo2;
        c2 = hi1 ^ c3 ^ k1;
        c3 = lo1;
    }

    // Full Philox 4x32-10: ten rounds with a key increment between the first nine.
    static inline void philox4_32(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                  uint32_t k0, uint32_t k1) {
        for (int i = 0; i < 9; ++i) {
            philox4_round(c0, c1, c2, c3, k0, k1);
            k0 += philox_w0;
            k1 += philox_w1;
        }
        philox4_round(c0, c1, c2, c3, k0, k1);
    }

    // Box-Muller transform of two uniform uint32s -> (sin, cos) pair, matching
    // cuRAND's _curand_box_muller: result.x = s*sin(v), result.y = s*cos(v).
    static inline void box_muller(uint32_t x, uint32_t y, float& r_sin, float& r_cos) {
        const float u = static_cast<float>(x) * two_pow32_inv + two_pow32_inv / 2;
        const float v = static_cast<float>(y) * two_pow32_inv_2pi + two_pow32_inv_2pi / 2;
        const float s = sqrt(-2.0f * log(u));
        r_sin = s * sin(v);
        r_cos = s * cos(v);
    }

public:
    PhiloxRNG(uint64_t seed = 0) : seed(seed), offset(0) {}

    void manual_seed(uint64_t seed) override {
        this->seed   = seed;
        this->offset = 0;
    }

    // Configure the grid-stride launch shape torch uses for CUDA randn on the
    // active device (values from cudaGetDeviceProperties). Leave unset to keep the
    // "no wrap" mapping (grid_x = ceil(n/256)), which is what torch produces on a
    // GPU with enough SMs to cover the tensor in a single wave.
    void set_device_properties(uint32_t sm_count, uint32_t max_threads_per_sm) override {
        sm_count_           = sm_count;
        max_threads_per_sm_ = max_threads_per_sm;
    }

    std::vector<float> randn(uint32_t n) override {
        if (n == 0) {
            return {};
        }

        const uint32_t block_size = 256;
        const uint32_t unroll     = 4;

        // Replicate torch's calc_execution_policy grid clamp.
        uint32_t grid_x = (n + block_size - 1) / block_size;
        if (sm_count_ > 0 && max_threads_per_sm_ > 0) {
            uint32_t blocks_per_sm = max_threads_per_sm_ / block_size;
            if (blocks_per_sm == 0) {
                blocks_per_sm = 1;
            }
            const uint32_t clamped = sm_count_ * blocks_per_sm;
            if (clamped < grid_x) {
                grid_x = clamped;
            }
        }

        const uint64_t gb      = static_cast<uint64_t>(grid_x) * block_size;
        const uint64_t rounded = ((static_cast<uint64_t>(n) - 1) / (gb * unroll) + 1) * gb * unroll;

        // torch advances the generator offset by max_generator_offsets_per_curand_call
        // (4) times the number of grid-stride iterations per thread.
        const uint64_t counter_offset = ((static_cast<uint64_t>(n) - 1) / (gb * unroll) + 1) * 4;
        const uint64_t base_offset    = this->offset;
        this->offset                  = static_cast<uint32_t>(base_offset + counter_offset);

        // curand_init forwards the generator offset through skipahead(), which
        // divides it by 4: the counter advances by counter_offset/4 per call,
        // not counter_offset. Verified bitwise against torch on RTX 3090.
        const uint64_t counter_base = base_offset / 4;

        const uint32_t k0 = static_cast<uint32_t>(seed & 0xFFFFFFFFu);
        const uint32_t k1 = static_cast<uint32_t>(seed >> 32);

        std::vector<float> result(n, 0.0f);

        for (uint64_t idx = 0; idx < gb; ++idx) {
            uint64_t ctr_x = counter_base;
            for (uint64_t linear = idx; linear < rounded; linear += gb * unroll) {
                // counter = (ctr_x, 0, idx, 0); key = seed.
                uint32_t c0 = static_cast<uint32_t>(ctr_x);
                uint32_t c1 = 0;
                uint32_t c2 = static_cast<uint32_t>(idx);
                uint32_t c3 = 0;
                philox4_32(c0, c1, c2, c3, k0, k1);
                // c0..c3 now hold the four uint32 outputs of Philox.
                float out0, out1, out2, out3;
                box_muller(c0, c1, out0, out1);  // sin, cos from (g0, g1)
                box_muller(c2, c3, out2, out3);  // sin, cos from (g2, g3)

                const uint64_t li0 = linear;
                if (li0 < n) result[li0] = out0;
                const uint64_t li1 = linear + gb;
                if (li1 < n) result[li1] = out1;
                const uint64_t li2 = linear + 2 * gb;
                if (li2 < n) result[li2] = out2;
                const uint64_t li3 = linear + 3 * gb;
                if (li3 < n) result[li3] = out3;

                ++ctr_x;
            }
        }
        return result;
    }
};

#endif  // __RNG_PHILOX_H__
