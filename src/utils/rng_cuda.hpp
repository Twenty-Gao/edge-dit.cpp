#ifndef __RNG_CUDA_H__
#define __RNG_CUDA_H__

// Shared helper for matching torch's CUDA randn launch shape. Only meaningful
// under CUDA; compiles out entirely for CPU-only builds.
#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>

#include "rng.hpp"

// Configure `rng` to reproduce torch's CUDA randn grid-stride launch shape
// (grid clamp from calc_execution_policy) on `device_ordinal`, so CPU-generated
// noise matches torch.Generator(device="cuda"). Call once per PhiloxRNG, right
// after creating it; no-op for RNGs that ignore set_device_properties.
inline void configure_rng_for_cuda_randn(RNG& rng, int device_ordinal) {
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_ordinal) == cudaSuccess) {
        rng.set_device_properties(props.multiProcessorCount, props.maxThreadsPerMultiProcessor);
    }
}
#endif  // GGML_USE_CUDA

#endif  // __RNG_CUDA_H__
