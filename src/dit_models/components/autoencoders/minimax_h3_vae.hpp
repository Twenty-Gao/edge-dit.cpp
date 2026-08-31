#ifndef __ED_DIT_MODELS_COMPONENTS_AUTOENCODERS_MINIMAX_H3_VAE_HPP__
#define __ED_DIT_MODELS_COMPONENTS_AUTOENCODERS_MINIMAX_H3_VAE_HPP__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dit_models/components/common/rope.hpp"
#include "dit_models/components/common/common_dit.hpp"
#include "dit_models/components/autoencoders/vae.hpp"

namespace MiniMaxH3VAE {

    constexpr int H3_VIDEO_VAE_GRAPH_SIZE = 262144;

    static bool h3_env_flag_enabled(const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    static bool h3_env_flag_enabled_or_default(const char* name, bool default_enabled) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return default_enabled;
        }
        return std::strcmp(value, "0") != 0;
    }

    static ggml_tensor* h3_profile_name(ggml_tensor* tensor, const std::string& name) {
        if (tensor != nullptr && h3_env_flag_enabled("ED_MINIMAX_H3_VAE_NAME_NODES")) {
            ggml_set_name(tensor, name.c_str());
        }
        return tensor;
    }

    struct ScopedEnvFlag {
        std::string name;
        std::string old_value;
        bool had_value = false;

        ScopedEnvFlag(const char* key, const char* value)
            : name(key) {
            const char* current = std::getenv(key);
            if (current != nullptr) {
                had_value = true;
                old_value = current;
            }
            setenv(name.c_str(), value, 1);
        }

        ~ScopedEnvFlag() {
            if (had_value) {
                setenv(name.c_str(), old_value.c_str(), 1);
            } else {
                unsetenv(name.c_str());
            }
        }
    };

    struct CausalConv3d : public Conv3d {
        std::tuple<int, int, int> temporal_padding;

        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     std::tuple<int, int, int> kernel_size,
                     std::tuple<int, int, int> stride  = {1, 1, 1},
                     std::tuple<int, int, int> padding = {0, 0, 0})
            : Conv3d(in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     {0, 0, 0}),
              temporal_padding(padding) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto reflect_pad = [&](ggml_tensor* value, int dim, int amount) {
                for (int i = 0; i < amount; ++i) {
                    GGML_ASSERT(value->ne[dim] > 1);
                    auto left  = ggml_ext_slice(ctx->ggml_ctx, value, dim, 1, 2);
                    auto right = ggml_ext_slice(ctx->ggml_ctx,
                                                value,
                                                dim,
                                                value->ne[dim] - 2,
                                                value->ne[dim] - 1);
                    value      = ggml_concat(ctx->ggml_ctx, left, value, dim);
                    value      = ggml_concat(ctx->ggml_ctx, value, right, dim);
                }
                return value;
            };

            const int spatial_pad_x = std::get<2>(temporal_padding);
            const int spatial_pad_y = std::get<1>(temporal_padding);
            const bool fused_reflect_supported = sd_backend_is(ctx->backend, "CUDA") ||
                                                  ggml_backend_is_cpu(ctx->backend);
            if ((spatial_pad_x > 0 || spatial_pad_y > 0) &&
                fused_reflect_supported &&
                h3_env_flag_enabled_or_default("ED_MINIMAX_H3_VAE_FUSED_REFLECT_PAD", true)) {
                x = ggml_pad_ext_reflect(ctx->ggml_ctx,
                                         x,
                                         spatial_pad_x,
                                         spatial_pad_x,
                                         spatial_pad_y,
                                         spatial_pad_y,
                                         0,
                                         0,
                                         0,
                                         0);
            } else {
                x = reflect_pad(x, 0, spatial_pad_x);
                x = reflect_pad(x, 1, spatial_pad_y);
            }
            int temporal_pad = std::get<0>(temporal_padding) * 2;
            if (temporal_pad > 0) {
                x = ggml_ext_pad_ext(ctx->ggml_ctx,
                                     x,
                                     0,
                                     0,
                                     0,
                                     0,
                                     temporal_pad,
                                     0,
                                     0,
                                     0);
            }
            return Conv3d::forward(ctx, x);
        }
    };

    struct TemporalGroupNorm : public GroupNorm {
        explicit TemporalGroupNorm(int64_t channels)
            : GroupNorm(32, channels, 1e-6f, true) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            if (h3_env_flag_enabled_or_default("ED_MINIMAX_H3_VAE_FAST_TEMPORAL_GROUP_NORM", true)) {
                GGML_ASSERT(x->ne[3] % num_channels == 0);
                const int64_t temporal_size = x->ne[2];
                const int64_t batch_size    = x->ne[3] / num_channels;
                if (batch_size == 1 && temporal_size > 1) {
                    const bool fused_group_norm_supported = sd_backend_is(ctx->backend, "CUDA") ||
                                                            ggml_backend_is_cpu(ctx->backend);
                    if (fused_group_norm_supported &&
                        h3_env_flag_enabled_or_default("ED_MINIMAX_H3_VAE_FUSED_TEMPORAL_GROUP_NORM", true)) {
                        ggml_tensor* result = ggml_group_norm_temporal(ctx->ggml_ctx, x, num_groups, eps);
                        if (affine) {
                            ggml_tensor* weight = params["weight"];
                            ggml_tensor* bias = params["bias"];
                            weight = ggml_reshape_4d(ctx->ggml_ctx, weight, 1, 1, 1, weight->ne[0]);
                            bias = ggml_reshape_4d(ctx->ggml_ctx, bias, 1, 1, 1, bias->ne[0]);
                            result = ggml_mul_inplace(ctx->ggml_ctx, result, weight);
                            result = ggml_add_inplace(ctx->ggml_ctx, result, bias);
                        }
                        return result;
                    }
                    x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
                    x = GroupNorm::forward(ctx, x);
                    return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
                }
            }
            ggml_tensor* result = nullptr;
            for (int64_t t = 0; t < x->ne[2]; ++t) {
                auto frame = ggml_ext_slice(ctx->ggml_ctx, x, 2, t, t + 1);
                GGML_ASSERT(frame->ne[3] % num_channels == 0);
                int64_t batch_size = frame->ne[3] / num_channels;
                frame              = ggml_cont(ctx->ggml_ctx, frame);
                frame              = ggml_reshape_4d(ctx->ggml_ctx,
                                                     frame,
                                                     frame->ne[0],
                                                     frame->ne[1],
                                                     num_channels,
                                                     batch_size);
                frame              = GroupNorm::forward(ctx, frame);
                frame              = ggml_reshape_4d(ctx->ggml_ctx,
                                                     frame,
                                                     frame->ne[0],
                                                     frame->ne[1],
                                                     1,
                                                     num_channels * batch_size);
                result             = result == nullptr ? frame : ggml_concat(ctx->ggml_ctx, result, frame, 2);
            }
            return result;
        }
    };

    struct Downsample3D : public GGMLBlock {
        int spatial_stride;

        Downsample3D(int64_t in_channels,
                     int64_t out_channels,
                     int temporal_stride,
                     int spatial_stride)
            : spatial_stride(spatial_stride) {
            blocks["conv"] = std::make_shared<CausalConv3d>(in_channels,
                                                            out_channels,
                                                            std::tuple{3, 3, 3},
                                                            std::tuple{temporal_stride, spatial_stride, spatial_stride},
                                                            std::tuple{1, 0, 0});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            if (spatial_stride == 2) {
                GGML_ASSERT(x->ne[0] > 1 && x->ne[1] > 1);
                auto right  = ggml_ext_slice(ctx->ggml_ctx, x, 0, x->ne[0] - 2, x->ne[0] - 1);
                x           = ggml_concat(ctx->ggml_ctx, x, right, 0);
                auto bottom = ggml_ext_slice(ctx->ggml_ctx, x, 1, x->ne[1] - 2, x->ne[1] - 1);
                x           = ggml_concat(ctx->ggml_ctx, x, bottom, 1);
            }
            return std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"])->forward(ctx, x);
        }
    };

    struct ResnetBlock3D : public GGMLBlock {
        int64_t in_channels;
        int64_t out_channels;

        ResnetBlock3D(int64_t in_channels,
                      int64_t out_channels)
            : in_channels(in_channels), out_channels(out_channels) {
            blocks["norm1"] = std::make_shared<TemporalGroupNorm>(in_channels);
            blocks["norm2"] = std::make_shared<TemporalGroupNorm>(out_channels);
            blocks["conv1"] = std::make_shared<CausalConv3d>(in_channels,
                                                             out_channels,
                                                             std::tuple{3, 3, 3},
                                                             std::tuple{1, 1, 1},
                                                             std::tuple{1, 1, 1});
            blocks["conv2"] = std::make_shared<CausalConv3d>(out_channels,
                                                             out_channels,
                                                             std::tuple{3, 3, 3},
                                                             std::tuple{1, 1, 1},
                                                             std::tuple{1, 1, 1});
            if (in_channels != out_channels) {
                blocks["nin_shortcut"] = std::make_shared<CausalConv3d>(in_channels,
                                                                        out_channels,
                                                                        std::tuple{1, 1, 1});
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm1 = std::dynamic_pointer_cast<TemporalGroupNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<TemporalGroupNorm>(blocks["norm2"]);
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);
            auto h     = conv1->forward(ctx, ggml_silu(ctx->ggml_ctx, norm1->forward(ctx, x)));
            h          = conv2->forward(ctx, ggml_silu(ctx->ggml_ctx, norm2->forward(ctx, h)));
            if (in_channels != out_channels) {
                x = std::dynamic_pointer_cast<CausalConv3d>(blocks["nin_shortcut"])->forward(ctx, x);
            }
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct Encoder : public GGMLBlock {
        static constexpr int levels                            = 6;
        static constexpr std::array<int, levels> multipliers   = {1, 2, 2, 4, 4, 8};
        static constexpr std::array<int, levels> spatial_down  = {2, 2, 2, 2, 1, 1};
        static constexpr std::array<int, levels> temporal_down = {1, 2, 2, 1, 1, 1};

        Encoder() {
            constexpr int ch  = 128;
            blocks["conv_in"] = std::make_shared<CausalConv3d>(3,
                                                               ch,
                                                               std::tuple{3, 3, 3},
                                                               std::tuple{1, 1, 1},
                                                               std::tuple{1, 1, 1});
            int64_t previous  = ch;
            for (int level = 0; level < levels; ++level) {
                int64_t current = ch * multipliers[level];
                for (int block = 0; block < 2; ++block) {
                    blocks["down." + std::to_string(level) + ".block." + std::to_string(block)] =
                        std::make_shared<ResnetBlock3D>(block == 0 ? previous : current,
                                                        current);
                }
                if (spatial_down[level] * temporal_down[level] > 1) {
                    blocks["down." + std::to_string(level) + ".downsample"] =
                        std::make_shared<Downsample3D>(current,
                                                       current,
                                                       temporal_down[level],
                                                       spatial_down[level]);
                }
                previous = current;
            }
            blocks["norm_out"] = std::make_shared<TemporalGroupNorm>(previous);
            blocks["conv_out"] = std::make_shared<CausalConv3d>(previous,
                                                                48,
                                                                std::tuple{3, 3, 3},
                                                                std::tuple{1, 1, 1},
                                                                std::tuple{1, 1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"])->forward(ctx, x);
            for (int level = 0; level < levels; ++level) {
                for (int block = 0; block < 2; ++block) {
                    x = std::dynamic_pointer_cast<ResnetBlock3D>(
                            blocks["down." + std::to_string(level) + ".block." + std::to_string(block)])
                            ->forward(ctx, x);
                }
                auto downsample = blocks.find("down." + std::to_string(level) + ".downsample");
                if (downsample != blocks.end()) {
                    x = std::dynamic_pointer_cast<Downsample3D>(downsample->second)->forward(ctx, x);
                }
            }
            auto norm = std::dynamic_pointer_cast<TemporalGroupNorm>(blocks["norm_out"]);
            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);
            return conv->forward(ctx, ggml_silu(ctx->ggml_ctx, norm->forward(ctx, x)));
        }
    };

    static ggml_tensor* attention_layout(ggml_context* ctx, ggml_tensor* x) {
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        return ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2] * x->ne[3]);
    }

    static ggml_tensor* apply_partial_rope(ggml_context* ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* pe,
                                           ggml_backend_t backend,
                                           const std::string& profile_prefix = "vae.attn.rope") {
        int64_t rot_dim = pe->ne[2] * 2;
        auto rot_src     = h3_profile_name(ggml_ext_slice(ctx, x, 0, 0, rot_dim),
                                           profile_prefix + ".rot_slice");
        auto rotated    = Rope::apply_rope(ctx,
                                           rot_src,
                                           pe,
                                           false,
                                           backend);
        h3_profile_name(rotated, profile_prefix + ".rotated");
        if (rot_dim == x->ne[0]) {
            return rotated;
        }
        auto tail_src = h3_profile_name(ggml_ext_slice(ctx, x, 0, rot_dim, x->ne[0]),
                                        profile_prefix + ".tail_slice");
        auto tail = attention_layout(ctx,
                                     tail_src);
        h3_profile_name(tail, profile_prefix + ".tail_layout");
        return h3_profile_name(ggml_concat(ctx, rotated, tail, 0), profile_prefix + ".concat");
    }

    struct DecoderAttention : public GGMLBlock {
        static constexpr int num_head = 32;
        static constexpr int head_dim = 64;
        static constexpr int dim      = num_head * head_dim;

        DecoderAttention() {
            blocks["to_qkv"] = std::make_shared<Linear>(dim, dim * 3, true);
            blocks["to_out"] = std::make_shared<Linear>(dim, dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             const std::string& profile_prefix = "vae.attn") {
            auto to_qkv         = std::dynamic_pointer_cast<Linear>(blocks["to_qkv"]);
            auto to_out         = std::dynamic_pointer_cast<Linear>(blocks["to_out"]);
            auto qkv_projection = to_qkv->forward(ctx, x);
            h3_profile_name(qkv_projection, profile_prefix + ".qkv_proj");
            int64_t sequence    = x->ne[1];
            int64_t batch_size  = x->ne[2] * x->ne[3];
            qkv_projection      = ggml_reshape_4d(ctx->ggml_ctx,
                                                  qkv_projection,
                                                  3 * head_dim,
                                                  num_head,
                                                  sequence,
                                                  batch_size);
            h3_profile_name(qkv_projection, profile_prefix + ".qkv");
            auto packed_qkv     = edgedit::ggml_ext::attention_qkv_split_pack_custom_f32(ctx->ggml_ctx, qkv_projection);
            if (packed_qkv != nullptr && !ggml_backend_supports_op(ctx->backend, packed_qkv)) {
                packed_qkv = nullptr;
            }
            h3_profile_name(packed_qkv, profile_prefix + ".qkv_split_pack");
            std::vector<ggml_tensor*> qkv;
            if (packed_qkv != nullptr) {
                qkv.reserve(3);
                for (int plane = 0; plane < 3; ++plane) {
                    qkv.push_back(ggml_view_4d(ctx->ggml_ctx,
                                               packed_qkv,
                                               head_dim,
                                               num_head,
                                               sequence,
                                               batch_size,
                                               packed_qkv->nb[1],
                                               packed_qkv->nb[2],
                                               packed_qkv->nb[3],
                                               plane * batch_size * packed_qkv->nb[3]));
                }
            } else {
                qkv = ggml_ext_chunk(ctx->ggml_ctx, qkv_projection, 3, 0);
            }
            auto q              = ggml_reshape_4d(ctx->ggml_ctx,
                                                  qkv[0],
                                                  head_dim,
                                                  num_head,
                                                  sequence,
                                                  batch_size);
            h3_profile_name(q, profile_prefix + ".q");
            auto k              = ggml_reshape_4d(ctx->ggml_ctx,
                                                  qkv[1],
                                                  head_dim,
                                                  num_head,
                                                  sequence,
                                                  batch_size);
            h3_profile_name(k, profile_prefix + ".k");
            auto v              = ggml_reshape_4d(ctx->ggml_ctx,
                                                  qkv[2],
                                                  head_dim,
                                                  num_head,
                                                  sequence,
                                                  batch_size);
            h3_profile_name(v, profile_prefix + ".v");
            q                   = ggml_rms_norm(ctx->ggml_ctx, q, 1e-5f);
            h3_profile_name(q, profile_prefix + ".q_norm");
            k                   = ggml_rms_norm(ctx->ggml_ctx, k, 1e-5f);
            h3_profile_name(k, profile_prefix + ".k_norm");
            q                   = apply_partial_rope(ctx->ggml_ctx, q, pe, ctx->backend,
                                                     profile_prefix + ".q_rope_pack");
            h3_profile_name(q, profile_prefix + ".q_rope");
            k                   = apply_partial_rope(ctx->ggml_ctx, k, pe, ctx->backend,
                                                     profile_prefix + ".k_rope_pack");
            h3_profile_name(k, profile_prefix + ".k_rope");
            auto out            = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                         ctx->backend,
                                                         q,
                                                         k,
                                                         v,
                                                         num_head,
                                                         nullptr,
                                                         true,
                                                         ctx->flash_attn_enabled,
                                                         1.0f,
                                                         true,
                                                         false,
                                                         -1,
                                                         -1,
                                                         false,
                                                         true);
            h3_profile_name(out, profile_prefix + ".attn");
            return h3_profile_name(to_out->forward(ctx, out), profile_prefix + ".out");
        }
    };

    struct DecoderFeedForward : public GGMLBlock {
        static constexpr int dim       = 2048;
        static constexpr int kInnerDim = dim * 4;

        DecoderFeedForward() {
            blocks["w1"] = std::make_shared<Linear>(dim, kInnerDim * 2, true);
            blocks["w2"] = std::make_shared<Linear>(kInnerDim, dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             const std::string& profile_prefix = "vae.ff") {
            auto w1   = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2   = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto projected = w1->forward(ctx, x);
            h3_profile_name(projected, profile_prefix + ".w1");
            if (h3_env_flag_enabled_or_default("ED_MINIMAX_H3_VAE_FUSED_SWIGLU", true)) {
                auto activated = ggml_swiglu(ctx->ggml_ctx, projected);
                h3_profile_name(activated, profile_prefix + ".swiglu");
                return h3_profile_name(w2->forward(ctx, activated), profile_prefix + ".out");
            }
            auto gate = ggml_ext_chunk(ctx->ggml_ctx, projected, 2, 0);
            auto activated = ggml_mul(ctx->ggml_ctx,
                                      ggml_silu(ctx->ggml_ctx, gate[0]),
                                      gate[1]);
            h3_profile_name(activated, profile_prefix + ".swiglu_legacy");
            return h3_profile_name(w2->forward(ctx, activated), profile_prefix + ".out");
        }
    };

    struct DecoderBlock : public GGMLBlock {
        static constexpr int dim = 2048;

        DecoderBlock() {
            blocks["norm1"] = std::make_shared<RMSNorm>(dim, 1e-5f);
            blocks["attn"]  = std::make_shared<DecoderAttention>();
            blocks["norm2"] = std::make_shared<RMSNorm>(dim, 1e-5f);
            blocks["ff"]    = std::make_shared<DecoderFeedForward>();
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ED_UNUSED(tensor_storage_map);
            ED_UNUSED(prefix);
            params["scale1"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["scale2"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             const std::string& profile_prefix = "vae.block") {
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto attn  = std::dynamic_pointer_cast<DecoderAttention>(blocks["attn"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto ff    = std::dynamic_pointer_cast<DecoderFeedForward>(blocks["ff"]);
            auto norm1_out = norm1->forward(ctx, x);
            h3_profile_name(norm1_out, profile_prefix + ".norm1");
            auto attn_out = attn->forward(ctx, norm1_out, pe, profile_prefix + ".attn");
            auto attn_scaled = ggml_mul(ctx->ggml_ctx, attn_out, params["scale1"]);
            h3_profile_name(attn_scaled, profile_prefix + ".attn_scaled");
            x          = ggml_add(ctx->ggml_ctx,
                                  x,
                                  attn_scaled);
            h3_profile_name(x, profile_prefix + ".attn_residual");
            auto norm2_out = norm2->forward(ctx, x);
            h3_profile_name(norm2_out, profile_prefix + ".norm2");
            auto ff_out = ff->forward(ctx, norm2_out, profile_prefix + ".ff");
            auto ff_scaled = ggml_mul(ctx->ggml_ctx, ff_out, params["scale2"]);
            h3_profile_name(ff_scaled, profile_prefix + ".ff_scaled");
            return ggml_add(ctx->ggml_ctx,
                            x,
                            ff_scaled);
        }
    };

    struct Decoder : public GGMLBlock {
        static constexpr int dim                 = 2048;
        static constexpr int num_layers          = 36;
        static constexpr int num_register_tokens = 4;
        static constexpr int patch_size          = 16;
        static constexpr int patch_size_t        = 4;

        Decoder() {
            blocks["x_embedder"] = std::make_shared<Linear>(24, dim, true);
            for (int i = 0; i < num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] =
                    std::make_shared<DecoderBlock>();
            }
            blocks["norm_out"] = std::make_shared<LayerNorm>(dim, 1e-5f, true, true);
            blocks["proj_out"] = std::make_shared<Linear>(dim,
                                                          3 * patch_size_t * patch_size * patch_size,
                                                          true,
                                                          true);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ED_UNUSED(tensor_storage_map);
            ED_UNUSED(prefix);
            params["register_tokens"] = ggml_new_tensor_2d(ctx,
                                                           GGML_TYPE_F32,
                                                           dim,
                                                           num_register_tokens);
            params["mask_token"]      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* z,
                             ggml_tensor* pe) {
            int64_t width      = z->ne[0];
            int64_t height     = z->ne[1];
            int64_t num_frames = z->ne[2];
            int64_t batch_size = z->ne[3] / 24;
            GGML_ASSERT(batch_size == 1);

            z                   = ggml_cont(ctx->ggml_ctx,
                                            ggml_ext_torch_permute(ctx->ggml_ctx, z, 3, 0, 1, 2));
            z                   = ggml_reshape_3d(ctx->ggml_ctx,
                                                  z,
                                                  24,
                                                  width * height * num_frames,
                                                  batch_size);
            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto h              = x_embedder->forward(ctx, z);
            int64_t num_patches = h->ne[1];
            h                   = ggml_concat(ctx->ggml_ctx, h, params["register_tokens"], 1);
            auto zero           = ggml_ext_scale(ctx->ggml_ctx,
                                                 ggml_ext_slice(ctx->ggml_ctx, h, 1, 0, 1),
                                                 0.f);
            h                   = ggml_concat(ctx->ggml_ctx, h, zero, 1);

            for (int i = 0; i < num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<DecoderBlock>(
                    blocks["transformer_blocks." + std::to_string(i)]);
                h = block->forward(ctx, h, pe, "vae.b" + std::to_string(i));
                h3_profile_name(h, "vae.b" + std::to_string(i) + ".out");
                sd::ggml_graph_cut::mark_graph_cut(h,
                                                   "minimax_h3_vae.decoder.blocks." + std::to_string(i),
                                                   "hidden_states");
            }

            auto norm_out = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            h             = proj_out->forward(ctx, norm_out->forward(ctx, h));
            h             = ggml_ext_slice(ctx->ggml_ctx, h, 1, 0, num_patches);
            return DiT::unpatchify_3d(ctx->ggml_ctx,
                                      h,
                                      num_frames,
                                      height,
                                      width,
                                      patch_size_t,
                                      patch_size,
                                      patch_size,
                                      true);
        }
    };

    struct MiniMaxH3VideoVAE : public GGMLBlock {
        MiniMaxH3VideoVAE() {
            blocks["encoder"]         = std::make_shared<Encoder>();
            blocks["quant_conv"]      = std::make_shared<Conv3d>(48,
                                                            48,
                                                            std::tuple{1, 1, 1});
            blocks["post_quant_conv"] = std::make_shared<Conv3d>(24,
                                                                 24,
                                                                 std::tuple{1, 1, 1});
            blocks["decoder"]         = std::make_shared<Decoder>();
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx,
                            ggml_tensor* pixels,
                            ggml_tensor* pixel_mean,
                            ggml_tensor* pixel_std,
                            bool return_moments = false) {
            pixels       = ggml_div(ctx->ggml_ctx,
                                    ggml_sub(ctx->ggml_ctx, pixels, pixel_mean),
                                    pixel_std);
            auto encoder = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);
            auto quant   = std::dynamic_pointer_cast<Conv3d>(blocks["quant_conv"]);
            auto moments = quant->forward(ctx, encoder->forward(ctx, pixels));
            if (return_moments) {
                return moments;
            }
            return ggml_ext_slice(ctx->ggml_ctx, moments, 3, 0, 24);
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx,
                            ggml_tensor* latent,
                            ggml_tensor* pe,
                            ggml_tensor* pixel_mean,
                            ggml_tensor* pixel_std) {
            auto post_quant = std::dynamic_pointer_cast<Conv3d>(blocks["post_quant_conv"]);
            auto decoder    = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);
            auto pixels     = decoder->forward(ctx, post_quant->forward(ctx, latent), pe);
            pixels          = ggml_add(ctx->ggml_ctx,
                                       ggml_mul(ctx->ggml_ctx, pixels, pixel_std),
                                       pixel_mean);
            return ggml_clamp(ctx->ggml_ctx, pixels, 0.f, 1.f);
        }
    };

    struct MiniMaxH3VideoVAERunner : public VAE {
        MiniMaxH3VideoVAE model;
        sd::Tensor<float> pixel_mean;
        sd::Tensor<float> pixel_std;
        sd::Tensor<float> latents_mean;
        sd::Tensor<float> latents_std;
        sd::Tensor<float> rope_cache;
        int64_t rope_cache_width  = 0;
        int64_t rope_cache_height = 0;
        int64_t rope_cache_frames = 0;

        MiniMaxH3VideoVAERunner(ggml_backend_t backend,
                                bool offload_params_to_cpu,
                                const String2TensorStorage& tensor_storage_map,
                                const std::string& prefix = "first_stage_model")
            : VAE(VERSION_MINIMAX_H3, backend, offload_params_to_cpu),
              pixel_mean({1, 1, 1, 3}, {0.485f, 0.456f, 0.406f}),
              pixel_std({1, 1, 1, 3}, {0.229f, 0.224f, 0.225f}),
              latents_mean({1, 1, 1, 24},
                           {0.858090341091156f, -0.960659146308899f, 1.066164016723633f, -0.509032547473907f,
                            -0.272758185863495f, -1.367541432380676f, -0.255325496196747f, -0.269075542688370f,
                            -0.537684082984924f, -0.046409729868174f, 0.665737032890320f, 0.196901276707649f,
                            -0.546060800552368f, -0.403534203767776f, -0.236830249428749f, 0.259284526109695f,
                            -0.301339447498322f, 0.211341992020607f, -1.120684862136841f, 0.358193337917328f,
                            -0.042251437902451f, 0.260482996702194f, 0.228640928864479f, 0.705603182315826f}),
              latents_std({1, 1, 1, 24},
                          {1.222377419471741f, 1.276726365089417f, 1.683177471160889f, 1.754945516586304f,
                           1.563621640205383f, 2.194143533706665f, 0.965313792228699f, 1.056988596916199f,
                           0.841948926448822f, 0.772995293140411f, 1.895593762397766f, 0.946841835975647f,
                           0.799680948257446f, 0.449889004230499f, 0.719739973545075f, 0.693629324436188f,
                           2.961095094680786f, 2.769419908523560f, 3.049618482589722f, 2.108805418014527f,
                           3.276226282119751f, 3.162735700607300f, 2.281681299209595f, 2.612784385681153f}) {
            scale_input = false;
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "minimax_h3_video_vae";
        }

        int get_encoder_output_channels(int input_channels) override {
            ED_UNUSED(input_channels);
            return 24;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output,
                                                std::shared_ptr<RNG> rng) override {
            ED_UNUSED(rng);
            return vae_output;
        }

        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            return latents * latents_std + latents_mean;
        }

        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            return (latents - latents_mean) / latents_std;
        }

        static sd::Tensor<float> ensure_video_shape(const sd::Tensor<float>& tensor) {
            if (tensor.dim() == 5) {
                return tensor;
            }
            GGML_ASSERT(tensor.dim() == 4);
            return tensor.reshape({tensor.shape()[0],
                                   tensor.shape()[1],
                                   1,
                                   tensor.shape()[2],
                                   tensor.shape()[3]});
        }

        static ed_tiling_params_t h3_tiling(ed_tiling_params_t params) {
            const char* disable_forced_tiling = std::getenv("ED_MINIMAX_H3_DISABLE_FORCED_VAE_TILING");
            if (disable_forced_tiling != nullptr && disable_forced_tiling[0] != '\0' && disable_forced_tiling[0] != '0') {
                params.enabled = false;
                return params;
            }
            auto tile_size_from_env = [](const char* name, int fallback) {
                const char* value = std::getenv(name);
                if (value == nullptr || value[0] == '\0') {
                    return fallback;
                }
                char* end = nullptr;
                const long parsed = std::strtol(value, &end, 10);
                return end != value && parsed >= 4 && parsed <= 256 ? static_cast<int>(parsed) : fallback;
            };
            params.enabled        = true;
            params.tile_size_x    = tile_size_from_env("ED_MINIMAX_H3_VAE_TILE_SIZE_X", 16);
            params.tile_size_y    = tile_size_from_env("ED_MINIMAX_H3_VAE_TILE_SIZE_Y", 16);
            params.rel_size_x     = 0.0f;
            params.rel_size_y     = 0.0f;
            params.target_overlap = 0.25f;
            return params;
        }

        static sd::Tensor<float> repeat_last_frame(const sd::Tensor<float>& input,
                                                   int64_t count) {
            if (count <= 0) {
                return input;
            }
            std::vector<int64_t> shape = input.shape();
            const int64_t input_frames = shape[2];
            shape[2] += count;
            sd::Tensor<float> result(std::move(shape));
            const int64_t frame_size = input.shape()[0] * input.shape()[1];
            const int64_t input_outer_stride = frame_size * input_frames;
            const int64_t output_outer_stride = frame_size * result.shape()[2];
            const int64_t outer_count = input.shape()[3] * input.shape()[4];
            sd_parallel_for(0, outer_count, 1, [&](int64_t outer) {
                const float* source = input.values().data() + outer * input_outer_stride;
                float* destination = result.values().data() + outer * output_outer_stride;
                std::copy_n(source, input_outer_stride, destination);
                const float* last_frame = source + frame_size * (input_frames - 1);
                for (int64_t frame = 0; frame < count; ++frame) {
                    std::copy_n(last_frame,
                                frame_size,
                                destination + frame_size * (input_frames + frame));
                }
            });
            return result;
        }

        static sd::Tensor<float> blend_temporal(const sd::Tensor<float>& previous,
                                                sd::Tensor<float> current,
                                                int64_t extent) {
            auto output            = std::move(current);
            extent                 = std::min({extent, previous.shape()[2], output.shape()[2]});
            int64_t previous_start = previous.shape()[2] - extent;
            if (h3_env_flag_enabled_or_default("ED_MINIMAX_H3_FAST_TEMPORAL_BLEND", true)) {
                const int64_t frame_size = output.shape()[0] * output.shape()[1];
                const int64_t frames_per_outer = output.shape()[2];
                const int64_t previous_frames_per_outer = previous.shape()[2];
                const int64_t outer_count = output.shape()[3] * output.shape()[4];
                float* output_data = output.values().data();
                const float* previous_data = previous.values().data();
                sd_parallel_for(0, outer_count * extent, 8, [&](int64_t index) {
                    const int64_t t = index % extent;
                    const int64_t outer = index / extent;
                    const float wb = static_cast<float>(t) / extent;
                    const float wa = 1.f - wb;
                    float* output_frame = output_data + frame_size * (t + frames_per_outer * outer);
                    const float* previous_frame = previous_data +
                        frame_size * (previous_start + t + previous_frames_per_outer * outer);
                    for (int64_t pixel = 0; pixel < frame_size; ++pixel) {
                        output_frame[pixel] = previous_frame[pixel] * wa + output_frame[pixel] * wb;
                    }
                });
                return output;
            }
            if (h3_env_flag_enabled_or_default("ED_MINIMAX_H3_PARALLEL_TEMPORAL_COPY", true)) {
                const int64_t channels = output.shape()[3];
                const int64_t batches = output.shape()[4];
                sd_parallel_for(0, batches * channels * extent, 8, [&](int64_t index) {
                    const int64_t t = index % extent;
                    const int64_t channel_batch = index / extent;
                    const int64_t c = channel_batch % channels;
                    const int64_t b = channel_batch / channels;
                    const float wb = static_cast<float>(t) / extent;
                    const float wa = 1.f - wb;
                    for (int64_t h = 0; h < output.shape()[1]; ++h) {
                        for (int64_t w = 0; w < output.shape()[0]; ++w) {
                            output.index(w, h, t, c, b) =
                                previous.index(w, h, previous_start + t, c, b) * wa +
                                output.index(w, h, t, c, b) * wb;
                        }
                    }
                });
                return output;
            }
            for (int64_t b = 0; b < output.shape()[4]; ++b) {
                for (int64_t c = 0; c < output.shape()[3]; ++c) {
                    for (int64_t t = 0; t < extent; ++t) {
                        float wb = static_cast<float>(t) / extent;
                        float wa = 1.f - wb;
                        for (int64_t h = 0; h < output.shape()[1]; ++h) {
                            for (int64_t w = 0; w < output.shape()[0]; ++w) {
                                output.index(w, h, t, c, b) =
                                    previous.index(w, h, previous_start + t, c, b) * wa +
                                    output.index(w, h, t, c, b) * wb;
                            }
                        }
                    }
                }
            }
            return output;
        }

        static sd::Tensor<float> concatenate_temporal_parts(const std::vector<sd::Tensor<float>>& parts) {
            GGML_ASSERT(!parts.empty());
            std::vector<int64_t> shape = parts.front().shape();
            int64_t total_frames       = 0;
            for (const auto& part : parts) {
                GGML_ASSERT(part.dim() == static_cast<int64_t>(shape.size()));
                for (size_t dim = 0; dim < shape.size(); ++dim) {
                    if (dim != 2) {
                        GGML_ASSERT(part.shape()[dim] == shape[dim]);
                    }
                }
                total_frames += part.shape()[2];
            }
            shape[2] = total_frames;
            sd::Tensor<float> result(shape);
            if (h3_env_flag_enabled_or_default("ED_MINIMAX_H3_PARALLEL_TEMPORAL_COPY", true)) {
                const int64_t frame_size = shape[0] * shape[1];
                const int64_t outer_count = shape[3] * shape[4];
                sd_parallel_for(0, outer_count, 8, [&](int64_t outer) {
                    int64_t temporal_offset = 0;
                    for (const auto& part : parts) {
                        const int64_t part_frames = part.shape()[2];
                        const int64_t part_size = frame_size * part_frames;
                        const int64_t source_offset = outer * part_size;
                        const int64_t destination_offset = frame_size * (temporal_offset + total_frames * outer);
                        std::copy_n(part.values().data() + source_offset,
                                    part_size,
                                    result.values().data() + destination_offset);
                        temporal_offset += part_frames;
                    }
                });
                return result;
            }
            int64_t offset = 0;
            for (const auto& part : parts) {
                sd::ops::slice_assign(&result, 2, offset, offset + part.shape()[2], part);
                offset += part.shape()[2];
            }
            return result;
        }

        sd::Tensor<float> encode(int n_threads,
                                 const sd::Tensor<float>& x,
                                 ed_tiling_params_t tiling_params,
                                 bool circular_x = false,
                                 bool circular_y = false) {
            auto input  = ensure_video_shape(x);
            auto tiling = h3_tiling(tiling_params);
            if (input.shape()[2] == 1) {
                auto encoded = VAE::encode(n_threads, input, tiling, circular_x, circular_y);
                if (!encoded.empty() && encoded.shape()[2] > 1) {
                    encoded = sd::ops::slice(encoded,
                                             2,
                                             encoded.shape()[2] - 1,
                                             encoded.shape()[2]);
                }
                return encoded;
            }

            int64_t pad = (-input.shape()[2]) % 17;
            if (pad < 0) {
                pad += 17;
            }
            if (pad > 0) {
                input = repeat_last_frame(input, pad);
            }
            std::vector<sd::Tensor<float>> encoded_parts;
            encoded_parts.reserve(static_cast<size_t>(input.shape()[2] / 17));
            for (int64_t start = 0; start < input.shape()[2]; start += 17) {
                auto chunk   = sd::ops::slice(input, 2, start, start + 17);
                auto encoded = VAE::encode(n_threads, chunk, tiling, circular_x, circular_y);
                if (encoded.empty()) {
                    return {};
                }
                encoded_parts.push_back(std::move(encoded));
            }
            sd::Tensor<float> result = concatenate_temporal_parts(encoded_parts);
            if (result.shape()[2] > 3) {
                result = sd::ops::slice(result, 2, 0, result.shape()[2] - 3);
            }
            return result;
        }

        sd::Tensor<float> encode_moments(int n_threads,
                                         const sd::Tensor<float>& x,
                                         ed_tiling_params_t tiling_params) {
            ScopedEnvFlag flag("ED_MINIMAX_H3_VAE_RETURN_MOMENTS", "1");
            auto input = ensure_video_shape(x);
            auto tiling = h3_tiling(tiling_params);
            if (input.shape()[2] == 1) {
                return VAE::encode(n_threads, input, tiling, false, false);
            }
            int64_t pad = (-input.shape()[2]) % 17;
            if (pad < 0) {
                pad += 17;
            }
            if (pad > 0) {
                input = repeat_last_frame(input, pad);
            }
            sd::Tensor<float> result;
            for (int64_t start = 0; start < input.shape()[2]; start += 17) {
                auto chunk   = sd::ops::slice(input, 2, start, start + 17);
                auto encoded = VAE::encode(n_threads, chunk, tiling, false, false);
                if (encoded.empty()) {
                    return {};
                }
                result = result.empty() ? std::move(encoded)
                                        : sd::ops::concat(result, encoded, 2);
            }
            if (result.shape()[2] > 3) {
                result = sd::ops::slice(result, 2, 0, result.shape()[2] - 3);
            }
            return result;
        }

        sd::Tensor<float> decode(int n_threads,
                                 const sd::Tensor<float>& x,
                                 ed_tiling_params_t tiling_params,
                                 bool decode_video = false,
                                 bool circular_x   = false,
                                 bool circular_y   = false,
                                 bool silent       = false) {
            auto input  = ensure_video_shape(x);
            auto tiling = h3_tiling(tiling_params);
            if (input.shape()[2] == 1) {
                auto decoded = VAE::decode(n_threads,
                                           input,
                                           tiling,
                                           decode_video,
                                           circular_x,
                                           circular_y,
                                           silent);
                if (!decoded.empty() && decoded.shape()[2] > 1) {
                    decoded = sd::ops::slice(decoded,
                                             2,
                                             decoded.shape()[2] - 1,
                                             decoded.shape()[2]);
                }
                return decoded;
            }

            constexpr int64_t tokens_per_chunk  = 5;
            constexpr int64_t token_drop        = 3;
            constexpr int64_t token_overlap     = 2;
            constexpr int64_t frames_per_chunk  = 20;
            constexpr int64_t frame_pre_padding = 3;
            constexpr int64_t frame_overlap     = 5;

            int64_t pseudo_tokens = input.shape()[2] + token_drop;
            int64_t pad_tokens    = (tokens_per_chunk - pseudo_tokens % tokens_per_chunk) % tokens_per_chunk;
            pseudo_tokens += pad_tokens;
            int64_t num_chunks = pseudo_tokens / tokens_per_chunk - 1;
            if (num_chunks < 1) {
                pad_tokens += tokens_per_chunk;
                num_chunks += 1;
            }
            if (pad_tokens > 0) {
                input = repeat_last_frame(input, pad_tokens);
            }

            const bool collect_temporal_parts = h3_env_flag_enabled("ED_MINIMAX_H3_COLLECT_TEMPORAL_PARTS");
            const bool fused_temporal_assembly =
                h3_env_flag_enabled_or_default("ED_MINIMAX_H3_FUSED_TEMPORAL_ASSEMBLY", true);
            sd::Tensor<float> result;
            std::vector<sd::Tensor<float>> temporal_parts;
            if (collect_temporal_parts && !fused_temporal_assembly) {
                temporal_parts.reserve(static_cast<size_t>(num_chunks + 1));
            }
            sd::Tensor<float> overlap;
            sd::Tensor<float> previous_overlap;
            int64_t assembled_frames = 0;
            const bool profile_temporal = h3_env_flag_enabled("ED_PROFILE_MINIMAX_H3_VAE_TEMPORAL");
            int64_t chunk_slice_us = 0;
            int64_t decode_us = 0;
            int64_t output_slice_us = 0;
            int64_t blend_us = 0;
            int64_t collect_us = 0;
            for (int64_t i = 0; i < num_chunks; ++i) {
                int64_t start = i * tokens_per_chunk;
                int64_t end   = std::min(start + tokens_per_chunk + token_overlap,
                                         input.shape()[2]);
                const int64_t chunk_slice_begin = profile_temporal ? ggml_time_us() : 0;
                auto chunk    = sd::ops::slice(input, 2, start, end);
                const int64_t decode_begin = profile_temporal ? ggml_time_us() : 0;
                if (profile_temporal) chunk_slice_us += decode_begin - chunk_slice_begin;
                auto decoded  = VAE::decode(n_threads,
                                            chunk,
                                            tiling,
                                            true,
                                            circular_x,
                                            circular_y,
                                            silent);
                const int64_t output_slice_begin = profile_temporal ? ggml_time_us() : 0;
                if (profile_temporal) decode_us += output_slice_begin - decode_begin;
                if (decoded.empty()) {
                    return {};
                }

                if (fused_temporal_assembly) {
                    const int64_t first_begin = std::min<int64_t>(frame_pre_padding, decoded.shape()[2]);
                    const int64_t first_end = std::min<int64_t>(frames_per_chunk, decoded.shape()[2]);
                    const int64_t first_frames = std::max<int64_t>(0, first_end - first_begin);
                    if (result.empty()) {
                        std::vector<int64_t> shape = decoded.shape();
                        shape[2] = std::max<int64_t>(1, ((x.shape()[2] - 2) / 5) * 17 + 5);
                        result = sd::Tensor<float>(std::move(shape));
                    }
                    const int64_t frame_size = decoded.shape()[0] * decoded.shape()[1];
                    const int64_t outer_count = decoded.shape()[3] * decoded.shape()[4];
                    const int64_t decoded_frames = decoded.shape()[2];
                    const int64_t result_frames = result.shape()[2];
                    const int64_t blend_frames = previous_overlap.empty()
                                                     ? 0
                                                     : std::min<int64_t>(frame_overlap, first_frames);
                    sd_parallel_for(0, outer_count, 8, [&](int64_t outer) {
                        const float* source = decoded.values().data() +
                                              frame_size * (first_begin + decoded_frames * outer);
                        float* destination = result.values().data() +
                                             frame_size * (assembled_frames + result_frames * outer);
                        if (blend_frames > 0) {
                            const int64_t previous_frames = previous_overlap.shape()[2];
                            const float* previous = previous_overlap.values().data() +
                                                    frame_size * (previous_frames * outer);
                            for (int64_t t = 0; t < blend_frames; ++t) {
                                const float wb = static_cast<float>(t) / blend_frames;
                                const float wa = 1.f - wb;
                                for (int64_t pixel = 0; pixel < frame_size; ++pixel) {
                                    destination[t * frame_size + pixel] =
                                        previous[t * frame_size + pixel] * wa +
                                        source[t * frame_size + pixel] * wb;
                                }
                            }
                        }
                        const int64_t copy_frames = first_frames - blend_frames;
                        if (copy_frames > 0) {
                            std::copy_n(source + blend_frames * frame_size,
                                        copy_frames * frame_size,
                                        destination + blend_frames * frame_size);
                        }
                    });
                    assembled_frames += first_frames;

                    if (i == num_chunks - 1 && decoded_frames > frames_per_chunk + frame_pre_padding) {
                        const int64_t tail_begin = frames_per_chunk + frame_pre_padding;
                        const int64_t tail_frames = std::min<int64_t>(
                            decoded_frames - tail_begin, result.shape()[2] - assembled_frames);
                        sd_parallel_for(0, outer_count, 8, [&](int64_t outer) {
                            const float* source = decoded.values().data() +
                                                  frame_size * (tail_begin + decoded_frames * outer);
                            float* destination = result.values().data() +
                                                 frame_size * (assembled_frames + result_frames * outer);
                            std::copy_n(source, tail_frames * frame_size, destination);
                        });
                        assembled_frames += tail_frames;
                    }
                    if (decoded_frames > frames_per_chunk + frame_pre_padding) {
                        previous_overlap = sd::ops::slice(decoded,
                                                          2,
                                                          frames_per_chunk + frame_pre_padding,
                                                          decoded_frames);
                    } else {
                        previous_overlap = {};
                    }
                    if (profile_temporal) {
                        collect_us += ggml_time_us() - output_slice_begin;
                    }
                    continue;
                }

                int64_t first_end = std::min<int64_t>(frames_per_chunk, decoded.shape()[2]);
                auto first        = sd::ops::slice(decoded,
                                                   2,
                                                   std::min<int64_t>(frame_pre_padding, first_end),
                                                   first_end);
                const int64_t blend_begin = profile_temporal ? ggml_time_us() : 0;
                if (profile_temporal) output_slice_us += blend_begin - output_slice_begin;
                if (!overlap.empty()) {
                    first   = h3_env_flag_enabled_or_default("ED_MINIMAX_H3_MOVE_TEMPORAL_BLEND", true)
                                  ? blend_temporal(overlap, std::move(first), frame_overlap)
                                  : blend_temporal(overlap, first, frame_overlap);
                    overlap = {};
                }
                const int64_t collect_begin = profile_temporal ? ggml_time_us() : 0;
                if (profile_temporal) blend_us += collect_begin - blend_begin;
                if (collect_temporal_parts) {
                    temporal_parts.push_back(std::move(first));
                } else {
                    result = result.empty() ? std::move(first)
                                            : sd::ops::concat(result, first, 2);
                }

                if (decoded.shape()[2] > frames_per_chunk + frame_pre_padding) {
                    overlap = sd::ops::slice(decoded,
                                             2,
                                             frames_per_chunk + frame_pre_padding,
                                             decoded.shape()[2]);
                }
                if (i == num_chunks - 1 && !overlap.empty()) {
                    if (collect_temporal_parts) {
                        temporal_parts.push_back(std::move(overlap));
                    } else {
                        result = sd::ops::concat(result, overlap, 2);
                    }
                    overlap = {};
                }
                if (profile_temporal) collect_us += ggml_time_us() - collect_begin;
            }

            const int64_t final_collect_begin = profile_temporal ? ggml_time_us() : 0;
            if (collect_temporal_parts && !fused_temporal_assembly) {
                result = concatenate_temporal_parts(temporal_parts);
            }
            if (profile_temporal) collect_us += ggml_time_us() - final_collect_begin;

            int64_t expected_frames = input.shape()[2] <= 1 ? 1 : ((x.shape()[2] - 2) / 5) * 17 + 5;
            expected_frames         = std::max<int64_t>(1, expected_frames);
            const int64_t trim_begin = profile_temporal ? ggml_time_us() : 0;
            if (result.shape()[2] > expected_frames) {
                result = sd::ops::slice(result, 2, 0, expected_frames);
            }
            if (profile_temporal) {
                LOG_INFO("ED_MINIMAX_H3_VAE_TEMPORAL_PROFILE chunks=%lld chunk-slice_ms=%.3f decode_ms=%.3f output-slice_ms=%.3f blend_ms=%.3f collect_ms=%.3f trim_ms=%.3f",
                         static_cast<long long>(num_chunks),
                         chunk_slice_us / 1000.0,
                         decode_us / 1000.0,
                         output_slice_us / 1000.0,
                         blend_us / 1000.0,
                         collect_us / 1000.0,
                         (ggml_time_us() - trim_begin) / 1000.0);
            }
            return result;
        }

        sd::Tensor<float> build_rope(int64_t width,
                                     int64_t height,
                                     int64_t num_frames) {
            std::vector<std::vector<float>> ids;
            ids.reserve(static_cast<size_t>(width * height * num_frames + 5));
            constexpr float two_pi = 6.28318530717958647692f;
            for (int64_t t = 0; t < num_frames; ++t) {
                float pt = (2.f * ((t + 0.5f) / num_frames) - 1.f) * two_pi;
                for (int64_t h = 0; h < height; ++h) {
                    float ph = (2.f * ((h + 0.5f) / height) - 1.f) * two_pi;
                    for (int64_t w = 0; w < width; ++w) {
                        float pw = (2.f * ((w + 0.5f) / width) - 1.f) * two_pi;
                        ids.push_back({pt, ph, pw});
                    }
                }
            }
            for (int i = 0; i < 5; ++i) {
                ids.push_back({0.f, 0.f, 0.f});
            }
            auto values = Rope::embed_nd(ids,
                                         1,
                                         100.f,
                                         std::vector<int>{16, 16, 16});
            return sd::Tensor<float>({2,
                                      2,
                                      24,
                                      static_cast<int64_t>(ids.size())},
                                     std::move(values));
        }

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& z,
                                   bool decode_graph) override {
            auto input = ensure_video_shape(z);
            if (decode_graph) {
                const bool reuse_rope = h3_env_flag_enabled_or_default("ED_MINIMAX_H3_VAE_REUSE_ROPE_CACHE", true);
                if (!reuse_rope ||
                    rope_cache.empty() ||
                    rope_cache_width != input.shape()[0] ||
                    rope_cache_height != input.shape()[1] ||
                    rope_cache_frames != input.shape()[2]) {
                    rope_cache        = build_rope(input.shape()[0],
                                                   input.shape()[1],
                                                   input.shape()[2]);
                    rope_cache_width  = input.shape()[0];
                    rope_cache_height = input.shape()[1];
                    rope_cache_frames = input.shape()[2];
                }
            }
            auto get_graph = [&]() -> ggml_cgraph* {
                auto value       = make_input(input);
                auto mean        = make_input(pixel_mean);
                auto std         = make_input(pixel_std);
                auto runner_ctx  = get_context();
                const bool use_cann_conv3d = sd_backend_is(runner_ctx.backend, "CANN");
                runner_ctx.conv3d_auto_direct_enabled =
                    use_cann_conv3d ||
                    (!decode_graph &&
                     h3_env_flag_enabled_or_default("ED_MINIMAX_H3_VAE_ENCODER_CUDNN_CONV3D", true) &&
                     should_use_cuda_auto_conv3d());
                runner_ctx.conv3d_force_direct_enabled =
                    use_cann_conv3d ||
                    (runner_ctx.conv3d_auto_direct_enabled &&
                     h3_env_flag_enabled_or_default("ED_MINIMAX_H3_VAE_ENCODER_CUDNN_CONV3D_ALL", true));
                ggml_tensor* out = nullptr;
                if (decode_graph) {
                    auto pe = make_input(rope_cache);
                    out     = model.decode(&runner_ctx, value, pe, mean, std);
                } else {
                    out = model.encode(&runner_ctx,
                                       value,
                                       mean,
                                       std,
                                       h3_env_flag_enabled("ED_MINIMAX_H3_VAE_RETURN_MOMENTS"));
                }
                auto graph = new_graph_custom(H3_VIDEO_VAE_GRAPH_SIZE);
                ggml_build_forward_expand(graph, out);
                return graph;
            };
            if (h3_env_flag_enabled("ED_MINIMAX_H3_VAE_TILE_GRAPH_REUSE")) {
                std::vector<const sd::Tensor<float>*> ordered_inputs = {&input, &pixel_mean, &pixel_std};
                if (decode_graph) {
                    ordered_inputs.push_back(&rope_cache);
                }
                if (auto out = GGMLRunner::compute_reuse<float>(get_graph, ordered_inputs, n_threads, false)) {
                    return restore_trailing_singleton_dims(std::move(*out), 5);
                }
            }
            return restore_trailing_singleton_dims(
                GGMLRunner::compute<float>(get_graph, n_threads, false),
                5);
        }
    };

}  // namespace MiniMaxH3VAE

#endif  // __ED_DIT_MODELS_COMPONENTS_AUTOENCODERS_MINIMAX_H3_VAE_HPP__
