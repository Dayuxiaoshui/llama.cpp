#pragma once

#include <cstddef>
#include <cstdint>

#ifdef GGML_USE_METAX
// ggml-cuda.cu may still reference block_q8_1_mmq in code paths that are
// runtime-disabled for MetaX. Provide a compile-time compatible alias so we
// can keep MetaX-specific changes isolated in ggml-metax.
using block_q8_1_mmq = block_q8_1;
#endif

static inline bool ggml_cuda_metax_enabled() {
#ifdef GGML_USE_METAX
    return true;
#else
    return false;
#endif
}

static inline bool ggml_cuda_backend_supports_cooperative_launch() {
    return !ggml_cuda_metax_enabled();
}

static inline int64_t ggml_cuda_row_rounding_for_backend(int cc) {
#ifdef GGML_USE_METAX
    GGML_UNUSED(cc);
    return 1;
#else
    return (int64_t) get_mmq_y_host(cc);
#endif
}

static inline bool ggml_cuda_backend_enable_mmq_mmf() {
    return !ggml_cuda_metax_enabled();
}

static inline bool ggml_cuda_is_mmq_quantizer(quantize_cuda_t quantize_src1) {
#ifdef GGML_USE_METAX
    GGML_UNUSED(quantize_src1);
    return false;
#else
    return quantize_src1 == quantize_mmq_q8_1_cuda;
#endif
}

static inline size_t ggml_cuda_mmq_extra_src1_size(int cc) {
#ifdef GGML_USE_METAX
    GGML_UNUSED(cc);
    return 0;
#else
    return get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
#endif
}

static inline size_t ggml_cuda_mmq_src1_col_offset(int64_t src1_col_0) {
#ifdef GGML_USE_METAX
    GGML_UNUSED(src1_col_0);
    return 0;
#else
    return src1_col_0 * sizeof(block_q8_1_mmq);
#endif
}

#ifdef GGML_USE_METAX
static inline bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
    GGML_UNUSED(type);
    GGML_UNUSED(cc);
    GGML_UNUSED(ne11);
    GGML_UNUSED(n_experts);
    return false;
}

static inline bool ggml_cuda_should_use_mmf(
        enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {
    GGML_UNUSED(type);
    GGML_UNUSED(cc);
    GGML_UNUSED(warp_size);
    GGML_UNUSED(src0_ne);
    GGML_UNUSED(src0_nb);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(mul_mat_id);
    return false;
}

static inline void ggml_cuda_mul_mat_f(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    GGML_ABORT("ggml_cuda_mul_mat_f is not enabled for MetaX");
}

static inline void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    GGML_ABORT("ggml_cuda_mul_mat_q is not enabled for MetaX");
}

static inline void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src0_dd_i);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(dst_dd_i);
    GGML_UNUSED(row_low);
    GGML_UNUSED(row_high);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
    GGML_UNUSED(stream);
    GGML_ABORT("ggml_cuda_op_mul_mat_q is not enabled for MetaX");
}
#endif

