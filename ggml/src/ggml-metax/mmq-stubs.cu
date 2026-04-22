#include "ggml-cuda/common.cuh"
#include "ggml.h"

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
    GGML_UNUSED(type);
    GGML_UNUSED(cc);
    GGML_UNUSED(ne11);
    GGML_UNUSED(n_experts);
    return false;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    GGML_ABORT("ggml_cuda_mul_mat_q is not enabled for MetaX");
}

void ggml_cuda_op_mul_mat_q(
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
