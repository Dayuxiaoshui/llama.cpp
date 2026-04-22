#pragma once

#include "../ggml-cuda/common.cuh"

struct block_q8_1_mmq {
    union {
        float d4[4];
        half2 ds4[4];
        half  d2s6[8];
    };
    int8_t qs[4*QK8_1];
};

struct block_fp4_mmq {
    uint32_t d4[4];
    int8_t   qs[4 * 32];
};

static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_fp4_mmq)  == sizeof(block_q8_1_mmq),    "Unexpected block_fp4_mmq size");

static inline int get_mmq_y_host(const int cc) {
    GGML_UNUSED(cc);
    return 1;
}

static inline int get_mmq_x_max_host(const int cc) {
    GGML_UNUSED(cc);
    return 0;
}

static inline bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
    GGML_UNUSED(type);
    GGML_UNUSED(cc);
    GGML_UNUSED(ne11);
    GGML_UNUSED(n_experts);
    return false;
}

// Declarations only – the call sites in ggml-cuda.cu are dead-code-eliminated
// because should_use_mmq() above always returns false, so no definitions are needed.
void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
