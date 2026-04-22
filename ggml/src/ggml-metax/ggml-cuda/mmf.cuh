#pragma once

#include "../ggml-cuda/common.cuh"

static inline bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {
    GGML_UNUSED(type);
    GGML_UNUSED(cc);
    GGML_UNUSED(warp_size);
    GGML_UNUSED(src0_ne);
    GGML_UNUSED(src0_nb);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(mul_mat_id);
    return false;
}

// Declaration only – call sites in ggml-cuda.cu are DCE'd because should_use_mmf() returns false.
void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
