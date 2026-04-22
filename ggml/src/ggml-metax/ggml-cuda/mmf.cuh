#pragma once

#include "../ggml-cuda/common.cuh"

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, const int src1_ncols, bool mul_mat_id);
