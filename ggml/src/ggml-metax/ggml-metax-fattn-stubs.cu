#include "ggml-cuda/fattn.cuh"

#ifdef GGML_USE_METAX
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("ggml_cuda_flash_attn_ext is not enabled for MetaX");
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}
#endif

