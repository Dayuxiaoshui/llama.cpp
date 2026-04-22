#pragma once

static inline bool ggml_cuda_metax_enabled() {
#ifdef GGML_USE_METAX
    return true;
#else
    return false;
#endif
}
