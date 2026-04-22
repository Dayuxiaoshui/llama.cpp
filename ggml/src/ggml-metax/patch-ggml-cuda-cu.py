#!/usr/bin/env python3
"""Patch ggml-cuda.cu for MetaX: disable custom kernels that require MMA PTX or have memory issues."""

import sys

PATCHES = [
    # Patch 1: In ggml_cuda_mul_mat, force use_mul_mat_f and use_mul_mat_vec_q to false on MetaX
    (
        """    bool use_mul_mat_q     = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    bool any_gpus_with_slow_fp16 = false;""",
        """    bool use_mul_mat_q     = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
#if defined(GGML_USE_METAX)
    use_mul_mat_f = false;     // MetaX: mmf kernel requires MMA PTX
    use_mul_mat_vec_q = false; // MetaX: mmvq kernel has memory alignment issues
#endif

    bool any_gpus_with_slow_fp16 = false;"""
    ),
    # Patch 2: In ggml_cuda_mul_mat_id, skip mul_mat_vec_q direct call on MetaX
    (
        """                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }""",
        """                if (ne2 <= mmvq_mmid_max) {
#if !defined(GGML_USE_METAX)
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
#endif
                }"""
    ),
    # Patch 3: In ggml_cuda_mul_mat_id, skip mmf path on MetaX
    (
        """        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }""",
        """#if !defined(GGML_USE_METAX)
        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
#endif"""
    ),
    # Patch 4: In ggml_cuda_should_fuse_mul_mat_vec_q, return false on MetaX
    (
        """static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;""",
        """static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
#if defined(GGML_USE_METAX)
    GGML_UNUSED(tensor);
    return false; // MetaX: mmvq kernel has memory alignment issues
#else
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;"""
    ),
]

# Also need to close the #else block in should_fuse_mul_mat_vec_q
CLOSE_ELSE_PATCH = (
    """    return use_mul_mat_vec_q;
}

static void ggml_cuda_mul_mat""",
    """    return use_mul_mat_vec_q;
#endif // defined(GGML_USE_METAX)
}

static void ggml_cuda_mul_mat"""
)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input> <output>", file=sys.stderr)
        sys.exit(1)

    in_path, out_path = sys.argv[1:3]

    with open(in_path, "r") as f:
        content = f.read()

    for old, new in PATCHES:
        if old not in content:
            print(f"ERROR: Patch target not found in {in_path}", file=sys.stderr)
            print(f"Looking for:\n{old[:100]}...", file=sys.stderr)
            sys.exit(1)
        content = content.replace(old, new, 1)

    # Close the #else block
    old, new = CLOSE_ELSE_PATCH
    if old not in content:
        print(f"ERROR: Close-else patch target not found", file=sys.stderr)
        sys.exit(1)
    content = content.replace(old, new, 1)

    with open(out_path, "w") as f:
        f.write(content)

    print(f"Patched {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
