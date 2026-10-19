// ggml_backend_shim.h — LOCAL STAND-IN, NOT ggml.
//
// Purpose: lets integrations/llama.cpp/sp_buffer.cpp be syntax-checked on a
// box with no llama.cpp checkout:
//
//     g++ -std=c++17 -fsyntax-only -I. sp_buffer.cpp
//
// It declares ONLY the ggml backend-buffer structs/functions sp_buffer.cpp
// uses, with the same names as llama.cpp's ggml/include/ggml-backend.h.
// Before compiling against a real tree, diff every declaration below
// against ggml/include/ggml-backend.h in your pinned llama.cpp checkout
// and adapt sp_buffer.cpp to that revision. Nothing here links or runs.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define GGML_CALL
#else
#define GGML_CALL
#endif

// ---- minimal ggml_tensor stand-in (real one has src[], op, flags, ...) ----
struct ggml_tensor {
    int      type;      // ggml_type; only element size matters here
    int64_t  ne[4];     // dimensions
    void   * data;      // set by buffer init_tensor / from_host_ptr
    char     name[64];
};

typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct ggml_backend_buffer      * ggml_backend_buffer_t;
typedef struct ggml_context             * ggml_context_t;

struct ggml_backend_buffer_type_i {
    const char *          (*GGML_CALL get_name)     (ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*GGML_CALL alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size);
    size_t                (*GGML_CALL get_alloc_size)(ggml_backend_buffer_type_t buft, struct ggml_tensor * tensor);
    size_t                (*GGML_CALL get_alignment)(ggml_backend_buffer_type_t buft);
    bool                  (*GGML_CALL is_host)      (ggml_backend_buffer_type_t buft);
};

struct ggml_backend_buffer_type {
    struct ggml_backend_buffer_type_i  iface;
    void * dev;      // ggml_backend_dev_t in real ggml
    void * context;  // we store our lacuna_buffer_context here
};

struct ggml_backend_buffer_i {
    const char * (*GGML_CALL get_name)   (ggml_backend_buffer_t buffer);
    void         (*GGML_CALL free_buffer)(ggml_backend_buffer_t buffer);
    void *       (*GGML_CALL get_base)   (ggml_backend_buffer_t buffer);
};

struct ggml_backend_buffer {
    struct ggml_backend_buffer_i  iface;
    ggml_backend_buffer_type_t    buft;
    void * context;  // we store our lacuna_buffer_meta here
    size_t size;
};

// ---- functions sp_buffer.cpp uses (verify against real ggml-backend.h) ----
#ifdef __cplusplus
extern "C" {
#endif

// Wrap caller-owned host memory in a buffer of this type (the escape hatch
// integrations/llama.cpp/INTEGRATION.md refers to). If your checkout lacks
// it, fill ggml_backend_buffer manually instead.
ggml_backend_buffer_t ggml_backend_buffer_from_host_ptr(
    ggml_backend_buffer_type_t buft, void * ptr, size_t size);

// Total byte size of a tensor (type size * ne[0]*ne[1]*...).
size_t ggml_nbytes(const struct ggml_tensor * tensor);

// Graph helpers used ONLY by the masked-matmul sketch at the bottom of
// sp_buffer.cpp (not by the buffer type itself).
struct ggml_tensor * ggml_view_2d(
    ggml_context_t ctx, struct ggml_tensor * a,
    int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
struct ggml_tensor * ggml_mul_mat(
    ggml_context_t ctx, struct ggml_tensor * a, struct ggml_tensor * b);

#ifdef __cplusplus
}
#endif
