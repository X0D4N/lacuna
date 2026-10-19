// sp_buffer.h — Lacuna pager-backed ggml backend buffer type.
//
// STATUS (Day 4, 2026-10-19): interface-level reference implementation of
// integrations/llama.cpp/INTEGRATION.md section 1. NOT compiled against a
// real llama.cpp tree on this box (no llama.cpp source, no GPU here).
// Syntax-checked with:
//
//     g++ -std=c++17 -fsyntax-only -I. sp_buffer.cpp
//
// against ggml_backend_shim.h — a minimal LOCAL stand-in declaring the ggml
// backend-buffer structs/functions used here (it is NOT ggml). Before
// production use: compile against a pinned llama.cpp checkout
// (ggml/include/ggml-backend.h, ggml/include/ggml.h) and adapt struct
// layouts and helper names to that revision.
// The runnable, tested contract is lacuna/buffer.py (tests/test_buffer.py);
// this file is its C++ translation.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "ggml_backend_shim.h"  // local stand-in; swap for ggml headers in-tree

#ifdef __cplusplus
extern "C" {
#endif

// Pager callbacks. Implement these against the real pager (C++ port of
// lacuna/pager.py, or the Python pager via ctypes): reserve/release manage
// the pinned arena, touch updates LRU+EWMA without I/O, fault does the
// blocking page read, is_resident is the measured residency query.
// `user` is passed through to every callback.
struct lacuna_pager_ops {
    void *       (*reserve)    (size_t size, void * user);  // pinned arena; null on failure
    void         (*release)    (void * ptr, void * user);
    void         (*touch)      (uint32_t page_id, void * user);
    const void * (*fault)      (uint32_t page_id, size_t * nbytes_out, void * user);  // blocking
    bool         (*is_resident)(uint32_t page_id, void * user);
    void * user;
};

// Buffer type whose allocations come from the pager's pinned arena.
// `ops` (and its `user`) must outlive the type; in practice both are static.
ggml_backend_buffer_type_t lacuna_buffer_type(const struct lacuna_pager_ops * ops);

// Associate a tensor with its pages. page_bytes[i] is the arena payload
// size of page_ids[i] (alignment included); offsets pack in order.
// Models lacuna/buffer.py PagerBackedBuffer.alloc.
void lacuna_buffer_bind_pages(
    ggml_backend_buffer_t buf, struct ggml_tensor * t,
    const uint32_t * page_ids, const size_t * page_bytes, size_t n_pages);

// Blocking residency fence: fault every non-resident page of the tensor
// into its arena slice. Returns pages faulted. Models ensure_resident.
size_t lacuna_buffer_ensure_resident(ggml_backend_buffer_t buf, struct ggml_tensor * t);

// LRU+EWMA update for resident pages, no I/O. Models touch.
void lacuna_buffer_touch(ggml_backend_buffer_t buf, struct ggml_tensor * t);

// Measured per-tensor residency. Models is_resident.
bool lacuna_buffer_is_resident(ggml_backend_buffer_t buf, struct ggml_tensor * t);

// Forget a tensor's page association (arena memory is pager-owned).
void lacuna_buffer_release_tensor(ggml_backend_buffer_t buf, struct ggml_tensor * t);

#ifdef __cplusplus
}
#endif
