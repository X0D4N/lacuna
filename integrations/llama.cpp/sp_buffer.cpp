// sp_buffer.cpp — Lacuna pager-backed ggml backend buffer type.
//
// See sp_buffer.h for the status banner (interface prototype; syntax-checked
// against ggml_backend_shim.h only). This is the C++ translation of
// lacuna/buffer.py: same operations, same contract.

#include "sp_buffer.h"

#include <cstring>  // memcpy
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

// One arena allocation = one ggml buffer. The pager owns the memory;
// we only track base/size here so ensure_resident can copy into it.
struct buffer_ctx {
    const lacuna_pager_ops * ops;
    void *  base;
    size_t  size;
};

struct page_binding {
    uint32_t page_id;
    size_t   page_bytes;  // arena payload size (alignment included)
    size_t   offset;      // packed offset inside the tensor's arena slice
    bool     resident;
};

struct tensor_ctx {
    ggml_backend_buffer_t        buf;
    std::vector<page_binding>    pages;
};

std::mutex g_mu;
std::unordered_map<ggml_backend_buffer_t, buffer_ctx> g_buffers;
std::unordered_map<struct ggml_tensor *, tensor_ctx>  g_tensors;

constexpr size_t kAlignment = 64;

size_t align_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

buffer_ctx * find_buffer(ggml_backend_buffer_t buf) {
    auto it = g_buffers.find(buf);
    return it == g_buffers.end() ? nullptr : &it->second;
}

tensor_ctx * find_tensor(struct ggml_tensor * t) {
    auto it = g_tensors.find(t);
    return it == g_tensors.end() ? nullptr : &it->second;
}

// ---- ggml_backend_buffer_type_i implementations ----

const char * lacuna_buft_get_name(ggml_backend_buffer_type_t /*buft*/) {
    return "Lacuna";
}

ggml_backend_buffer_t lacuna_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const auto * ops =
        static_cast<const lacuna_pager_ops *>(buft->context);
    if (!ops || !ops->reserve) return nullptr;
    void * ptr = ops->reserve(size, ops->user);
    if (!ptr) return nullptr;  // pager could not provide pinned memory: loud null
    ggml_backend_buffer_t buf = ggml_backend_buffer_from_host_ptr(buft, ptr, size);
    if (!buf) {
        ops->release(ptr, ops->user);
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    g_buffers[buf] = buffer_ctx{ops, ptr, size};
    return buf;
}

size_t lacuna_buft_get_alloc_size(ggml_backend_buffer_type_t /*buft*/, struct ggml_tensor * tensor) {
    return align_up(ggml_nbytes(tensor), kAlignment);
}

size_t lacuna_buft_get_alignment(ggml_backend_buffer_type_t /*buft*/) {
    return kAlignment;
}

bool lacuna_buft_is_host(ggml_backend_buffer_type_t /*buft*/) {
    return true;  // pager-owned host (pinned) memory
}

}  // namespace

extern "C" {

ggml_backend_buffer_type_t lacuna_buffer_type(const struct lacuna_pager_ops * ops) {
    // One static type per ops table; ops/user must be static storage duration.
    static ggml_backend_buffer_type type{};
    static bool initialized = false;
    if (!initialized) {
        type.iface.get_name      = lacuna_buft_get_name;
        type.iface.alloc_buffer  = lacuna_buft_alloc_buffer;
        type.iface.get_alloc_size = lacuna_buft_get_alloc_size;
        type.iface.get_alignment = lacuna_buft_get_alignment;
        type.iface.is_host       = lacuna_buft_is_host;
        type.dev                 = nullptr;  // host buffer: no device
        type.context             = const_cast<lacuna_pager_ops *>(ops);
        initialized = true;
    }
    return &type;
}

void lacuna_buffer_bind_pages(
    ggml_backend_buffer_t buf, struct ggml_tensor * t,
    const uint32_t * page_ids, const size_t * page_bytes, size_t n_pages) {
    std::lock_guard<std::mutex> lock(g_mu);
    tensor_ctx ctx;
    ctx.buf = buf;
    ctx.pages.reserve(n_pages);
    size_t offset = 0;
    for (size_t i = 0; i < n_pages; ++i) {
        ctx.pages.push_back(page_binding{page_ids[i], page_bytes[i], offset, false});
        offset += page_bytes[i];
    }
    g_tensors[t] = std::move(ctx);
}

size_t lacuna_buffer_ensure_resident(ggml_backend_buffer_t buf, struct ggml_tensor * t) {
    // The whole fence runs under the table lock: simple and sound. Faults
    // are blocking, so a production port should move to per-tensor locks or
    // lock-free residency bits; the contract (all-or-counted faults before
    // compute) is unchanged.
    std::lock_guard<std::mutex> lock(g_mu);
    buffer_ctx * bctx = find_buffer(buf);
    tensor_ctx * tctx = find_tensor(t);
    if (!bctx || !tctx) return static_cast<size_t>(-1);  // unknown buffer/tensor: loud
    const lacuna_pager_ops * ops = bctx->ops;
    size_t faulted = 0;
    uint8_t * base = static_cast<uint8_t *>(bctx->base);
    for (auto & p : tctx->pages) {
        if (p.resident) continue;
        size_t n = 0;
        const void * bytes = ops->fault(p.page_id, &n, ops->user);  // blocking
        if (!bytes || n > p.page_bytes) {
            return static_cast<size_t>(-1);  // pager contract violation: loud
        }
        std::memcpy(base + p.offset, bytes, n);
        p.resident = true;
        ++faulted;
    }
    return faulted;
}

void lacuna_buffer_touch(ggml_backend_buffer_t buf, struct ggml_tensor * t) {
    std::lock_guard<std::mutex> lock(g_mu);
    buffer_ctx * bctx = find_buffer(buf);
    tensor_ctx * tctx = find_tensor(t);
    if (!bctx || !tctx) return;
    const lacuna_pager_ops * ops = bctx->ops;
    for (const auto & p : tctx->pages) {
        if (p.resident && ops->touch) ops->touch(p.page_id, ops->user);
    }
}

bool lacuna_buffer_is_resident(ggml_backend_buffer_t buf, struct ggml_tensor * t) {
    std::lock_guard<std::mutex> lock(g_mu);
    tensor_ctx * tctx = find_tensor(t);
    if (!tctx || tctx->buf != buf) return false;
    for (const auto & p : tctx->pages) {
        if (!p.resident) return false;
    }
    return true;
}

void lacuna_buffer_release_tensor(ggml_backend_buffer_t buf, struct ggml_tensor * t) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_tensors.find(t);
    if (it != g_tensors.end() && it->second.buf == buf) {
        g_tensors.erase(it);
    }
    // NOTE: the arena memory itself is pager-owned; if this was the last
    // tensor on the buffer, destroy the ggml buffer via the backend API and
    // let the pager's release callback reclaim the arena (wiring TBD against
    // the real ggml buffer destroy path).
}

}  // extern "C"

// ---- row-masked matmul: INTEGRATION SKETCH ---------------------------------
// The TESTED semantics are lacuna/buffer.py::masked_matvec (bit-identical on
// selected rows, unselected rows' pages never faulted). In ggml terms the
// production form is: for each maximal contiguous run of masked rows, view
// those rows and ggml_mul_mat them against x, then scatter the partial
// results into the output rows. The sketch below shows the loop structure;
// VALIDATE every ggml call (view strides, graph build, scatter) against your
// ggml.h revision before using — this part is design, not compiled code.
/*
static struct ggml_tensor * lacuna_mul_mat_masked_sketch(
    ggml_context_t ctx,
    struct ggml_tensor * w,          // [cols, rows] weight tensor in sp_buffer
    struct ggml_tensor * x,          // [cols] input
    const uint8_t * row_mask,        // rows entries; nonzero = selected
    int64_t rows, int64_t cols,
    size_t row_stride_bytes)
{
    // 1. ensure residency ONLY for pages intersecting selected rows, via
    //    lacuna_buffer_ensure_resident on a page subset bound with
    //    lacuna_buffer_bind_pages (bind_pages takes the subset).
    // 2. for each maximal [r0, r1) run with row_mask set:
    //        struct ggml_tensor * w_run =
    //            ggml_view_2d(ctx, w, cols, r1 - r0, row_stride_bytes,
    //                         r0 * row_stride_bytes);
    //        struct ggml_tensor * y_run = ggml_mul_mat(ctx, w_run, x);
    //        // ... build graph, scatter y_run into output rows [r0, r1)
    // 3. rows outside the mask contribute zero; their pages were never
    //    faulted, so they cost neither NVMe bytes nor time.
    (void)ctx; (void)w; (void)x; (void)row_mask;
    (void)rows; (void)cols; (void)row_stride_bytes;
    return nullptr;
}
*/
