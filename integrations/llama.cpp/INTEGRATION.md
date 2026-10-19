# llama.cpp integration notes

Lacuna is an orchestration + paging layer **above** llama.cpp's execution
core. It does not fork the hot path. This document describes exactly where the
two meet, what is implemented today, and what must move to C++ for production.

## Current state (honest)

- **Day 4 (2026-10-19): the buffer seam is now concrete on both sides.**
  - `lacuna/buffer.py` — `PagerBackedBuffer`: a runnable Python prototype of
    the custom ggml backend buffer type below (alloc / bind+ensure_resident
    fence / touch / LRU eviction / row-masked matvec / per-layer
    `tiled_exact_matvec`). Covered by `tests/test_buffer.py` (11 tests, all
    MEASURED): masked rows are bit-identical to the same row-wise ops on the
    dense assembly, unmasked pages never fault, eviction is loud and counted,
    and the tiled loop holds the exactness guarantee with one layer resident
    at a time. Precision note: a single full-matrix BLAS call can differ from
    row-wise dots by a few ULPs (kernel reduction order) — bytes are identical
    either way; that is documented in the module, not hidden.
  - `integrations/llama.cpp/sp_buffer.{h,cpp}` — the C++ translation:
    `lacuna_buffer_type()` (custom `ggml_backend_buffer_type` backed by pager
    callbacks: reserve/release/touch/fault/is_resident),
    `lacuna_buffer_bind_pages`, `lacuna_buffer_ensure_resident` (blocking
    fence, fault count, loud `SIZE_MAX` on contract violation),
    `lacuna_buffer_touch`, `lacuna_buffer_is_resident`, plus a fenced
    row-masked-matmul sketch. Status: **interface prototype** — syntax-checked
    on this box with `g++ -std=c++17 -fsyntax-only -Wall -Wextra -I.`
    against `ggml_backend_shim.h` (a local stand-in, NOT ggml); NOT compiled
    against a real llama.cpp tree (no source/GPU here). To use: pin a
    llama.cpp checkout, diff the shim against `ggml/include/ggml-backend.h`,
    adapt, and compile there.
- `lacuna/engine.py` ships a **guarded ctypes loader** (`_try_load_libllama`)
  that finds `libllama.so` when present and checks for the core entry points.
  On this dev box (no GPU, no libllama build) it returns `None` and the engine
  runs the deterministic mock target. The mock path is labeled `[mock]` in
  every output — nothing is faked.
- `lacuna/engine.py::run_llama_server` shells out to a real `llama-server`
  binary for the **dense baseline** path. Requires llama.cpp on PATH.
- The pager, oracle, scheduler, and speculative loop are implemented in Python
  + numpy and validated by `tests/` on synthetic inputs.

## Where the pager meets ggml (production design)

The pager owns weight bytes on NVMe; ggml owns compute. The seam is the
**ggml backend buffer**:

1. **Custom buffer type.** Implemented as `sp_buffer.{h,cpp}` (interface
   prototype — see the status banner there) and runnable as
   `lacuna/buffer.py` (`PagerBackedBuffer`, tested). The design sketch below
   is kept for reference; the files are authoritative.

   ```c
   // pseudocode — not compiled here
   struct sp_buffer { page_id_t *pages; size_t n_pages; void *pinned; };

   static void * sp_buffer_alloc(ggml_backend_buffer_t buf, size_t size) {
       // ask the pager (via a C callback into the Python pager, or a port
       // of the hot-set to C++) for the resident bytes of these pages
       return pager_reserve(buf->context->pages, size);
   }
   // on tensor use: pager_touch(page_id)  -> LRU + EWMA update
   // on eviction:   pager_evict(page_id)  -> buffer marked not-resident
   ```

   `ggml_backend_buffer_from_host_ptr` is the existing escape hatch for
   host-owned memory; the custom type above makes residency explicit instead.

2. **Oracle-gated matmuls.** The oracle's union mask selects which FFN rows /
   attention heads participate in the next K steps. In ggml terms this is a
   row-masked `ggml_mul_mat`: rows outside the mask are skipped, so their pages
   are never faulted. The mask is produced in Python (numpy MLP, microseconds)
   and passed down as a bitmask tensor.

3. **Prefetch thread.** The scheduler's prefetch calls become `io_uring`
   submissions (Linux) against the `.lac` file, completing into the pinned
   hot-set while `llama_decode` runs the current step. Ordering fence: a decode
   batch for step *t+1* is not built until its union-mask pages are resident.

## Exact mode in production (per-layer tiling)

The reference engine ensures the *full* page set per decode step, which needs
`hot_budget_bytes` ≥ model size. On real small hardware the required set is
one **layer** (or one expert for MoE): the production loop is

```
for layer in model.layers:                     # or expert shard
    ensure_resident(pages_of(layer))            # blocking; counted
    llama_decode(layer_batch)                   # compute never sees a miss
    prefetch(union_mask_pages_of(layer + 1))    # overlapped with compute
```

`OverlapScheduler(required_pages_for_step=...)` is exactly this hook: pass a
callable returning one layer's page ids and `exact_guarantee_held` is measured
per tile. The oracle's union mask remains the prefetch schedule only — it
never removes a weight from the math, so output stays bit-identical to dense.

## What stays in Python vs what must move to C++

| Stays in Python | Must move to C++ (or C via ctypes callbacks) |
|---|---|
| Oracle training + inference (numpy, µs-scale) | Hot-set LRU + EWMA (per-page-touch hot loop) |
| Speculative draft/verify policy | io_uring submission/completion ring |
| `.lac` conversion (offline) | ggml backend buffer type + row-masked matmul |
| OpenAI server, MCP server, CLI | Pinned-memory allocator backing the buffers |

The Python pager in this repo is the **reference implementation**: correct
eviction, correct stats, correct format. The C++ port must be byte-compatible
with `.lac` (see `lacuna/pager.py` header docs) so converted models work
with both.

## Ollama

Ollama embeds a llama.cpp fork and delegates all inference to it. Wherever
Ollama exposes llama.cpp's C API / server hooks, the Lacuna orchestration
layer applies unchanged — no Ollama-specific code is needed or shipped.
