"""Example 1 — the Lacuna Python API, end to end in one file.

Runs offline with the built-in mock backend (clearly labeled "[mock]") so
you can see the whole surface before downloading anything:

  1. auto-tune the hardware policy (zero config),
  2. generate with the one-line Engine API,
  3. build a tiny real .lac store and watch warm-start work — run this
     script TWICE: the second run restores the persisted hot-set and
     faults zero pages.

    python examples/python_api.py
"""

import os
import tempfile

import numpy as np

from lacuna import Engine, auto_tune
from lacuna.hardware import detect_hardware_info
from lacuna.pager import LacWriter, PagedWeightStore
from lacuna.quant import Q4
from lacuna.warmstart import default_hotset_path


def main() -> None:
    # 1) zero-config tuning for THIS machine
    cfg = auto_tune(detect_hardware_info())
    print(f"auto_tune: ram_cap={cfg.ram_cap} GB, vram_cap={cfg.vram_cap} GB, "
          f"page={cfg.page_size // 1024} KiB, K={cfg.k_lookahead}, "
          f"precision={cfg.precision_policy}")

    # 2) one-line generation (mock backend — no model downloaded)
    eng = Engine("mock")
    print("generate:", eng.generate("hello from the example", max_tokens=8))
    eng.close()

    # 3) warm start on a tiny real store kept between runs
    store_path = os.path.join(tempfile.gettempdir(), "lacuna-example.lac")
    if not os.path.exists(store_path):
        rng = np.random.default_rng(0)
        with LacWriter(store_path, page_size=4096) as w:
            for i in range(4):
                w.add_tensor(f"blk.{i}.w", rng.standard_normal((32, 16)),
                             layer=i, precision=Q4)
    cache = default_hotset_path(store_path)
    store = PagedWeightStore(store_path, hot_budget_bytes=1 << 30)
    rep = store.restore_hotset(cache)
    for i in range(store.n_pages):
        store.get_page(i)
    print(f"warm start: restored={rep['restored']} "
          f"({rep['reason']}), faults this run={store.stats['faults']}")
    saved = store.save_hotset(cache)
    print(f"saved hot-set: {saved['saved']} pages -> {cache}")
    store.close()
    print("run this script again — restored should equal the page count "
          "and faults should be 0.")


if __name__ == "__main__":
    main()
