"""M2: what do axis reordering and coalescing buy?

The same copy kernel (a plain strided scalar loop) runs under three iterator
plans:
  naive   : index order, no simplification (last axis innermost, always)
  reorder : extent-1 axes dropped + axes sorted so the innermost walks memory
  full    : reorder + coalescing of contiguous neighbouring axes
numpy's own copy(order='K') is shown for reference (it has a contiguous
memcpy fast path, which ours only gets in M7).

Run:  .venv/bin/python bench/m2_iterator.py
"""

import time

import numpy as np

import stardust_numpy as sdnp

copy_with = sdnp._core._copy_with_mode
plan = sdnp._core._iter_plan


def best_ms(fn, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def filled(*shape):
    # Real data: never-written calloc pages all map to one shared zero page,
    # which would make reads look impossibly fast.
    return np.random.default_rng(0).random(shape)


CASES = [
    ("C-contiguous (4000, 4000)", lambda: filled(4000, 4000)),
    ("transposed (4000, 4000).T", lambda: filled(4000, 4000).T),
    ("3-d permuted (256,256,256)", lambda: filled(256, 256, 256).transpose(2, 0, 1)),
    ("every 2nd column (4000, 8000)", lambda: filled(4000, 8000)[:, ::2]),
    ("many short rows (4_000_000, 4)", lambda: filled(4_000_000, 4)),
]


def main():
    print(f"{'case':32} {'naive':>9} {'reorder':>9} {'full':>9} {'numpy':>9}   full plan")
    for name, make in CASES:
        src_np = make()
        src = sdnp.asarray(src_np)  # zero-copy: keeps numpy's strides
        times = {m: best_ms(lambda m=m: copy_with(src, m)) for m in ("naive", "reorder", "full")}
        t_np = best_ms(lambda: src_np.copy(order="K"))
        p = plan([src, src.copy()])
        print(
            f"{name:32} {times['naive']:7.1f}ms {times['reorder']:7.1f}ms {times['full']:7.1f}ms "
            f"{t_np:7.1f}ms   shape={p['shape']}"
        )


if __name__ == "__main__":
    main()
