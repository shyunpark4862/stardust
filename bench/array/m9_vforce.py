"""M9: the 16 float64 functions -- time per element at 1 and 10 threads, and
the largest difference from numpy in ULPs.

Run before and after switching to vForce:
    .venv/bin/python bench/m9_vforce.py --save libm
    .venv/bin/python bench/m9_vforce.py --save vforce
"""

import argparse
import pathlib
import timeit

import numpy as np

import stardust_numpy as sdnp

RESULTS = pathlib.Path(__file__).parent / "results"
N = 1_000_000

# name -> input generator (inside each function's domain, plus a few specials)
UNARY = {
    "exp": lambda r: r.uniform(-700, 700, N), "log": lambda r: r.uniform(1e-300, 1e300, N) * r.random(N),
    "log2": lambda r: r.random(N) * 1e10, "log10": lambda r: r.random(N) * 1e10,
    "sin": lambda r: r.uniform(-1e4, 1e4, N), "asin": lambda r: r.uniform(-1, 1, N),
    "acos": lambda r: r.uniform(-1, 1, N), "atan": lambda r: r.uniform(-1e6, 1e6, N),
    "sinh": lambda r: r.uniform(-700, 700, N), "cosh": lambda r: r.uniform(-700, 700, N),
    "tanh": lambda r: r.uniform(-20, 20, N), "asinh": lambda r: r.uniform(-1e10, 1e10, N),
    "acosh": lambda r: 1 + r.random(N) * 1e6, "atanh": lambda r: r.uniform(-1, 1, N),
}
BINARY = {
    "pow": lambda r: (r.random(N) * 100, r.uniform(-50, 50, N)),
    "atan2": lambda r: (r.uniform(-1e3, 1e3, N), r.uniform(-1e3, 1e3, N)),
}


def ulps(a, b):
    ia, ib = a.view(np.int64).astype(object), b.view(np.int64).astype(object)
    to = lambda v: [x if x >= 0 else -(2**63) - x for x in v]  # noqa: E731
    both_nan = np.isnan(a) & np.isnan(b)
    d = [abs(x - y) for x, y, n in zip(to(ia), to(ib), both_nan) if not n]
    return max(d, default=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    lines = [f"{'function':8} {'1 thread':>12} {'10 threads':>12} {'numpy':>12}   max ULP vs numpy"]
    cases = [(k, (g(rng),)) for k, g in UNARY.items()] + [(k, g(rng)) for k, g in BINARY.items()]
    for name, inputs in cases:
        sd_in = [sdnp.asarray(a) for a in inputs]
        fn_sd, fn_np = getattr(sdnp, name), getattr(np, name)
        t = []
        for threads in (1, 10):
            sdnp.set_num_threads(threads)
            t.append(min(timeit.repeat(lambda: fn_sd(*sd_in), number=3, repeat=5)) / 3 / N)
        t_np = min(timeit.repeat(lambda: fn_np(*inputs), number=3, repeat=5)) / 3 / N
        with np.errstate(all="ignore"):
            worst = ulps(np.asarray(fn_sd(*sd_in))[:200_000], fn_np(*inputs)[:200_000])
        lines.append(f"{name:8} {t[0] * 1e9:9.2f} ns {t[1] * 1e9:9.2f} ns {t_np * 1e9:9.2f} ns   {worst}")
    sdnp.set_num_threads(10)
    text = "\n".join(lines) + "\n"
    print(text)
    if args.save:
        (RESULTS / f"m9-{args.save}.txt").write_text(text)


if __name__ == "__main__":
    main()
