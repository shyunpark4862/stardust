"""The operations targeted after the opportunity scan (1 and 10 threads, numpy).

Run:  .venv/bin/python bench/next_opt.py [--save NAME]
"""

import argparse
import pathlib
import timeit

import numpy as np

import stardust_numpy as sdnp

CASES = [
    ("clip(f, 0.7, 1.2)", "M.clip(f, min=0.7, max=1.2)", "np.clip(f, 0.7, 1.2)", 10),
    ("astype f->i", "M.astype(f, M.int64)", "f.astype(np.int64)", 10),
    ("any(b) (True early)", "M.any(b)", "np.any(b)", 200),
    ("all(b) (False early)", "M.all(b)", "np.all(b)", 200),
    ("any(z) (all False)", "M.any(z)", "np.any(z)", 50),
    ("min(i)", "M.min(i)", "np.min(i)", 20),
    ("max(i)", "M.max(i)", "np.max(i)", 20),
    ("b & b", "b & b", "b & b", 50),
    ("sum(m, axis=0) square", "M.sum(m, axis=0)", "np.sum(m, axis=0)", 20),
    ("mean(m, axis=0)", "M.mean(m, axis=0)", "np.mean(m, axis=0)", 20),
    ("sum(t3, axis=(0, 2))", "M.sum(t3, axis=(0, 2))", "np.sum(t3, axis=(0, 2))", 20),
    ("prod(p)", "M.prod(p)", "np.prod(p)", 10),
    ("prod(pm, axis=1)", "M.prod(pm, axis=1)", "np.prod(pm, axis=1)", 10),
    ("cumulative_sum(f)", "M.cumulative_sum(f)", "np.cumulative_sum(f)", 10),
    ("cumulative_prod(p)", "M.cumulative_prod(p)", "np.cumulative_prod(p)", 10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    args = ap.parse_args()
    r = np.random.default_rng(0)
    n = 1_000_000
    b = r.random(n) > 0.5
    arrays = {
        "f": r.random(n) + 0.5,
        "i": r.integers(-1000, 1000, n),
        "b": b,
        "z": np.zeros(n, bool),
        "m": r.random((1000, 1000)),
        "t3": r.random((100, 100, 100)),
        "p": 1 + (r.random(n) - 0.5) * 1e-6,
        "pm": 1 + (r.random((1000, 1000)) - 0.5) * 1e-6,
    }
    lines = ["| case | 1 thread | 10 threads | numpy |", "|---|---:|---:|---:|"]
    fmt = lambda t: f"{t * 1e6:.1f}µs"  # noqa: E731
    for label, sd_stmt, np_stmt, number in CASES:
        res = []
        for threads in (1, 10):
            sdnp.set_num_threads(threads)
            ns = {k: sdnp.asarray(v) for k, v in arrays.items()} | {"M": sdnp}
            res.append(min(timeit.repeat(sd_stmt, globals=ns, number=number, repeat=7)) / number)
        t_np = min(timeit.repeat(np_stmt, globals=dict(arrays) | {"np": np}, number=number, repeat=7)) / number
        lines.append(f"| {label} | {fmt(res[0])} | {fmt(res[1])} | {fmt(t_np)} |")
    sdnp.set_num_threads(10)
    text = "\n".join(lines) + "\n"
    print(text)
    if args.save:
        (pathlib.Path(__file__).parent / "results" / f"next-{args.save}.md").write_text(text)


if __name__ == "__main__":
    main()
