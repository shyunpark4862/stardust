"""M8: merge and concat vs pandas 3.1.1 and Polars. Writes bench/results/m8-merge.md.

Left: 1M rows; right: 100k rows with unique keys (many-to-one), 90% of left
keys found on the right.
"""

import pathlib

import numpy as np
import pandas as pd
import polars as pl
from _harness import env_info, fmt_ns, per_call_ns

import stardust_pandas as sdpd

N, M = 1_000_000, 100_000
rng = np.random.default_rng(0)
lk = rng.integers(0, int(M / 0.9), N)
rk = rng.permutation(M)
left = {"k": lk, "ks": [f"k{v}" for v in lk.tolist()], "k2": lk % 7, "a": rng.standard_normal(N)}
right = {"k": rk, "ks": [f"k{v}" for v in rk.tolist()], "k2": rk % 7, "b": rng.standard_normal(M)}
Lo, Ro = sdpd.DataFrame(left), sdpd.DataFrame(right)
Lp, Rp = pd.DataFrame(left), pd.DataFrame(right)
Ll, Rl = pl.DataFrame(left), pl.DataFrame(right)

runs = []
for how in ("inner", "left", "outer"):
    runs.append(
        (
            f"int key, {how}",
            lambda how=how: sdpd.merge(Lo[["k", "a"]], Ro[["k", "b"]], how=how, on="k"),
            lambda how=how: pd.merge(Lp[["k", "a"]], Rp[["k", "b"]], how=how, on="k"),
            lambda how=how: Ll.select("k", "a").join(
                Rl.select("k", "b"),
                on="k",
                how="full" if how == "outer" else how,
                maintain_order="left_right",
                coalesce=True,
            ),
        )
    )
runs += [
    (
        "string key, inner",
        lambda: sdpd.merge(Lo[["ks", "a"]], Ro[["ks", "b"]], on="ks"),
        lambda: pd.merge(Lp[["ks", "a"]], Rp[["ks", "b"]], on="ks"),
        lambda: Ll.select("ks", "a").join(Rl.select("ks", "b"), on="ks", maintain_order="left_right"),
    ),
    (
        "2 keys (int, int), inner",
        lambda: sdpd.merge(Lo[["k", "k2", "a"]], Ro[["k", "k2", "b"]], on=["k", "k2"]),
        lambda: pd.merge(Lp[["k", "k2", "a"]], Rp[["k", "k2", "b"]], on=["k", "k2"]),
        lambda: Ll.select("k", "k2", "a").join(Rl.select("k", "k2", "b"), on=["k", "k2"], maintain_order="left_right"),
    ),
    (
        "concat 2 x 1M rows (2 cols)",
        lambda: sdpd.concat([Lo[["k", "a"]], Lo[["k", "a"]]], ignore_index=True),
        lambda: pd.concat([Lp[["k", "a"]], Lp[["k", "a"]]], ignore_index=True),
        lambda: pl.concat([Ll.select("k", "a"), Ll.select("k", "a")]),
    ),
]
lines = [
    "# M8: merge and concat",
    "",
    "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()),
    "",
    "Left 1M rows, right 100k rows with unique keys; ~90% of left keys match. Ours single-threaded.",
    "Polars with maintain_order='left_right' (our order), possibly threaded.",
    "",
    "| case | ours | pandas | Polars |",
    "|---|---:|---:|---:|",
]
for name, *fns in runs:
    t = [per_call_ns(fn, target_s=0.3, repeats=3) for fn in fns]
    lines.append(f"| {name} | {fmt_ns(t[0])} | {fmt_ns(t[1])} | {fmt_ns(t[2])} |")
    print(lines[-1], flush=True)
(pathlib.Path(__file__).parent / "results" / "m8-merge.md").write_text("\n".join(lines) + "\n")
