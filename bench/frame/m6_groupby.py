"""M6: groupby vs pandas 3.1.1 and Polars. Writes bench/results/m6-groupby.md.

Polars runs multi-threaded; ours is single-threaded until M11.
"""

import pathlib

import numpy as np
import pandas as pd
import polars as pl
from _harness import env_info, fmt_ns, per_call_ns

import stardust_pandas as sdpd

N = 1_000_000
rng = np.random.default_rng(0)
data = {
    "k1000": rng.integers(0, 1000, N),
    "k100k": rng.integers(0, 100_000, N),
    "kstr": [f"g{v}" for v in rng.integers(0, 10_000, N).tolist()],
    "f": rng.standard_normal(N),
    "i": rng.integers(0, 1000, N),
}
o = sdpd.DataFrame(data)
p = pd.DataFrame(data)
l = pl.DataFrame(data)

runs = []
for key in ("k1000", "k100k", "kstr"):
    runs += [
        (
            f"{key}: f sum",
            lambda key=key: o.groupby(key)["f"].sum(),
            lambda key=key: p.groupby(key, as_index=False)["f"].sum(),
            lambda key=key: l.group_by(key).agg(pl.col("f").sum()).sort(key),
        ),
        (
            f"{key}: f mean",
            lambda key=key: o.groupby(key)["f"].mean(),
            lambda key=key: p.groupby(key, as_index=False)["f"].mean(),
            lambda key=key: l.group_by(key).agg(pl.col("f").mean()).sort(key),
        ),
    ]
runs += [
    (
        "k1000: size",
        lambda: o.groupby("k1000").size(),
        lambda: p.groupby("k1000", as_index=False).size(),
        lambda: l.group_by("k1000").len().sort("k1000"),
    ),
    (
        "k1000: i max",
        lambda: o.groupby("k1000")["i"].max(),
        lambda: p.groupby("k1000", as_index=False)["i"].max(),
        lambda: l.group_by("k1000").agg(pl.col("i").max()).sort("k1000"),
    ),
    (
        "k1000: f std",
        lambda: o.groupby("k1000")["f"].std(),
        lambda: p.groupby("k1000", as_index=False)["f"].std(),
        lambda: l.group_by("k1000").agg(pl.col("f").std()).sort("k1000"),
    ),
    (
        "k1000: f first",
        lambda: o.groupby("k1000")["f"].first(),
        lambda: p.groupby("k1000", as_index=False)["f"].first(),
        lambda: l.group_by("k1000", maintain_order=True).agg(pl.col("f").first()).sort("k1000"),
    ),
    (
        "k1000 + kstr: f sum",
        lambda: o.groupby(["k1000", "kstr"])["f"].sum(),
        lambda: p.groupby(["k1000", "kstr"], as_index=False)["f"].sum(),
        lambda: l.group_by("k1000", "kstr").agg(pl.col("f").sum()).sort("k1000", "kstr"),
    ),
    (
        "k1000: agg 3 named",
        lambda: o.groupby("k1000").agg(a=("f", "sum"), b=("i", "max"), c=("f", "count")),
        lambda: p.groupby("k1000", as_index=False).agg(a=("f", "sum"), b=("i", "max"), c=("f", "count")),
        lambda: (
            l.group_by("k1000")
            .agg(pl.col("f").sum().alias("a"), pl.col("i").max().alias("b"), pl.col("f").count().alias("c"))
            .sort("k1000")
        ),
    ),
]
lines = [
    "# M6: groupby",
    "",
    "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()),
    "",
    "1M rows. Keys: k1000 / k100k (ints, 1,000 / 100,000 groups), kstr (10,000 string groups). sort=True for all.",
    "Minimum per-call time over 5 batches. Ours is single-threaded (M11 parallelizes); Polars uses threads.",
    "",
    "| case | ours | pandas | Polars |",
    "|---|---:|---:|---:|",
]
for name, *fns in runs:
    t = [per_call_ns(fn, target_s=0.3, repeats=5) for fn in fns]
    lines.append(f"| {name} | {fmt_ns(t[0])} | {fmt_ns(t[1])} | {fmt_ns(t[2])} |")
    print(lines[-1], flush=True)
(pathlib.Path(__file__).parent / "results" / "m6-groupby.md").write_text("\n".join(lines) + "\n")
