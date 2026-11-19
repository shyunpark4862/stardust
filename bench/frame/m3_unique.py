"""M3: factorize-based unique family vs pandas 3.1.1 and Polars.

Low cardinality (1,000 keys) keeps the hash table in cache; high
cardinality (~630k keys from 1M rows) makes nearly every probe a cache miss.
Writes bench/results/m3-unique.md.
"""

import pathlib

import numpy as np
import pandas as pd
import polars as pl

import stardust_pandas as sdpd
from _harness import env_info, fmt_ns, per_call_ns

N = 1_000_000
rng = np.random.default_rng(0)
low = rng.integers(0, 1000, N)
high = rng.integers(0, N, N)
flt = rng.integers(0, 1000, N) / 7.0
strs = [f"key{v}" for v in rng.integers(0, 100_000, N).tolist()]

data = {"low": low, "high": high, "float": flt}
O = {k: sdpd.Series(v) for k, v in data.items()} | {"str": sdpd.Series(strs)}
P = {k: pd.Series(v) for k, v in data.items()} | {"str": pd.Series(strs)}
L = {k: pl.Series(v) for k, v in data.items()} | {"str": pl.Series(strs)}
df_o = sdpd.DataFrame({"a": O["low"], "b": O["str"]})
df_p = pd.DataFrame({"a": P["low"], "b": P["str"]})
df_l = pl.DataFrame({"a": L["low"], "b": L["str"]})

runs = []
for key in ("low", "high", "float", "str"):
    o, p, l = O[key], P[key], L[key]
    runs += [
        (f"{key}: nunique", lambda o=o: o.nunique(), lambda p=p: p.nunique(), lambda l=l: l.n_unique()),
        (f"{key}: unique", lambda o=o: o.unique(), lambda p=p: p.unique(), lambda l=l: l.unique(maintain_order=True)),
        (
            f"{key}: value_counts",
            lambda o=o: o.value_counts(),
            lambda p=p: p.value_counts(),
            lambda l=l: l.value_counts(sort=True),
        ),
        (
            f"{key}: duplicated",
            lambda o=o: o.duplicated(),
            lambda p=p: p.duplicated(),
            lambda l=l: ~l.is_first_distinct(),
        ),
    ]
runs += [
    (
        "2 keys (int, str): duplicated",
        lambda: df_o.duplicated(),
        lambda: df_p.duplicated(),
        lambda: df_l.select(~pl.struct(pl.all()).is_first_distinct()),
    ),
    (
        "2 keys (int, str): drop_duplicates",
        lambda: df_o.drop_duplicates(),
        lambda: df_p.drop_duplicates(),
        lambda: df_l.unique(maintain_order=True),
    ),
]

lines = [
    "# M3: unique family",
    "",
    "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()),
    "",
    "1M rows. low: 1,000 distinct ints; high: ~630k distinct ints; float: 1,000 distinct; str: 100,000 distinct.",
    "Minimum per-call time over 5 batches. Single thread. Polars runs multi-threaded where it can.",
    "",
    "| case | ours | pandas | Polars |",
    "|---|---:|---:|---:|",
]
for name, *fns in runs:
    t = [per_call_ns(f, target_s=0.2, repeats=5) for f in fns]
    lines.append(f"| {name} | {fmt_ns(t[0])} | {fmt_ns(t[1])} | {fmt_ns(t[2])} |")
    print(lines[-1], flush=True)
(pathlib.Path(__file__).parent / "results" / "m3-unique.md").write_text("\n".join(lines) + "\n")
