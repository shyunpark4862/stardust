"""M5: reductions vs pandas 3.1.1 and Polars. Writes bench/results/m5-reduce.md."""

import pathlib

import numpy as np
import pandas as pd
import polars as pl
from _harness import env_info, fmt_ns, per_call_ns

import stardust_pandas as sdpd

N = 1_000_000
rng = np.random.default_rng(0)
f = rng.standard_normal(N)
i = rng.integers(-(10**9), 10**9, N)
na = rng.random(N) < 0.1
f_na = [None if m else v for m, v in zip(na.tolist(), f.tolist())]
strs = [f"s{v}" for v in rng.integers(0, 100_000, N).tolist()]

O = {
    "f": sdpd.Series(f),
    "i": sdpd.Series(i),
    "fna": sdpd.Series(f_na),
    "b": sdpd.Series(f > 0),
    "s": sdpd.Series(strs),
}
P = {
    "f": pd.Series(f),
    "i": pd.Series(i),
    "fna": pd.Series(pd.array(f_na, dtype="Float64")),
    "b": pd.Series(f > 0),
    "s": pd.Series(strs),
}
L = {"f": pl.Series(f), "i": pl.Series(i), "fna": pl.Series(f_na), "b": pl.Series(f > 0), "s": pl.Series(strs)}
df_o = sdpd.DataFrame({"f": O["f"], "i": O["i"], "g": O["f"]})
df_p = pd.DataFrame({"f": P["f"], "i": P["i"], "g": P["f"]})
df_l = pl.DataFrame({"f": L["f"], "i": L["i"], "g": L["f"]})

runs = []
for key, label in (("f", "float64"), ("i", "int64"), ("fna", "float64, 10% NA")):
    for op in ("sum", "mean", "min", "std"):
        runs.append(
            (
                f"{label}: {op}",
                lambda o=O[key], op=op: getattr(o, op)(),
                lambda p=P[key], op=op: getattr(p, op)(),
                lambda l=L[key], op=op: getattr(l, op)(),
            )
        )
runs += [
    ("bool: sum", lambda: O["b"].sum(), lambda: P["b"].sum(), lambda: L["b"].sum()),
    ("string: min", lambda: O["s"].min(), lambda: P["s"].min(), lambda: L["s"].min()),
    ("DataFrame (3 numeric cols): sum", lambda: df_o.sum(), lambda: df_p.sum(), lambda: df_l.sum()),
    (
        "10 rows: sum",
        lambda s=sdpd.Series(list(range(10))): s.sum(),
        lambda s=pd.Series(range(10)): s.sum(),
        lambda s=pl.Series(range(10)): s.sum(),
    ),
]
lines = [
    "# M5: reductions",
    "",
    "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()),
    "",
    "1M rows. Minimum per-call time over 5 batches. Single thread (M11 parallelizes); Polars may use threads.",
    "",
    "| case | ours | pandas | Polars |",
    "|---|---:|---:|---:|",
]
for name, *fns in runs:
    t = [per_call_ns(fn, target_s=0.2, repeats=5) for fn in fns]
    lines.append(f"| {name} | {fmt_ns(t[0])} | {fmt_ns(t[1])} | {fmt_ns(t[2])} |")
    print(lines[-1], flush=True)
(pathlib.Path(__file__).parent / "results" / "m5-reduce.md").write_text("\n".join(lines) + "\n")
