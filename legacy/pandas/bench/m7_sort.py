"""M7: sorting vs pandas 3.1.1 (default quicksort, and kind="stable") and Polars.

Ours is always stable. Writes bench/results/m7-sort.md.
"""

import pathlib

import numpy as np
import pandas as pd
import polars as pl

import stardust_pandas as sdpd
from _harness import env_info, fmt_ns, per_call_ns

N = 1_000_000
rng = np.random.default_rng(0)
ints = rng.integers(-(10**9), 10**9, N)
floats = rng.standard_normal(N)
na = rng.random(N) < 0.1
f_na = [None if m else v for m, v in zip(na.tolist(), floats.tolist())]
strs = [f"s{v}" for v in rng.integers(0, 100_000, N).tolist()]
low = rng.integers(0, 100, N)

O = {"i": sdpd.Series(ints), "f": sdpd.Series(floats), "fna": sdpd.Series(f_na), "s": sdpd.Series(strs)}
P = {"i": pd.Series(ints), "f": pd.Series(floats), "fna": pd.Series(pd.array(f_na, dtype="Float64")), "s": pd.Series(strs)}
L = {"i": pl.Series(ints), "f": pl.Series(floats), "fna": pl.Series(f_na), "s": pl.Series(strs)}
df_o = sdpd.DataFrame({"low": low, "s": strs, "f": floats})
df_p = pd.DataFrame({"low": low, "s": strs, "f": floats})
df_l = pl.DataFrame({"low": low, "s": strs, "f": floats})

runs = []
for key, label in (("i", "int64"), ("f", "float64"), ("fna", "float64, 10% NA"), ("s", "string (100k distinct)")):
    runs.append((f"{label}: sort_values", lambda k=key: O[k].sort_values(), lambda k=key: P[k].sort_values(),
                 lambda k=key: P[k].sort_values(kind="stable"), lambda k=key: L[k].sort()))
runs += [
    ("2 keys (int 100 distinct, string)", lambda: df_o.sort_values(["low", "s"]), lambda: df_p.sort_values(["low", "s"]),
     lambda: df_p.sort_values(["low", "s"], kind="stable"), lambda: df_l.sort("low", "s", maintain_order=True)),
    ("int64 descending", lambda: O["i"].sort_values(ascending=False), lambda: P["i"].sort_values(ascending=False),
     lambda: P["i"].sort_values(ascending=False, kind="stable"), lambda: L["i"].sort(descending=True)),
]
lines = ["# M7: sorting", "", "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()), "",
         "1M rows. Ours is a stable sort, single-threaded. Polars: Series.sort (DataFrame.sort with maintain_order=True), possibly threaded.", "",
         "| case | ours | pandas (quicksort) | pandas (stable) | Polars |", "|---|---:|---:|---:|---:|"]
for name, *fns in runs:
    t = [per_call_ns(fn, target_s=0.3, repeats=3) for fn in fns]
    lines.append(f"| {name} | " + " | ".join(fmt_ns(x) for x in t) + " |")
    print(lines[-1], flush=True)
(pathlib.Path(__file__).parent / "results" / "m7-sort.md").write_text("\n".join(lines) + "\n")
