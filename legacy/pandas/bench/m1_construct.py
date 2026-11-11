"""M1: building columns and getting them back out, vs pandas 3.1.1 and Polars.

Semantics differ at the edges (pandas turns None in an int list into a
float64 NaN; Polars into null), so each row notes what each library builds.
Writes bench/results/m1-construct.md.
"""

import pathlib
import random

import numpy as np
import pandas as pd
import polars as pl

import stardust_pandas as sdpd
from _harness import env_info, fmt_ns, per_call_ns

N = 1_000_000
rng = random.Random(0)
ints = [rng.randrange(-(10**9), 10**9) for _ in range(N)]
ints_na = [None if i % 10 == 0 else v for i, v in enumerate(ints)]
floats = [v / 7 for v in ints]
strs = [f"s{v % 100000}" for v in ints]
arr_i64 = np.array(ints, dtype=np.int64)
arr_i32 = arr_i64.astype(np.int32)

cases = [
    ("list[int] -> int64", lambda: sdpd.Series(ints), lambda: pd.Series(ints), lambda: pl.Series(ints)),
    ("list[int|None] (10% None)", lambda: sdpd.Series(ints_na), lambda: pd.Series(ints_na), lambda: pl.Series(ints_na)),
    ("list[float] -> float64", lambda: sdpd.Series(floats), lambda: pd.Series(floats), lambda: pl.Series(floats)),
    ("list[str] -> string", lambda: sdpd.Series(strs), lambda: pd.Series(strs), lambda: pl.Series(strs)),
    ("numpy int64 (copy)", lambda: sdpd.Series(arr_i64), lambda: pd.Series(arr_i64), lambda: pl.Series(arr_i64)),
    ("numpy int32 -> int64", lambda: sdpd.Series(arr_i32), lambda: pd.Series(arr_i32, dtype="int64"), lambda: pl.Series(arr_i32).cast(pl.Int64)),
]
s_ours, s_pd, s_pl = sdpd.Series(ints), pd.Series(ints), pl.Series(ints)
s_str = (sdpd.Series(strs), pd.Series(strs), pl.Series(strs))
df = (
    sdpd.DataFrame({"a": ints, "b": floats, "c": strs}),
    pd.DataFrame({"a": ints, "b": floats, "c": strs}),
    pl.DataFrame({"a": ints, "b": floats, "c": strs}),
)
cases += [
    ("to_numpy() int64", lambda: s_ours.to_numpy(), lambda: s_pd.to_numpy(), lambda: s_pl.to_numpy()),
    ("to_list() int64", lambda: s_ours.to_list(), lambda: s_pd.to_list(), lambda: s_pl.to_list()),
    ("to_list() string", lambda: s_str[0].to_list(), lambda: s_str[1].to_list(), lambda: s_str[2].to_list()),
    ("head() of 1M-row frame", lambda: df[0].head(), lambda: df[1].head(), lambda: df[2].head()),
    ("repr of 1M-row frame", lambda: repr(df[0]), lambda: repr(df[1]), lambda: repr(df[2])),
]
small = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cases += [
    ("Series(10 ints)", lambda: sdpd.Series(small), lambda: pd.Series(small), lambda: pl.Series(small)),
    ("DataFrame(3 cols x 10)", lambda: sdpd.DataFrame({"a": small, "b": small, "c": small}),
     lambda: pd.DataFrame({"a": small, "b": small, "c": small}), lambda: pl.DataFrame({"a": small, "b": small, "c": small})),
]

lines = ["# M1: construction and conversion", "", "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()), "",
         f"1M rows unless stated. Minimum per-call time over 7 batches.", "",
         "| case | ours | pandas | Polars |", "|---|---:|---:|---:|"]
for name, *fns in cases:
    t = [per_call_ns(f, target_s=0.2, repeats=5) for f in fns]
    lines.append(f"| {name} | {fmt_ns(t[0])} | {fmt_ns(t[1])} | {fmt_ns(t[2])} |")
    print(lines[-1], flush=True)
out = pathlib.Path(__file__).parent / "results" / "m1-construct.md"
out.write_text("\n".join(lines) + "\n")
