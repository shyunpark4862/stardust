"""M2: element-wise operators, astype, NA handling and filters, vs pandas and Polars.

pandas runs numpy dtypes for columns without NA and nullable ones (Int64,
boolean) for columns with NA, the closest match to ours. Writes
bench/results/m2-elementwise.md.
"""

import pathlib

import numpy as np
import pandas as pd
import polars as pl

import stardust_pandas as sdpd
from _harness import env_info, fmt_ns, per_call_ns

N = 1_000_000
rng = np.random.default_rng(0)
a_np = rng.integers(-(10**9), 10**9, N)
b_np = rng.integers(1, 10**6, N)
f_np = rng.standard_normal(N)
na_mask = rng.random(N) < 0.1
strs = [f"s{v % 100000}" for v in a_np.tolist()]

ours = {"a": sdpd.Series(a_np), "b": sdpd.Series(b_np), "f": sdpd.Series(f_np),
        "na": sdpd.Series([None if m else v for m, v in zip(na_mask.tolist(), a_np.tolist())]),
        "s": sdpd.Series(strs)}
theirs = {"a": pd.Series(a_np), "b": pd.Series(b_np), "f": pd.Series(f_np),
          "na": pd.Series(pd.array(np.where(na_mask, 0, a_np), dtype="Int64").copy()), "s": pd.Series(strs)}
theirs["na"][na_mask] = pd.NA
polars = {"a": pl.Series(a_np), "b": pl.Series(b_np), "f": pl.Series(f_np),
          "na": pl.Series([None if m else v for m, v in zip(na_mask.tolist(), a_np.tolist())]), "s": pl.Series(strs)}
df_o = sdpd.DataFrame({"a": ours["a"], "f": ours["f"], "s": ours["s"]})
df_p = pd.DataFrame({"a": theirs["a"], "f": theirs["f"], "s": theirs["s"]})
df_l = pl.DataFrame({"a": polars["a"], "f": polars["f"], "s": polars["s"]})
mask_o, mask_p, mask_l = ours["a"] > 0, theirs["a"] > 0, polars["a"] > 0
bo, bp, bl = ours["na"] > 0, theirs["na"] > 0, polars["na"] > 0

cases = [
    ("int64 + int64", lambda d: d["a"] + d["b"]),
    ("int64 + scalar", lambda d: d["a"] + 1),
    ("float64 * float64", lambda d: d["f"] * d["f"]),
    ("int64 // int64", lambda d: d["a"] // d["b"]),
    ("int64 + int64, 10% NA", lambda d: d["na"] + d["a"]),
    ("int64 > 0", lambda d: d["a"] > 0),
    ("int64 > 0.5 (mixed)", lambda d: d["a"] > 0.5),
    ("string == scalar", lambda d: d["s"] == "s5"),
    ("string + scalar", lambda d: d["s"] + "!"),
    ("isna, 10% NA", lambda d: d["na"].isna() if hasattr(d["na"], "isna") else d["na"].is_null()),
    ("fillna(0), 10% NA", lambda d: d["na"].fillna(0) if hasattr(d["na"], "fillna") else d["na"].fill_null(0)),
    ("dropna, 10% NA", lambda d: d["na"].dropna() if hasattr(d["na"], "dropna") else d["na"].drop_nulls()),
    ("float64 astype string", lambda d: d["f"].astype("string") if not isinstance(d["f"], pl.Series) else d["f"].cast(pl.String)),
    ("int64 astype float64", lambda d: d["a"].astype("float64") if not isinstance(d["a"], pl.Series) else d["a"].cast(pl.Float64)),
]
runs = [(name, lambda f=f: f(ours), lambda f=f: f(theirs), lambda f=f: f(polars)) for name, f in cases]
runs += [
    ("bool & bool, 10% NA (Kleene)", lambda: bo & mask_o, lambda: bp & mask_p, lambda: bl & mask_l),
    ("df[mask] (3 cols, ~50% kept)", lambda: df_o[mask_o], lambda: df_p[mask_p], lambda: df_l.filter(mask_l)),
]
small_o, small_p, small_l = sdpd.Series(list(range(10))), pd.Series(range(10)), pl.Series(range(10))
runs.append(("10 rows: s + 1", lambda: small_o + 1, lambda: small_p + 1, lambda: small_l + 1))

lines = ["# M2: element-wise operators", "", "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()), "",
         "1M rows unless stated. Minimum per-call time over 5 batches. Single thread (M11 parallelizes).", "",
         "| case | ours | pandas | Polars |", "|---|---:|---:|---:|"]
for name, *fns in runs:
    t = [per_call_ns(f, target_s=0.2, repeats=5) for f in fns]
    lines.append(f"| {name} | {fmt_ns(t[0])} | {fmt_ns(t[1])} | {fmt_ns(t[2])} |")
    print(lines[-1], flush=True)
(pathlib.Path(__file__).parent / "results" / "m2-elementwise.md").write_text("\n".join(lines) + "\n")
