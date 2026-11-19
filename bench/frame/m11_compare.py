"""M11: the main cases against pandas and Polars, ours with all threads.

Writes bench/results/m11-compare.md.
"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import polars as pl

import stardust_pandas as sdpd
from _harness import env_info, fmt_ns, per_call_ns

N = 1_000_000
rng = np.random.default_rng(0)
low, high = rng.integers(0, 1000, N), rng.integers(0, N, N)
strs = [f"key{v}" for v in rng.integers(0, 100_000, N).tolist()]
f, g = rng.standard_normal(N), rng.standard_normal(N)
lk, rk, rb = rng.integers(0, 111_111, N), rng.permutation(100_000), rng.standard_normal(100_000)
tmp = pathlib.Path(tempfile.mkdtemp())
csv_f, csv_m = tmp / "f.csv", tmp / "m.csv"
pd.DataFrame({"x": f}).to_csv(csv_f, index=False)
pd.DataFrame({"id": np.arange(N), "x": f, "name": strs, "flag": f > 0}).to_csv(csv_m, index=False)

ours = {
    "f": sdpd.Series(f),
    "g": sdpd.Series(g),
    "low": sdpd.Series(low),
    "high": sdpd.Series(high),
    "str": sdpd.Series(strs),
    "df": sdpd.DataFrame({"k": low, "kh": high, "ks": strs, "f": f}),
    "L": sdpd.DataFrame({"k": lk, "a": f}),
    "R": sdpd.DataFrame({"k": rk, "b": rb}),
}
pds = {
    "f": pd.Series(f),
    "g": pd.Series(g),
    "low": pd.Series(low),
    "high": pd.Series(high),
    "str": pd.Series(strs),
    "df": pd.DataFrame({"k": low, "kh": high, "ks": strs, "f": f}),
    "L": pd.DataFrame({"k": lk, "a": f}),
    "R": pd.DataFrame({"k": rk, "b": rb}),
}
pls = {
    "f": pl.Series(f),
    "g": pl.Series(g),
    "low": pl.Series(low),
    "high": pl.Series(high),
    "str": pl.Series(strs),
    "df": pl.DataFrame({"k": low, "kh": high, "ks": strs, "f": f}),
    "L": pl.DataFrame({"k": lk, "a": f}),
    "R": pl.DataFrame({"k": rk, "b": rb}),
}

CASES = [
    ("float + float", lambda o: o["f"] + o["g"], lambda p: p["f"] + p["g"], lambda q: q["f"] + q["g"]),
    ("float sum", lambda o: o["f"].sum(), lambda p: p["f"].sum(), lambda q: q["f"].sum()),
    ("mask filter", lambda o: o["f"][o["f"] > 0], lambda p: p["f"][p["f"] > 0], lambda q: q["f"].filter(q["f"] > 0)),
    ("nunique int, 1k keys", lambda o: o["low"].nunique(), lambda p: p["low"].nunique(), lambda q: q["low"].n_unique()),
    (
        "nunique int, 630k keys",
        lambda o: o["high"].nunique(),
        lambda p: p["high"].nunique(),
        lambda q: q["high"].n_unique(),
    ),
    ("nunique string", lambda o: o["str"].nunique(), lambda p: p["str"].nunique(), lambda q: q["str"].n_unique()),
    (
        "groupby int 1k sum",
        lambda o: o["df"].groupby("k")["f"].sum(),
        lambda p: p["df"].groupby("k")["f"].sum(),
        lambda q: q["df"].group_by("k").agg(pl.col("f").sum()).sort("k"),
    ),
    (
        "groupby int 630k sum",
        lambda o: o["df"].groupby("kh")["f"].sum(),
        lambda p: p["df"].groupby("kh")["f"].sum(),
        lambda q: q["df"].group_by("kh").agg(pl.col("f").sum()).sort("kh"),
    ),
    (
        "groupby string sum",
        lambda o: o["df"].groupby("ks")["f"].sum(),
        lambda p: p["df"].groupby("ks")["f"].sum(),
        lambda q: q["df"].group_by("ks").agg(pl.col("f").sum()).sort("ks"),
    ),
    (
        "groupby int + string sum",
        lambda o: o["df"].groupby(["k", "ks"])["f"].sum(),
        lambda p: p["df"].groupby(["k", "ks"])["f"].sum(),
        lambda q: q["df"].group_by(["k", "ks"]).agg(pl.col("f").sum()).sort(["k", "ks"]),
    ),
    (
        "merge 1M x 100k inner",
        lambda o: sdpd.merge(o["L"], o["R"], on="k"),
        lambda p: p["L"].merge(p["R"], on="k"),
        lambda q: q["L"].join(q["R"], on="k", maintain_order="left"),
    ),
    (
        "sort int64",
        lambda o: o["high"].sort_values(),
        lambda p: p["high"].sort_values(kind="stable"),
        lambda q: q["high"].sort(),
    ),
    (
        "sort string",
        lambda o: o["str"].sort_values(),
        lambda p: p["str"].sort_values(kind="stable"),
        lambda q: q["str"].sort(),
    ),
    ("read_csv floats", lambda o: sdpd.read_csv(csv_f), lambda p: pd.read_csv(csv_f), lambda q: pl.read_csv(csv_f)),
    ("read_csv mixed", lambda o: sdpd.read_csv(csv_m), lambda p: pd.read_csv(csv_m), lambda q: pl.read_csv(csv_m)),
]

lines = [
    "# M11: against pandas and Polars",
    "",
    "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()),
    "",
    f"1M rows. Minimum per-call time. Ours with {sdpd.get_num_threads()} threads; pandas single-threaded; Polars its own pool.",
    "",
    "| case | ours | pandas | Polars |",
    "|---|---:|---:|---:|",
]
for name, a, b, c in CASES:
    t = [
        per_call_ns(lambda: a(ours), target_s=0.3, repeats=3),
        per_call_ns(lambda: b(pds), target_s=0.3, repeats=3),
        per_call_ns(lambda: c(pls), target_s=0.3, repeats=3),
    ]
    lines.append(f"| {name} | " + " | ".join(fmt_ns(x) for x in t) + " |")
    print(lines[-1], flush=True)
(pathlib.Path(__file__).parent / "results" / "m11-compare.md").write_text("\n".join(lines) + "\n")
