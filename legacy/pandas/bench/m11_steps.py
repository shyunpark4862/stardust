"""M11: the multithreading steps, measured on one fixed set of cases.

Usage: python m11_steps.py <step label> [threads] [case prefixes]
threads defaults to the performance-core count. The column is
"<step>@<threads>". Appends to bench/results/m11-steps.json and rewrites
bench/results/m11-steps.md (one row per case, one column per step).
"""

import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

import stardust_pandas as sdpd
from _harness import env_info, fmt_ns, per_call_ns

N = 1_000_000
rng = np.random.default_rng(0)
low, high = rng.integers(0, 1000, N), rng.integers(0, N, N)
strs = [f"key{v}" for v in rng.integers(0, 100_000, N).tolist()]
f = rng.standard_normal(N)
g = rng.standard_normal(N)
na = rng.random(N) < 0.1
ints_na = sdpd.Series([None if m else v for m, v in zip(na.tolist(), high.tolist())])
f_na = sdpd.Series([None if m else v for m, v in zip(na.tolist(), f.tolist())])
S = {"low": sdpd.Series(low), "high": sdpd.Series(high), "str": sdpd.Series(strs), "f": sdpd.Series(f), "g": sdpd.Series(g)}
df = sdpd.DataFrame({"k": low, "kh": high, "ks": strs, "f": f})
L = sdpd.DataFrame({"k": rng.integers(0, 111_111, N), "a": f})
R = sdpd.DataFrame({"k": rng.permutation(100_000), "b": rng.standard_normal(100_000)})
mask = S["f"] > 0
tmp = pathlib.Path(tempfile.mkdtemp())
csv_f, csv_i, csv_m = tmp / "f.csv", tmp / "i.csv", tmp / "m.csv"
pd.DataFrame({"x": f}).to_csv(csv_f, index=False)
pd.DataFrame({"id": np.arange(N), "n": rng.integers(-1000, 1000, N)}).to_csv(csv_i, index=False)
pd.DataFrame({"id": np.arange(N), "x": f, "name": strs, "flag": f > 0}).to_csv(csv_m, index=False)

CASES = [
    ("elem: float + float", lambda: S["f"] + S["g"]),
    ("elem: int + int, 10% NA", lambda: ints_na + ints_na),
    ("elem: float > 0", lambda: S["f"] > 0),
    ("elem: isna, 10% NA", lambda: ints_na.isna()),
    ("elem: astype float -> int", lambda: S["f"].astype("int64")),
    ("filter: df[mask], 4 cols", lambda: df[mask]),
    ("reduce: float sum", lambda: S["f"].sum()),
    ("reduce: float std, 10% NA", lambda: f_na.std()),
    ("reduce: float min", lambda: S["f"].min()),
    ("hash: int nunique, 1k keys", lambda: S["low"].nunique()),
    ("hash: int nunique, 630k keys", lambda: S["high"].nunique()),
    ("hash: str nunique, 100k keys", lambda: S["str"].nunique()),
    ("groupby: int 1k, sum", lambda: df.groupby("k")["f"].sum()),
    ("groupby: int 630k, sum", lambda: df.groupby("kh")["f"].sum()),
    ("groupby: str 100k, sum", lambda: df.groupby("ks")["f"].sum()),
    ("groupby: int + str, sum", lambda: df.groupby(["k", "ks"])["f"].sum()),
    ("merge: 1M x 100k int, inner", lambda: sdpd.merge(L, R, on="k")),
    ("sort: int64", lambda: S["high"].sort_values()),
    ("sort: float64", lambda: S["f"].sort_values()),
    ("sort: string 100k", lambda: S["str"].sort_values()),
    ("csv: 1M floats", lambda: sdpd.read_csv(csv_f)),
    ("csv: 1M x 2 ints", lambda: sdpd.read_csv(csv_i)),
    ("csv: 1M x 4 mixed", lambda: sdpd.read_csv(csv_m)),
]

step = sys.argv[1] if len(sys.argv) > 1 else "now"
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 10
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
sdpd.set_num_threads(threads)
col = f"{step}@{threads}"
results = pathlib.Path(__file__).parent / "results"
store = results / "m11-steps.json"
data = json.loads(store.read_text()) if store.exists() else {"steps": [], "times": {}}
if col not in data["steps"]:
    data["steps"].append(col)
for name, fn in CASES:
    if only and not any(name.startswith(p) for p in only):
        continue
    t = per_call_ns(fn, target_s=0.3, repeats=3)
    data["times"].setdefault(name, {})[col] = t
    print(f"{name}: {fmt_ns(t)}", flush=True)
store.write_text(json.dumps(data, indent=1))
lines = ["# M11: multithreading steps", "", "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()), "",
         "1M rows. Minimum per-call time. Column = step@threads.", "",
         "| case | " + " | ".join(data["steps"]) + " |", "|---|" + "---:|" * len(data["steps"])]
for name, _ in CASES:
    row = data["times"].get(name, {})
    lines.append(f"| {name} | " + " | ".join(fmt_ns(row[s]) if s in row else "" for s in data["steps"]) + " |")
(results / "m11-steps.md").write_text("\n".join(lines) + "\n")
