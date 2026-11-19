"""M10: the single-core optimization steps, measured on one fixed set of cases.

Usage: python m10_steps.py <step label>
Appends a column for the step to bench/results/m10-steps.json and rewrites
bench/results/m10-steps.md (one row per case, one column per step).
"""

import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
from _harness import env_info, fmt_ns, per_call_ns

import stardust_pandas as sdpd

N = 1_000_000
rng = np.random.default_rng(0)
low, high = rng.integers(0, 1000, N), rng.integers(0, N, N)
strs = [f"key{v}" for v in rng.integers(0, 100_000, N).tolist()]
f = rng.standard_normal(N)
na = rng.random(N) < 0.1
ints_na = sdpd.Series([None if m else v for m, v in zip(na.tolist(), high.tolist())])
f_na = sdpd.Series([None if m else v for m, v in zip(na.tolist(), f.tolist())])
b_na = f_na > 0
S = {"low": sdpd.Series(low), "high": sdpd.Series(high), "str": sdpd.Series(strs), "f": sdpd.Series(f)}
df = sdpd.DataFrame({"k": low, "ks": strs, "f": f})
labels = [f"k{v}" for v in rng.permutation(N).tolist()]
lk = rng.integers(0, 111_111, N)
L = sdpd.DataFrame({"k": lk, "a": f})
R = sdpd.DataFrame({"k": rng.permutation(100_000), "b": rng.standard_normal(100_000)})
mask = S["f"] > 0
tmp = pathlib.Path(tempfile.mkdtemp())
csv_f, csv_i, csv_m = tmp / "f.csv", tmp / "i.csv", tmp / "m.csv"
pd.DataFrame({"x": f}).to_csv(csv_f, index=False)
pd.DataFrame({"id": np.arange(N), "n": rng.integers(-1000, 1000, N)}).to_csv(csv_i, index=False)
pd.DataFrame({"id": np.arange(N), "x": f, "name": strs, "flag": f > 0}).to_csv(csv_m, index=False)


def first_loc():
    s = sdpd.Series(low, index=labels)
    return s.loc[labels[7]]


CASES = [
    ("hash: int nunique, 1k keys", lambda: S["low"].nunique()),
    ("hash: int nunique, 630k keys", lambda: S["high"].nunique()),
    ("hash: str nunique, 100k keys", lambda: S["str"].nunique()),
    ("hash: groupby int 1k, sum", lambda: df.groupby("k")["f"].sum()),
    ("hash: groupby str 100k, sum", lambda: df.groupby("ks")["f"].sum()),
    ("hash: Index engine, 1M str labels", first_loc),
    ("hash: merge 1M x 100k int, inner", lambda: sdpd.merge(L, R, on="k")),
    ("NA: isna, 10% NA", lambda: ints_na.isna()),
    ("NA: int + int, 10% NA", lambda: ints_na + ints_na),
    ("NA: bool & bool, 10% NA", lambda: b_na & b_na),
    ("NA: float min, 10% NA", lambda: f_na.min()),
    ("NA: float std, 10% NA", lambda: f_na.std()),
    ("filter: df[mask], 3 cols", lambda: df[mask]),
    ("filter: dropna, 10% NA", lambda: f_na.dropna()),
    ("csv: 1M floats", lambda: sdpd.read_csv(csv_f)),
    ("csv: 1M x 2 ints", lambda: sdpd.read_csv(csv_i)),
    ("csv: 1M x 4 mixed", lambda: sdpd.read_csv(csv_m)),
]

step = sys.argv[1] if len(sys.argv) > 1 else "now"
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None  # case prefixes to run (others keep old numbers)
results = pathlib.Path(__file__).parent / "results"
store = results / "m10-steps.json"
data = json.loads(store.read_text()) if store.exists() else {"steps": [], "times": {}}
if step not in data["steps"]:
    data["steps"].append(step)
for name, fn in CASES:
    if only and not any(name.startswith(p) for p in only):
        continue
    t = per_call_ns(fn, target_s=0.3, repeats=3)
    data["times"].setdefault(name, {})[step] = t
    print(f"{name}: {fmt_ns(t)}", flush=True)
store.write_text(json.dumps(data, indent=1))
lines = [
    "# M10: single-core optimization steps",
    "",
    "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()),
    "",
    "1M rows. Minimum per-call time. Single thread. Each column is the state after that step.",
    "",
    "| case | " + " | ".join(data["steps"]) + " |",
    "|---|" + "---:|" * len(data["steps"]),
]
for name, _ in CASES:
    row = data["times"].get(name, {})
    lines.append(f"| {name} | " + " | ".join(fmt_ns(row[s]) if s in row else "" for s in data["steps"]) + " |")
(results / "m10-steps.md").write_text("\n".join(lines) + "\n")
