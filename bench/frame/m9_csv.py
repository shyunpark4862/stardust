"""M9: read_csv vs pandas 3.1.1 (C engine) and Polars. Writes bench/results/m9-csv.md.

1M rows x 5 columns (int, float, string, bool, int) written to a scratch
file. Ours is single-threaded; Polars reads in parallel.
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
frame = pd.DataFrame(
    {
        "id": np.arange(N),
        "x": rng.standard_normal(N),
        "name": [f"user{v}" for v in rng.integers(0, 100_000, N).tolist()],
        "flag": rng.random(N) < 0.5,
        "n": rng.integers(-1000, 1000, N),
    }
)
tmp = pathlib.Path(tempfile.mkdtemp())
files = {"mixed": tmp / "mixed.csv", "floats": tmp / "floats.csv", "ints": tmp / "ints.csv"}
frame.to_csv(files["mixed"], index=False)
frame[["x"]].to_csv(files["floats"], index=False)
frame[["id", "n"]].to_csv(files["ints"], index=False)

lines = [
    "# M9: read_csv",
    "",
    "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()),
    "",
    f"1M rows. mixed: int, float (repr of float64), string, bool, int ({files['mixed'].stat().st_size / 1e6:.0f} MB). "
    "Ours single-threaded; Polars parallel.",
    "",
    "| file | ours | pandas | Polars |",
    "|---|---:|---:|---:|",
]
for name, path in files.items():
    t = [
        per_call_ns(lambda: sdpd.read_csv(path), target_s=0.5, repeats=3),
        per_call_ns(lambda: pd.read_csv(path), target_s=0.5, repeats=3),
        per_call_ns(lambda: pl.read_csv(path), target_s=0.5, repeats=3),
    ]
    lines.append(f"| {name} | {fmt_ns(t[0])} | {fmt_ns(t[1])} | {fmt_ns(t[2])} |")
    print(lines[-1], flush=True)
(pathlib.Path(__file__).parent / "results" / "m9-csv.md").write_text("\n".join(lines) + "\n")
