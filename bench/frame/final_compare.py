"""M12: one library's times on the final case set.

bench/final_compare.sh builds the end commits of M9, M10 and M11 into
build/stage/<tag>/pkg and runs this script against each build in turn, and
against pandas and Polars, several rounds, so that the machine's drift over
the session hits every column alike. Each run keeps each row's best over the
rounds.

Usage: python bench/final_compare.py LIB OUT.json DATA_DIR [--threads N] [--fresh]
  LIB      : sdpd (the build on PYTHONPATH), pandas or polars
  DATA_DIR : CSV inputs, written by the first run that misses them
  --threads: sdpd only (Polars takes POLARS_MAX_THREADS from the environment)
  --fresh  : each call gets a different input of the same shape (fresh(n)
             of them, called in turn), so no column is timed on an input
             whose branches the CPU has just learned (_harness.per_call_ns_cycle)
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np  # noqa: E402
from _harness import per_call_ns, per_call_ns_cycle  # noqa: E402

SIZES = [1_000, 100_000, 1_000_000]
SMALL = 10  # M0's per-call overhead rows


def fresh(n):
    """Inputs per size in --fresh mode: many small ones, a few big ones."""
    return min(64, max(4, 4_000_000 // n))


def data(n, seed=0):
    """The inputs at n rows; seed 0 is M11's at every size."""
    rng = np.random.default_rng(seed)
    low, high = rng.integers(0, 1000, n), rng.integers(0, n, n)
    strs = [f"key{v}" for v in rng.integers(0, max(n // 10, 1), n).tolist()]
    f, g = rng.standard_normal(n), rng.standard_normal(n)
    m = max(n // 10, 1)
    lk, rk, rb = rng.integers(0, m + m // 9, n), rng.permutation(m), rng.standard_normal(m)
    return dict(low=low, high=high, strs=strs, f=f, g=g, lk=lk, rk=rk, rb=rb)


def write_csvs(d, n, data_dir, seed=0):
    tag = f"{n}" if seed == 0 else f"{n}-{seed}"
    fpath, mpath = data_dir / f"f{tag}.csv", data_dir / f"m{tag}.csv"
    if not fpath.exists():
        import pandas as pd

        pd.DataFrame({"x": d["f"]}).to_csv(fpath, index=False)
        pd.DataFrame({"id": np.arange(n), "x": d["f"], "name": d["strs"], "flag": d["f"] > 0}).to_csv(
            mpath, index=False
        )
    return fpath, mpath


def sdpd_cases(d, fpath, mpath):
    import stardust_pandas as sdpd

    o = {
        "f": sdpd.Series(d["f"]),
        "g": sdpd.Series(d["g"]),
        "low": sdpd.Series(d["low"]),
        "high": sdpd.Series(d["high"]),
        "str": sdpd.Series(d["strs"]),
        "df": sdpd.DataFrame({"k": d["low"], "kh": d["high"], "ks": d["strs"], "f": d["f"]}),
        "L": sdpd.DataFrame({"k": d["lk"], "a": d["f"]}),
        "R": sdpd.DataFrame({"k": d["rk"], "b": d["rb"]}),
    }
    return o, {
        "float + float": lambda: o["f"] + o["g"],
        "float sum": lambda: o["f"].sum(),
        "mask filter": lambda: o["f"][o["f"] > 0],
        "nunique int, 1k keys": lambda: o["low"].nunique(),
        "nunique int, n keys": lambda: o["high"].nunique(),
        "nunique string": lambda: o["str"].nunique(),
        "groupby int 1k sum": lambda: o["df"].groupby("k")["f"].sum(),
        "groupby int, n keys, sum": lambda: o["df"].groupby("kh")["f"].sum(),
        "groupby string sum": lambda: o["df"].groupby("ks")["f"].sum(),
        "groupby int + string sum": lambda: o["df"].groupby(["k", "ks"])["f"].sum(),
        "merge n x n/10 inner": lambda: sdpd.merge(o["L"], o["R"], on="k"),
        "sort int64": lambda: o["high"].sort_values(),
        "sort string": lambda: o["str"].sort_values(),
        "read_csv floats": lambda: sdpd.read_csv(fpath),
        "read_csv mixed": lambda: sdpd.read_csv(mpath),
    }


def pandas_cases(d, fpath, mpath):
    import pandas as pd

    p = {
        "f": pd.Series(d["f"]),
        "g": pd.Series(d["g"]),
        "low": pd.Series(d["low"]),
        "high": pd.Series(d["high"]),
        "str": pd.Series(d["strs"]),
        "df": pd.DataFrame({"k": d["low"], "kh": d["high"], "ks": d["strs"], "f": d["f"]}),
        "L": pd.DataFrame({"k": d["lk"], "a": d["f"]}),
        "R": pd.DataFrame({"k": d["rk"], "b": d["rb"]}),
    }
    return p, {
        "float + float": lambda: p["f"] + p["g"],
        "float sum": lambda: p["f"].sum(),
        "mask filter": lambda: p["f"][p["f"] > 0],
        "nunique int, 1k keys": lambda: p["low"].nunique(),
        "nunique int, n keys": lambda: p["high"].nunique(),
        "nunique string": lambda: p["str"].nunique(),
        "groupby int 1k sum": lambda: p["df"].groupby("k")["f"].sum(),
        "groupby int, n keys, sum": lambda: p["df"].groupby("kh")["f"].sum(),
        "groupby string sum": lambda: p["df"].groupby("ks")["f"].sum(),
        "groupby int + string sum": lambda: p["df"].groupby(["k", "ks"])["f"].sum(),
        "merge n x n/10 inner": lambda: p["L"].merge(p["R"], on="k"),
        "sort int64": lambda: p["high"].sort_values(kind="stable"),
        "sort string": lambda: p["str"].sort_values(kind="stable"),
        "read_csv floats": lambda: pd.read_csv(fpath),
        "read_csv mixed": lambda: pd.read_csv(mpath),
    }


def polars_cases(d, fpath, mpath):
    import polars as pl

    q = {
        "f": pl.Series(d["f"]),
        "g": pl.Series(d["g"]),
        "low": pl.Series(d["low"]),
        "high": pl.Series(d["high"]),
        "str": pl.Series(d["strs"]),
        "df": pl.DataFrame({"k": d["low"], "kh": d["high"], "ks": d["strs"], "f": d["f"]}),
        "L": pl.DataFrame({"k": d["lk"], "a": d["f"]}),
        "R": pl.DataFrame({"k": d["rk"], "b": d["rb"]}),
    }
    # groupby: ours and pandas return the groups sorted by key, so Polars sorts too.
    # merge: ours and pandas keep the left order, so Polars is asked to.
    return q, {
        "float + float": lambda: q["f"] + q["g"],
        "float sum": lambda: q["f"].sum(),
        "mask filter": lambda: q["f"].filter(q["f"] > 0),
        "nunique int, 1k keys": lambda: q["low"].n_unique(),
        "nunique int, n keys": lambda: q["high"].n_unique(),
        "nunique string": lambda: q["str"].n_unique(),
        "groupby int 1k sum": lambda: q["df"].group_by("k").agg(pl.col("f").sum()).sort("k"),
        "groupby int, n keys, sum": lambda: q["df"].group_by("kh").agg(pl.col("f").sum()).sort("kh"),
        "groupby string sum": lambda: q["df"].group_by("ks").agg(pl.col("f").sum()).sort("ks"),
        "groupby int + string sum": lambda: q["df"].group_by(["k", "ks"]).agg(pl.col("f").sum()).sort(["k", "ks"]),
        "merge n x n/10 inner": lambda: q["L"].join(q["R"], on="k", maintain_order="left"),
        "sort int64": lambda: q["high"].sort(),
        "sort string": lambda: q["str"].sort(),
        "read_csv floats": lambda: pl.read_csv(fpath),
        "read_csv mixed": lambda: pl.read_csv(mpath),
    }


def small_cases(lib):
    """M0's four rows at n = 10: the fixed cost of one call."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(SMALL)
    if lib == "sdpd":
        import stardust_pandas as m

        s, df = m.Series(a), m.DataFrame({"a": a, "b": a})
        flt = lambda: df[df["a"] > 0]  # noqa: E731
    elif lib == "pandas":
        import pandas as m

        s, df = m.Series(a), m.DataFrame({"a": a, "b": a})
        flt = lambda: df[df["a"] > 0]  # noqa: E731
    else:
        import polars as m

        s, df = m.Series(a), m.DataFrame({"a": a, "b": a})
        flt = lambda: df.filter(df["a"] > 0)  # noqa: E731
    return {"s + s": lambda: s + s, "s.sum()": lambda: s.sum(), 'df["a"]': lambda: df["a"], 'df[df["a"] > 0]': flt}


def main():
    lib, out, data_dir = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    data_dir.mkdir(parents=True, exist_ok=True)
    threads = None
    if lib == "sdpd":
        import stardust_pandas as sdpd

        if "--threads" in sys.argv:
            sdpd.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]))
        threads = sdpd.get_num_threads()
    elif lib == "polars":
        import polars as pl

        threads = pl.thread_pool_size()
    else:
        threads = 1
    make = {"sdpd": sdpd_cases, "pandas": pandas_cases, "polars": polars_cases}[lib]
    rows = {}
    for name, fn in small_cases(lib).items():
        rows[f"{name} | {SMALL}"] = per_call_ns(fn, target_s=0.05, repeats=7)
    for n in SIZES:
        variants = []
        for seed in range(fresh(n) if "--fresh" in sys.argv else 1):
            d = data(n, seed)
            variants.append(make(d, *write_csvs(d, n, data_dir, seed))[1])
        for name in variants[0]:
            fns = [v[name] for v in variants]
            rows[f"{name} | {n}"] = per_call_ns_cycle(fns, target_s=0.05 if n < SIZES[-1] else 0.2, repeats=5)
        print(f"  {lib} n={n} done", flush=True)
    old = json.loads(out.read_text()) if out.exists() else {"lib": lib, "rounds": 0, "rows": {}}
    for k, v in rows.items():  # each row's best over the rounds
        prev = old["rows"].get(k)
        old["rows"][k] = v if prev is None else min(prev, v)
    old["rounds"] += 1
    old["threads"] = threads
    out.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
