"""M4: label lookups, alignment and assignment, vs pandas 3.1.1 (and Polars where it has a match).

Polars has no row labels, so lookups and alignment are pandas-only rows.
Writes bench/results/m4-indexing.md.
"""

import pathlib
import time

import numpy as np
import pandas as pd

import stardust_pandas as sdpd
from _harness import env_info, fmt_ns, per_call_ns

N = 1_000_000
rng = np.random.default_rng(0)
labels = [f"k{v}" for v in rng.permutation(N).tolist()]
shuffled = [labels[i] for i in rng.permutation(N).tolist()]
values = rng.integers(0, 100, N)

o = sdpd.Series(values, index=labels)
p = pd.Series(values, index=pd.Index(labels, dtype="str"))
o2 = sdpd.Series(values, index=shuffled)
p2 = pd.Series(values, index=pd.Index(shuffled, dtype="str"))
some = labels[::1000]


def first_lookup(make):
    best = float("inf")
    for _ in range(3):
        s = make()
        t0 = time.perf_counter_ns()
        s.loc[labels[123]]
        best = min(best, time.perf_counter_ns() - t0)
    return best


lines = ["# M4: indexing", "", "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()), "",
         "1M rows with unique string labels in random order. Minimum per-call time. Single thread.", "",
         "| case | ours | pandas |", "|---|---:|---:|"]


def row(name, t_ours, t_pd):
    lines.append(f"| {name} | {fmt_ns(t_ours)} | {fmt_ns(t_pd)} |")
    print(lines[-1], flush=True)


row("first .loc[label] (builds the lookup table)",
    first_lookup(lambda: sdpd.Series(values, index=labels)),
    first_lookup(lambda: pd.Series(values, index=pd.Index(labels, dtype="str"))))
for name, fo, fp in [
    (".loc[label] (table built)", lambda: o.loc[labels[123]], lambda: p.loc[labels[123]]),
    (".loc[1,000 labels]", lambda: o.loc[some], lambda: p.loc[some]),
    (".iloc[i]", lambda: o.iloc[500_000], lambda: p.iloc[500_000]),
    (".iloc[a:b] (zero-copy)", lambda: o.iloc[1000:900_000], lambda: p.iloc[1000:900_000]),
    ("s + s, same index", lambda: o + o, lambda: p + p),
    ("s + s, same labels shuffled (align)", lambda: o + o2, lambda: p + p2),
    ("reindex(shuffled labels)", lambda: o.reindex(shuffled), lambda: p.reindex(shuffled)),
]:
    row(name, per_call_ns(fo, target_s=0.2, repeats=5), per_call_ns(fp, target_s=0.2, repeats=5))


def assign_ours():
    s = sdpd.Series(values)
    s.iloc[10] = 5
    return s


def assign_pandas():
    s = pd.Series(values)
    s.iloc[10] = 5
    return s


small_o, small_p = sdpd.Series(list(range(10))), pd.Series(range(10))


def small_set_o():
    small_o.iloc[3] = 1


def small_set_p():
    small_p.iloc[3] = 1


row("s.iloc[i] = v on a 10-row Series (in place)", per_call_ns(small_set_o), per_call_ns(small_set_p))
mask_o, mask_p = sdpd.Series(values) > 50, pd.Series(values) > 50
df_o = sdpd.DataFrame({"a": values, "b": values})
df_p = pd.DataFrame({"a": values, "b": values})


def mask_assign_o():
    df_o.loc[mask_o, "a"] = 0


def mask_assign_p():
    df_p.loc[mask_p, "a"] = 0


row("df.loc[mask, 'a'] = 0 (~50% rows)", per_call_ns(mask_assign_o, target_s=0.2, repeats=5),
    per_call_ns(mask_assign_p, target_s=0.2, repeats=5))
(pathlib.Path(__file__).parent / "results" / "m4-indexing.md").write_text("\n".join(lines) + "\n")
