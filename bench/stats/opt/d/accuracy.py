"""The noncentral three against the Arb references (tests/stats/ref): the largest error in
the contract's units inside and outside the guaranteed region, per method (the rows of
bench/stats/t6/accuracy.py for these distributions, without scipy's column: fast to rerun).

    .venv/bin/python bench/stats/opt/d/accuracy.py [substring ...]
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from tests.stats import refdata  # noqa: E402


def main():
    want = sys.argv[1:]
    print("| method | points | sd max (in) | sd max (out) |")
    print("|---|---:|---:|---:|")
    for name in refdata.names():
        dist, method = refdata.split(name)
        if not dist.startswith("noncentral") or (want and not any(w in name for w in want)):
            continue
        xs, hi, lo, g, sc = refdata.load(name)
        e = refdata.contract_error(name, refdata.call(name, xs), hi, lo, sc)
        e = np.where(np.isnan(e), np.inf, e)
        ein = e[g].max() if g.any() else math.nan
        eout = e[~g].max() if (~g).any() else math.nan
        print(f"| `{name}` | {hi.size} | {ein:.1e} | {eout:.1e} |", flush=True)


if __name__ == "__main__":
    main()
