# bench/special/boost: run as .venv/bin/python bench/special/boost/fails.py <dir with boost_eval> NAME ... (from the repository root)
# For each candidate: the points where Boost breaks the contract (test_reference's measure), by input.
import subprocess
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests/special")
from tests.special import refdata
from tests.special.test_reference import LOOSE, contract_error

D = sys.argv[1]


def write(p, xs):
    with open(p, "wb") as f:
        np.array([xs[0].size, len(xs)], dtype=np.int64).tofile(f)
        for x in xs:
            np.ascontiguousarray(x, dtype=np.float64).tofile(f)


for name in sys.argv[2:]:
    xs, hi, lo, g = refdata.load(name)
    sc = refdata.scale(name)
    write(D + "/in.bin", xs)
    subprocess.run([D + "/boost_eval", "eval", name, D + "/in.bin", D + "/out.bin"], check=True)
    b = np.fromfile(D + "/out.bin")
    e = contract_error(b, hi, lo) / sc / 1e-14  # 1 = the bound
    if name in refdata.MIXED:
        mx = refdata.mixed_error(b, hi, lo) / 1e-14
        e = np.where(g, e, mx)
    elif (~g).any():
        e = np.where(g, e, e * 1e-14 / LOOSE.get(name, 1e-14))
    bad = e > 1
    print(f"== {name}: {bad.sum()} of {e.size} over the bound")
    if bad.any():
        X = np.stack(xs, axis=1)[bad]
        for k in range(X.shape[1]):
            print(
                f"   x{k} in [{X[:, k].min():.6g}, {X[:, k].max():.6g}]   (all: [{xs[k].min():.4g}, {xs[k].max():.4g}])"
            )
        idx = np.argsort(e)[-6:][::-1]
        for i in idx:
            print("   ", tuple(f"{x[i]:.6g}" for x in xs), f"{e[i]:.3g}x", f"ref {hi[i]:.4g}")
