"""References for hyp1f1 with a large negative a (beyond the reference file's |a| <= 50),
for bench/special/boost/regions.py: a = -10^u (u in [1, 12]), b in [0.5, 10], x in [0.1, 300].
Arb (python-flint) with tools/special_ref.py's rule (a ball settled to 2^-110); 30 s per point,
each printed at once.

    .venv/bin/python bench/special/boost/extra_refs.py <dir>   ->  <dir>/hyp1f1_bigneg.npz
"""

import pathlib
import signal
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import special_ref as R  # noqa: E402

D = pathlib.Path(sys.argv[1])
rng = np.random.default_rng(7)
n = 80
a = -(10 ** rng.uniform(1, 12, n)) - 0.37
b = rng.uniform(0.5, 10, n)
x = 10 ** rng.uniform(-1, 2.5, n)
keep, hi, lo = [], [], []
signal.signal(signal.SIGALRM, R._alarm)
for i in range(n):
    t = time.time()
    signal.alarm(30)
    try:
        v = R.evaluate(lambda a_, b_, x_: x_.hypgeom_1f1(a_, b_), (float(a[i]), float(b[i]), float(x[i])))
        h, l = R.double_double(v)
        keep.append(i)
        hi.append(h)
        lo.append(l)
        print(f"{i}: a={a[i]:.4g} b={b[i]:.3g} x={x[i]:.4g} -> {h:.6g} ({time.time() - t:.1f} s)", flush=True)
    except R._DROPS as e:
        print(f"{i}: a={a[i]:.4g} dropped ({type(e).__name__})", flush=True)
    finally:
        signal.alarm(0)
k = np.array(keep, dtype=int)
np.savez(D / "hyp1f1_bigneg.npz", x0=a[k], x1=b[k], x2=x[k], hi=np.array(hi), lo=np.array(lo))
print(f"{k.size} of {n} kept")
