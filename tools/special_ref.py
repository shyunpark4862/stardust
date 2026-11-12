"""Reference values for sd.special from mpmath (docs/design/special.md §6).

    .venv/bin/python tools/special_ref.py            # every function in FUNCTIONS
    .venv/bin/python tools/special_ref.py NAME ...   # only these

writes tests/special/ref/<name>.npz (format: tests/special/refdata.py). The files are
committed; tests and bench/special/accuracy.py read them and never call mpmath.

Each function's input set (docs/design/special.md §6) is
  - a grid over its guaranteed region,
  - log-uniform random points (both signs where the domain has them), inside the
    guaranteed region and in the loose region around it,
  - edge points: a few ULP either side of every region boundary, poles and zeros,
    0, and the domain ends.
The generator is seeded: running the tool again gives the same file.
"""

import sys
from pathlib import Path

import mpmath
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "special" / "ref"
DPS = 50  # significant decimal digits: far beyond float64's 16, so hi + lo is exact to rounding


# ---------------------------------------------------------------- input sets


def grid(lo, hi, n):
    return np.linspace(lo, hi, n)


def loguniform(rng, lo, hi, n, signed=False):
    """n points with log|x| uniform in [log lo, log hi] (lo > 0); random signs when signed."""
    x = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    if signed:
        x *= rng.choice([-1.0, 1.0], n)
    return x


def around(points, k=4):
    """Each point and the k float64 neighbours on either side."""
    out = []
    for p in points:
        v = float(p)
        out.append(v)
        up = down = v
        for _ in range(k):
            up, down = np.nextafter(up, np.inf), np.nextafter(down, -np.inf)
            out += [float(up), float(down)]
    return np.array(out)


# ---------------------------------------------------------------- the functions
#
# name: (nin, mpmath function of mpf inputs, input-set builder rng -> (inputs, guaranteed))


def _cf_tan_inputs(rng):
    # Guaranteed |x| <= 1; loose 1 < |x| < pi/2, where the fraction needs more terms and
    # tan's own condition number x / (sin x cos x) grows without bound.
    g = np.concatenate([grid(-1, 1, 2001), loguniform(rng, 1e-300, 1, 4000, signed=True),
                        around([0.0, 1.0, -1.0, 5e-324, -5e-324])])
    loose = np.concatenate([rng.uniform(1, 1.5707963267948966, 2000) * rng.choice([-1.0, 1.0], 2000),
                            around([1.5707963267948966, -1.5707963267948966])])
    x = np.concatenate([g, loose])
    return (x,), np.abs(x) <= 1.0


FUNCTIONS = {
    # the engine's test function (csrc/special/ops/testing.hpp): checks the pipeline end to end
    "_cf_tan": (1, mpmath.tan, _cf_tan_inputs),
}


# ---------------------------------------------------------------- writing


def double_double(v):
    """(hi, lo) with hi + lo ~= v, hi = v rounded to float64."""
    if mpmath.isnan(v):
        return np.nan, 0.0
    if mpmath.isinf(v):
        return (np.inf if v > 0 else -np.inf), 0.0
    hi = float(v)
    if np.isinf(hi):  # beyond float64: the reference overflows
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def build(name):
    nin, f, inputs = FUNCTIONS[name]
    rng = np.random.default_rng(20261113)
    xs, guaranteed = inputs(rng)
    xs = tuple(np.asarray(x, dtype=np.float64) for x in xs)
    n = xs[0].size
    hi, lo = np.empty(n), np.empty(n)
    mpmath.mp.dps = DPS
    for i in range(n):
        try:
            v = f(*(mpmath.mpf(float(x[i])) for x in xs))
        except (ValueError, ZeroDivisionError):
            v = mpmath.nan
        if isinstance(v, mpmath.mpc):  # a real function outside its real domain
            v = v.real if v.imag == 0 else mpmath.nan
        hi[i], lo[i] = double_double(v)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", nin=nin, hi=hi, lo=lo, guaranteed=np.asarray(guaranteed, dtype=bool),
                        **{f"x{k}": x for k, x in enumerate(xs)}, mpmath=mpmath.__version__, dps=DPS)
    return n


def main(argv):
    names = argv or list(FUNCTIONS)
    for name in names:
        n = build(name)
        print(f"{name}: {n} points -> {(OUT / name).with_suffix('.npz').relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
