"""Reference values for sd.special and the error measures (docs/design/special.md §6).

tools/special_ref.py computes each function at 50 significant digits with mpmath and
stores, in tests/special/ref/<name>.npz,

    x0, x1, ...   the inputs (float64)
    hi, lo        the reference as a double-double: value ~= hi + lo, |lo| <= ulp(hi) / 2
    guaranteed    bool: the input lies in the function's guaranteed region (docs/api/special.md §1.4)

so an error can be measured to a fraction of an ULP without mpmath at test time.
"""

import pathlib

import numpy as np

REF = pathlib.Path(__file__).parent / "ref"


def names():
    """The functions with a reference file."""
    return sorted(p.stem for p in REF.glob("*.npz"))


def load(name):
    """(inputs, hi, lo, guaranteed) for `name`; inputs is a tuple of float64 arrays."""
    with np.load(REF / f"{name}.npz") as f:
        nin = int(f["nin"])
        return tuple(f[f"x{k}"] for k in range(nin)), f["hi"], f["lo"], f["guaranteed"]


def _special(got, hi):
    """The error where the reference is NaN or infinite, or the result is: 0 for an
    exact match (NaN for NaN, the same infinity), inf otherwise. None where both are finite."""
    got = np.asarray(got, dtype=np.float64)
    finite = np.isfinite(hi) & np.isfinite(got)
    match = (np.isnan(hi) & np.isnan(got)) | (np.isinf(hi) & (got == hi))
    return finite, np.where(match, 0.0, np.inf)


def ulp_error(got, hi, lo):
    """|got - (hi + lo)| in units of the spacing of float64 at |hi|."""
    finite, special = _special(got, hi)
    got = np.asarray(got, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        err = np.abs((got - hi) - lo) / np.spacing(np.abs(hi))
    return np.where(finite, err, special)


def rel_error(got, hi, lo):
    """|got - (hi + lo)| / |hi + lo|; 0 where both are 0."""
    finite, special = _special(got, hi)
    got = np.asarray(got, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        diff = np.abs((got - hi) - lo)
        err = np.where(diff == 0, 0.0, diff / np.abs(hi))
    return np.where(finite, err, special)
