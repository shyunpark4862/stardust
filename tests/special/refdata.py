"""Reference values for sd.special and the error measures (docs/design/special.md §6).

tools/special_ref.py computes each function at 50 significant digits with mpmath and
stores, in tests/special/ref/<name>.npz,

    x0, x1, ...   the inputs (float64)
    hi, lo        the reference as a double-double: value ~= hi + lo, |lo| <= ulp(hi) / 2
    guaranteed    bool: the input lies in the function's guaranteed region (docs/api/special.md §1.4)
    scale         the bound's multiplier at each point (S4 on): max(1, kappa) for the inverses,
                  whose contract is |dx / x| <= 1e-14 max(1, kappa); 1 elsewhere (absent in
                  older files)

so an error can be measured to a fraction of an ULP without mpmath at test time.
"""

import pathlib

import numpy as np

REF = pathlib.Path(__file__).parent / "ref"


# Functions whose region outside the guaranteed one is held to |error| <= 1e-14 max(1, |value|)
# (S3 user decision), and reference files named apart from their function.
MIXED = {"gammaln", "digamma", "polygamma", "beta", "betaln", "binom", "zeta_hurwitz"}
ALIAS = {"zeta_hurwitz": "zeta"}


def mixed_error(got, hi, lo):
    """|got - (hi + lo)| / max(1, |hi|)."""
    finite, special = _special(got, hi)
    with np.errstate(invalid="ignore", over="ignore"):
        err = np.abs((np.asarray(got, dtype=np.float64) - hi) - lo) / np.maximum(1.0, np.abs(hi))
    return np.where(finite, err, special)


# Reference files that are not one function's points (tests/special/test_reduce_expint_ellip.py).
CASE_FILES = {"reduce_cases"}


def names():
    """The functions with a reference file."""
    return sorted(p.stem for p in REF.glob("*.npz") if p.stem not in CASE_FILES)


def load(name):
    """(inputs, hi, lo, guaranteed) for `name`; inputs is a tuple of float64 arrays."""
    with np.load(REF / f"{name}.npz") as f:
        nin = int(f["nin"])
        return tuple(f[f"x{k}"] for k in range(nin)), f["hi"], f["lo"], f["guaranteed"]


def scale(name):
    """The bound's multiplier at each reference point (ones where the file has none)."""
    with np.load(REF / f"{name}.npz") as f:
        return f["scale"] if "scale" in f.files else np.ones(f["hi"].size)


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
