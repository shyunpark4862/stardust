"""Reference values for sd.special and the error measures (docs/design/special.md §6).

tools/special_ref.py computes each function with Arb (python-flint), as a ball that contains
the true value and is settled to 2^-110 of it, and stores tests/special/ref/<name>.npz in the
layout of tests/reffile.py: the inputs x0, x1, ... (float64, or complex128 for the complex
files) and one method, the function, with

    hi, lo        the reference as a double-double: value ~= hi + lo, |lo| <= ulp(hi) / 2
    guaranteed    bool: the input lies in the function's guaranteed region (docs/api/special.md §1.4)
    scale         the bound's multiplier at each point: the contract is |error| <= 1e-14 scale
                  (relative). max(1, kappa) for the inverses (|dx / x| <= 1e-14 max(1, kappa));
                  max(1, M / |f|), M the envelope of |f|, near the zeros of the oscillatory
                  Bessel and Airy functions and of the complex error functions and wofz;
                  1 elsewhere

so an error can be measured far below an ULP without Arb at test time (the ULP measure is
bench/accuracy/common.py's ulp_distance to hi). The stats
reference files (tools/stats_ref.py, tests/stats/refdata.py) have the same layout.
"""

import pathlib

import numpy as np

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"


# Functions whose region outside the guaranteed one is held to |error| <= 1e-14 max(1, |value|)
# (docs/design/special.md §2.3), and reference files named apart from their function.
MIXED = {"gammaln", "digamma", "polygamma", "beta", "betaln", "binom", "zeta_hurwitz"}
ALIAS = {"zeta_hurwitz": "zeta"}
# Reference files for one output of a several-output function: name -> (function, index).
PART = {"airy_ai": ("airy", 0), "airy_aip": ("airy", 1), "airy_bi": ("airy", 2), "airy_bip": ("airy", 3)}
# Reference files for a keyword option of a function, or for a complex argument of a
# function whose real file has its name: name -> (function, keywords).
KW = {
    "lambertw_m1": ("lambertw", {"k": -1}),
    # complex arguments (x0 is complex128; hi, lo hold both parts as double-doubles)
    "erf_complex": ("erf", {}),
    "erfc_complex": ("erfc", {}),
    "erfcx_complex": ("erfcx", {}),
    "loggamma_complex": ("loggamma", {}),
    "lambertw_c0": ("lambertw", {"k": 0}),
    "lambertw_cm1": ("lambertw", {"k": -1}),
    "lambertw_c1": ("lambertw", {"k": 1}),
    "lambertw_c4": ("lambertw", {"k": 4}),
}


def mixed_error(got, hi, lo):
    """|got - (hi + lo)| / max(1, |hi|)."""
    finite, special = _special(got, hi)
    with np.errstate(invalid="ignore", over="ignore"):
        err = _diff(got, hi, lo) / np.maximum(1.0, np.abs(hi))
    return np.where(finite, err, special)


# Reference files that are not one function's points (tests/special/test_reduce_expint_ellip.py).
CASE_FILES = {"reduce_cases"}


def names():
    """The functions with a reference file."""
    return sorted(p.stem for p in REF.glob("*.npz") if p.stem not in CASE_FILES)


def _read(name):
    r = reffile.read(REF / f"{name}.npz")
    return r, r.methods[name]


def load(name):
    """(inputs, hi, lo, guaranteed) for `name`; inputs is a tuple of float64 (or complex128) arrays."""
    r, m = _read(name)
    return r.args(name), m.hi, m.lo, m.fields["guaranteed"]


def scale(name):
    """The bound's multiplier at each reference point."""
    return _read(name)[1].fields["scale"]


def provenance(name, ref=REF):
    """{"arb", "target_bits", "tool", "n_requested"} of the reference file `name` in `ref`."""
    return reffile.provenance(ref / f"{name}.npz")


def reduce_cases():
    """(x, (lse hi, lo), (softmax hi, lo), (log_softmax hi, lo)) per vector of
    tests/special/ref/reduce_cases.npz (logsumexp, softmax and log_softmax)."""
    r = reffile.read(REF / "reduce_cases.npz")
    x, M = r.inputs["x"], r.methods
    for c in range(r.ngroups):
        sel = r.group == c
        yield (
            x[sel],
            (M["lse"].hi[c], M["lse"].lo[c]),
            (M["softmax"].hi[sel], M["softmax"].lo[sel]),
            (M["log_softmax"].hi[sel], M["log_softmax"].lo[sel]),
        )


def _special(got, hi):
    """The error where the reference is NaN or infinite, or the result is: 0 for an
    exact match (NaN for NaN, the same infinity; for complex, part by part), inf otherwise.
    None where both are finite."""
    if np.iscomplexobj(hi):
        got = np.asarray(got, dtype=np.complex128)
        finite = np.isfinite(hi) & np.isfinite(got)

        def same(g, h):
            return (np.isnan(h) & np.isnan(g)) | (g == h)

        match = same(got.real, hi.real) & same(got.imag, hi.imag)
        return finite, np.where(match, 0.0, np.inf)
    got = np.asarray(got, dtype=np.float64)
    finite = np.isfinite(hi) & np.isfinite(got)
    match = (np.isnan(hi) & np.isnan(got)) | (np.isinf(hi) & (got == hi))
    return finite, np.where(match, 0.0, np.inf)


def _diff(got, hi, lo):
    """|got - (hi + lo)|, for complex the modulus of the difference (the relative error of a
    complex value is |error| / |value|)."""
    if np.iscomplexobj(hi):
        got = np.asarray(got, dtype=np.complex128)
        return np.hypot((got.real - hi.real) - lo.real, (got.imag - hi.imag) - lo.imag)
    return np.abs((np.asarray(got, dtype=np.float64) - hi) - lo)


def rel_error(got, hi, lo):
    """|got - (hi + lo)| / |hi + lo|; 0 where both are 0."""
    finite, special = _special(got, hi)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        diff = _diff(got, hi, lo)
        err = np.where(diff == 0, 0.0, diff / np.abs(hi))
    return np.where(finite, err, special)
