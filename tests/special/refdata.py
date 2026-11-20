"""Reference values for sd.special and the error measures (docs/design/special.md §6).

tools/special_ref.py computes each function with Arb (python-flint), as a ball that contains
the true value and is settled to 2^-110 of it, and stores, in tests/special/ref/<name>.npz,

    x0, x1, ...   the inputs (float64)
    hi, lo        the reference as a double-double: value ~= hi + lo, |lo| <= ulp(hi) / 2
    guaranteed    bool: the input lies in the function's guaranteed region (docs/api/special.md §1.4)
    scale         the bound's multiplier at each point: the contract is |error| <= 1e-14 scale
                  (relative). max(1, kappa) for the inverses (|dx / x| <= 1e-14 max(1, kappa));
                  max(1, M / |f|), M the envelope of |f|, near the zeros of the oscillatory
                  Bessel and Airy functions and of the complex error functions and wofz;
                  1 elsewhere (absent in older files)
    arb           the provenance: "python-flint <version>" that computed the reference
    target_bits   the settling target: each ball's radius is below 2^-target_bits of its value
                  (some older files call it bits, or give the bare version in arb;
                  provenance() reads both)

so an error can be measured to a fraction of an ULP without Arb at test time. The stats
reference files (tools/stats_ref.py, tests/stats/refdata.py) have the same format.
"""

import pathlib

import numpy as np

REF = pathlib.Path(__file__).parent / "ref"


# Functions whose region outside the guaranteed one is held to |error| <= 1e-14 max(1, |value|)
# (S3 user decision), and reference files named apart from their function.
MIXED = {"gammaln", "digamma", "polygamma", "beta", "betaln", "binom", "zeta_hurwitz"}
ALIAS = {"zeta_hurwitz": "zeta"}
# Reference files for one output of a several-output function: name -> (function, index).
PART = {"airy_ai": ("airy", 0), "airy_aip": ("airy", 1), "airy_bi": ("airy", 2), "airy_bip": ("airy", 3)}
# Reference files for a keyword option of a function, or for a complex argument of a
# function whose real file has its name: name -> (function, keywords).
KW = {
    "lambertw_m1": ("lambertw", {"k": -1}),
    # S8: complex arguments (x0 is complex128; hi, lo hold both parts as double-doubles)
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


def load(name):
    """(inputs, hi, lo, guaranteed) for `name`; inputs is a tuple of float64 arrays."""
    with np.load(REF / f"{name}.npz") as f:
        nin = int(f["nin"])
        return tuple(f[f"x{k}"] for k in range(nin)), f["hi"], f["lo"], f["guaranteed"]


def provenance(name, ref=REF):
    """{"arb": "python-flint <version>", "target_bits": int} of a reference file in `ref`
    (tests/stats/ref too), from either spelling of the fields; None where the file has none."""
    with np.load(ref / f"{name}.npz") as f:
        arb = str(f["arb"]) if "arb" in f.files else None
        bits = next((int(f[k]) for k in ("target_bits", "bits") if k in f.files), None)
    if arb is not None and not arb.startswith("python-flint"):
        arb = f"python-flint {arb}"
    return {"arb": arb, "target_bits": bits}


def scale(name):
    """The bound's multiplier at each reference point (ones where the file has none)."""
    with np.load(REF / f"{name}.npz") as f:
        return f["scale"] if "scale" in f.files else np.ones(f["hi"].size)


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
    """|got - (hi + lo)|, for complex the modulus of the difference (S8: the relative error
    of a complex value is |error| / |value|)."""
    if np.iscomplexobj(hi):
        got = np.asarray(got, dtype=np.complex128)
        return np.hypot((got.real - hi.real) - lo.real, (got.imag - hi.imag) - lo.imag)
    return np.abs((np.asarray(got, dtype=np.float64) - hi) - lo)


def ulp_error(got, hi, lo):
    """|got - (hi + lo)| in units of the spacing of float64 at |hi|."""
    finite, special = _special(got, hi)
    with np.errstate(invalid="ignore", over="ignore"):
        err = _diff(got, hi, lo) / np.spacing(np.abs(hi))
    return np.where(finite, err, special)


def rel_error(got, hi, lo):
    """|got - (hi + lo)| / |hi + lo|; 0 where both are 0."""
    finite, special = _special(got, hi)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        diff = _diff(got, hi, lo)
        err = np.where(diff == 0, 0.0, diff / np.abs(hi))
    return np.where(finite, err, special)
