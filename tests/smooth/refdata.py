"""sd.smooth's references and their contract (docs/design/smooth.md §4, §7), as the tests and the accuracy
measurement (bench/accuracy) both judge them.

tests/smooth/ref/<family>.npz (tools/smooth_ref.py; layout: tests/reffile.py) holds one point per fitted
value, evaluation point, coefficient or output sample, with the inputs "set" (the id in
tests/smooth/samples.py), "kind" (0: a data point, coefficient or output sample; 1: an evaluation point of
the set; 2: a far evaluation point), "index" (the point's number in its set) and "digest" (the data's
digest); one group per set, whose inputs are the tuning the references were computed with; and per method
the reference hi + lo:

    local          lowess_f0.3, lowess_f0.67, lowess_f1, loess_d0_s0.75, loess_d1_s0.75, loess_d2_s0.75,
                   loess_d2_s0.5, loess_d1_s1.5; fields scale, leverage, kappa, npos
    local_nd       loess_d0_s0.5, loess_d1_s0.75, loess_d2_s0.75 (normalize off); the same fields
    spline         the coefficients: ss_1e-8 .. ss_1e2 (lam = c range^3: group inputs lam_ss_<c>, knots t_ss),
                   ps_1e-2, ps_1, ps_1e3 (knots t_ps, n_segments), lsq (knots t_lsq)
    spline_fit     the fitted values of the same methods with the field leverage; over the groups
                   trl_<m> (field kappa), gcv_<m>, cv_<m>
    kreg           nw_a, nw_b, ll_a, ll_b (group inputs bw_a, bw_b, far); field scale
    kde            pdf, logpdf, cdf, pdf_<kernel>, pdf_reflect, pdf_diag, logpdf_diag (group inputs H, bw, bw_diag)
    bandwidth      scott, silverman, nrd0, nrd, cvls_value, cvml_value (group input h), sj_ste, sj_dpi, isj
                   (group input isj_n), cvls_min, cvml_min (field width: the bisection's relative width)
    savgol         coeffs (group inputs window_length, polyorder, deriv, pos, norm_inf)
    savgol_filter  interp, mirror, nearest, constant, wrap (group inputs signal, window_length, polyorder,
                   deriv); field scale

tests/smooth/data holds the other libraries' records (tools/smooth_probe.py: scipy.json, statsmodels.json,
sklearn.json, r.json) and the prototype measurements of milestone SM0 (prototype.json), each with its own
"_provenance".

The contracts (docs/design/smooth.md §4; the constants are the drafts milestone SM0 measured with float64
prototypes of the planned algorithms (tests/smooth/data/prototype.json; tests/smooth/test_reference_data.py
holds them to these constants), and each milestone settles its own):

    local fits       |got - ref| <= C_LOCAL kappa u scale                         (§4.1)
    coefficients     |got - ref| <= C_SPLINE u kappa max|c|                       (§4.2)
    leverage         |got - ref| <= u kappa max(C_LEVERAGE, kappa); tr L: n times that   (§4.2)
    kernel regression  the local-fit bound with C_KREG, plus u scale (the cut)    (§4.4)
    pdf              |got - ref| <= kde_rel(n, p, d0) kappa_A |ref| (+ 2^-1073 below the normal range), d0 the
                     smallest squared distance in bandwidths (the exponent's rounding), kappa_A = sqrt(cond H) for
                     a full covariance, 1 for a diagonal one (decision SM117); logpdf: kde_rel(n, p) kappa_A
                     max(1, |ref|)                                                 (§4.5)
    cdf              |got - ref| <= kde_cdf_bound(n)                               (§4.5)
    compact kernels  |got - ref| <= kde_rel(n, 1) |ref| + 4 u sum w |z K'(z)| / h  (decision SM119)
    binned density   |got - ref| <= binned_bound(delta, h) (n-D: binned_bound_nd) (§4.5)
    bandwidths       closed forms C_BW u relative; roots and minimizers BW_ROOT_REL relative (§4.5)
    savgol_coeffs    |got - ref| <= C_SAVGOL u max|h|                             (§4.6)
    savgol_filter    |got - ref| <= gamma(N + 8) scale                            (§4.6)
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("bandwidth", "kde", "kreg", "local", "local_nd", "savgol", "savgol_filter", "spline", "spline_fit")
U = 2.0**-53
C_LOCAL = 16.0
C_SPLINE = 8.0
C_LEVERAGE = 8.0
C_KREG = 16.0
C_SAVGOL = 8.0


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


def ref(family: str) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def gamma(k: int) -> float:
    return k * U / (1 - k * U)


def at(r: reffile.Ref, method: str, sid: str, kind: int | None = None):
    """(positions in the method's arrays, the points' "index" input) of a set's points with that method, in
    index order of the kind."""
    m = r.methods[method]
    sel = r.inputs["set"][m.index] == sid
    if kind is not None:
        sel &= r.inputs["kind"][m.index] == kind
    pos = np.flatnonzero(sel)
    return pos, r.inputs["index"][m.index[pos]]


def group_of(r: reffile.Ref, sid: str) -> int:
    return int(np.flatnonzero(r.group_inputs["set"] == sid)[0])


def local_bound(scale: float, kappa: float) -> float:
    return C_LOCAL * kappa * U * scale


def spline_bound(kappa: float, cmax: float) -> float:
    return C_SPLINE * U * kappa * cmax


def leverage_bound(kappa: float) -> float:
    return U * kappa * max(C_LEVERAGE, kappa)


def trace_bound(n: int, kappa: float) -> float:
    return n * leverage_bound(kappa)


def kreg_bound(scale: float, kappa: float) -> float:
    return (C_KREG * kappa + 1.0) * U * scale


def kde_rel(n: int, p: int, d0: float = 0.0) -> float:
    return (8.0 + 2.0 * p + math.log2(n) + d0) * U


def kde_pdf_bound(n: int, p: int, d0: float, ref: float, kappa: float = 1.0) -> float:
    """pdf's contract (decisions SM31, SM117): relative kde_rel(n, p, d0) kappa_A, plus 2^-1073 for the subnormal
    values (scalbn's one rounding)."""
    return kde_rel(n, p, d0) * kappa * abs(ref) + 2.0**-1073


def kde_logpdf_bound(n: int, p: int, ref: float, kappa: float = 1.0) -> float:
    return kde_rel(n, p) * kappa * max(1.0, abs(ref))


def kde_cdf_bound(n: int) -> float:
    """cdf's absolute contract: the sum of n distribution functions (each a few u) and the prefix sums."""
    return (8.0 + math.log2(n)) * U


def kde_kernel_bound(n: int, ref: float, slope: float) -> float:
    """A compact kernel's contract (decision SM119): kde_rel(n, 1) |ref| for the sum and 4 u sum w |z K'(z)| / h for
    the rounding of each z (slope: that sum, computed by the caller)."""
    return kde_rel(n, 1) * abs(ref) + 4.0 * U * slope


C_BW = 8.0  # the closed-form rules (scott, silverman, nrd0, nrd): relative C_BW u
BW_ROOT_REL = 1e-8  # the roots and minimizers (sj, isj, cv_ls, cv_ml): relative (§4.5)


TAU = 8.3  # the binned path's kernel cut, in bandwidths (docs/design/smooth.md §3.5)


def binned_bound(delta: float, h: float) -> float:
    """The method's error of the linear-binning FFT density at a grid point (Gaussian kernel, weights summing to 1):
    the binning delta^2 / 8 max|K''| plus the cut phi(TAU) / h; the FFT's own rounding is far below."""
    return delta * delta / (8.0 * math.sqrt(2.0 * math.pi) * h**3) + math.exp(-0.5 * TAU * TAU) / (
        math.sqrt(2.0 * math.pi) * h
    )


def binned_bound_nd(deltas, Hinv_diag, k0: float) -> float:
    """The n-D binned density's method error (decision SM121): multilinear binning sum_l delta_l^2 (H^-1)_ll K(0) / 8
    (the largest second derivative along axis l of the Gaussian K_H is (H^-1)_ll K(0)) plus the cut K(0)
    exp(-TAU^2 / 2); k0 = K(0) = (2 pi)^(-p/2) det(H)^(-1/2)."""
    return sum(d * d * h for d, h in zip(deltas, Hinv_diag)) * k0 / 8.0 + k0 * math.exp(-0.5 * TAU * TAU)


def savgol_bound(norm_inf: float) -> float:
    return C_SAVGOL * U * norm_inf


def savgol_filter_bound(window_length: int, scale: float) -> float:
    return gamma(window_length + 8) * scale


def units(got: float, hi: float, lo: float, bound: float) -> float:
    """|got - ref| in units of a bound (0 when both are exactly equal; inf when the bound is 0 and they differ)."""
    err = abs((got - hi) - lo)
    if err == 0 or (math.isnan(got) and math.isnan(hi)):
        return 0.0
    return err / bound if bound > 0 else math.inf
