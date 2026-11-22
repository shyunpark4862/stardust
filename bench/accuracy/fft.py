"""sd.fft's accuracy sweeps: one per reference family of tests/fft/ref (tools/fft_ref.py; c2c, real,
hermitian, r2r, nd) once the functions it measures are public. FT1 brings

  c2c   fft (method forward) and ifft(norm='forward') (method backward: unnormalised, sign +) on every
        line of tests/fft/ref/c2c.npz, plus the long lines the file leaves out (2^16, 65537, 3 10007 and
        2^20: tests/fft/refdata.LONG, decision FT33) with Arb computed where the run is. Regions by
        method and length form: pow2 (2^k), f2k (f 2^k, f in {3, 5, 15}, k >= 3: vDSP's classes, decision
        FT22), mixed (the other 11-smooth lengths) and bluestein (a prime factor above 11, decision
        FT36).

A point is one line, judged by the contract of docs/design/fft.md §4 through tests/fft/refdata.py
(bound, units; Bluestein lengths with their rho and inner length); rel is the normwise relative error
||y_hat - y|| / ||y||. No ULP column: an entry of an FFT has no accuracy of its own (an entry near 0
moves by the size of the others, docs/design/fft.md §4.1), so entry-wise distances measure nothing.
numpy 2.5.4, scipy 1.18.1 and R 4.6.2's stats::fft (forward only) are the comparator columns, computed in
place on the same lines.

Later milestones: `real` and `hermitian` with FT2, `nd` with FT3, `r2r` with FT4.
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region

_FORMS = ("pow2", "f2k", "mixed", "bluestein")


def _form(n: int) -> str:
    from tests.fft import refdata

    if n & (n - 1) == 0:
        return "pow2"
    for f in (3, 5, 15):
        if n % f == 0:
            q = n // f
            if q >= 8 and q & (q - 1) == 0:
                return "f2k"
    return "mixed" if refdata.is_smooth11(n) else "bluestein"


def _lines():
    """(method, signal id, x, n, hi, lo) of every point: the stored lines, then the long ones."""
    from tests.fft import refdata
    from tests.fft import signals as S

    r = refdata.ref("c2c")
    for method in ("forward", "backward"):
        m = r.methods[method]
        for j, k in enumerate(m.index):
            yield method, str(r.inputs["signal"][k]), r.inputs["x"][k], int(r.inputs["n"][k]), m.hi[j], m.lo[j]
    for sid in refdata.LONG:
        s = S.get(sid)
        for method in ("forward", "backward"):
            hi, lo = refdata.arb_dft(s.x, method == "backward")
            yield method, sid, s.x, s.n, hi, lo


def _compute(lib, method, x):
    if lib == "sd":
        import stardust as sd

        sx = sd.asarray(x)
        return np.asarray(sd.fft.fft(sx) if method == "forward" else sd.fft.ifft(sx, norm="forward"))
    if lib == "numpy":
        return np.fft.fft(x) if method == "forward" else np.fft.ifft(x, norm="forward")
    if lib == "scipy":
        import scipy.fft as sf

        return sf.fft(x) if method == "forward" else sf.ifft(x, norm="forward")
    return None


def _regions(lib):
    from tests.fft import refdata

    pts = {}
    lines = list(_lines())
    got_r = _r_values(lines) if lib == "R" else None
    for i, (method, sid, x, n, hi, lo) in enumerate(lines):
        if lib == "R":
            got = got_r.get(i)
            if got is None:
                continue
        else:
            got = _compute(lib, method, x)
        if got is None:
            return None
        err = refdata.error(got, hi, lo)
        ynorm = refdata.norm2(hi)
        key = f"{method}/{_form(n)}"
        p = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
        p["where"].append(sid)
        p["contract"].append(refdata.units(err, refdata.bound("c2c", n, refdata.norm2(x))))
        p["rel"].append(err / ynorm if ynorm else (0.0 if err == 0 else np.inf))
    out = {}
    for method in ("forward", "backward"):
        for form in _FORMS:
            p = pts.get(f"{method}/{form}")
            if p:
                out[f"{method}/{form}"] = region(p["where"], rel=np.array(p["rel"]),
                                                 contract=np.array(p["contract"]), bound=1.0)  # fmt: skip
    return out or None


def _r_values(lines):
    """R's fft(z) of every forward line with finite values (one R call): {line index: result}."""
    from . import rsession

    keep = [(i, x) for i, (method, _, x, _, _, _) in enumerate(lines) if method == "forward" and np.all(np.isfinite(x))]
    if not keep:
        return {}
    xs = [np.asarray(x, np.complex128) for _, x in keep]
    expr = (
        "{ off <- c(0, cumsum(N)); unlist(lapply(seq_along(N), function(k) { i <- (off[k] + 1):off[k + 1]; "
        "y <- fft(complex(real = RE[i], imaginary = IM[i])); c(Re(y), Im(y)) })) }"
    )
    vals = rsession.evaluate_named(
        expr,
        {"RE": np.concatenate([x.real for x in xs]), "IM": np.concatenate([x.imag for x in xs]),
         "N": np.array([x.size for x in xs], dtype=np.float64)},
    )  # fmt: skip
    out, at = {}, 0
    for (i, _), x in zip(keep, xs):
        n = x.size
        out[i] = vals[at : at + n] + 1j * vals[at + n : at + 2 * n]
        at += 2 * n
    return out


def sweeps():
    return [
        Custom("fft", "c2c", lambda: _regions("sd"), compare_libs(_regions, ("numpy", "scipy", "R"), seconds=600.0))
    ]
