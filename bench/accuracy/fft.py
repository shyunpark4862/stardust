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

FT2 brings

  real       rfft (method rfft) and the whole spectrum of fft of real input (method fft, against the
             mirrored half-spectrum reference) on every line of tests/fft/ref/real.npz, plus the long lines
             of tests/fft/refdata.LONG_REAL (2^16, 65537, 2^20) with Arb where the run is
  hermitian  irfft(norm='forward') (method irfft: unnormalised) on every line of
             tests/fft/ref/hermitian.npz, plus the half spectra of the long real lines

with regions by the real path's form (decision FT43): pow2 (2^k), even (another even length whose half
is 11-smooth: the complex plan of n / 2 and the twiddle pass), odd (an odd 11-smooth length: the complex
plan of n and the Hermitian part) and bluestein (the inner length n / 2 or n has a prime factor above
11). numpy and scipy are the comparator columns (R has no real transform: it is compared on c2c only).

FT3 brings

  nd   fftn (method fftn) and ifftn(norm='forward') (method ifftn) on every array of tests/fft/ref/nd.npz
       and the complex points of tests/fft/refdata.nd_points (shapes with a Bluestein axis), rfftn (method
       rfftn) and irfftn(norm='forward') (method irfftn, the parts numpy and scipy drop from a
       non-Hermitian input dropped as theirs) on its real and Hermitian points; Arb computed where the run
       is (small shapes, a second or so). Regions by method and by the axes' paths: smooth (every axis, and
       a real axis's inner length, 11-smooth) and bluestein (an axis or a real axis's inner length with a
       prime factor above 11), judged by tests/fft/refdata.bound_nd. numpy (no hfftn: the four methods
       above) and scipy are the comparator columns.

FT4 brings

  r2r  dct and dst of types 1-4 (methods dct1 .. dst4, norm='backward': unnormalised) on every line of
       tests/fft/ref/r2r.npz, plus the lines of tests/fft/refdata.LONG_R2R (a prime, 2 1009, 2^16 and 65537)
       with Arb where the run is. Regions by method and by the inner length's form (csrc/fft/r2r.hpp,
       refdata.r2r_inner): even and odd (an 11-smooth inner complex length, from an even or an odd n) and
       bluestein (an inner length with a prime factor above 11, decision FT53). scipy is the comparator
       column (numpy has no DCT).

FT5 brings

  fftlog  fht and ifht on every point of tests/fft/refdata.fht_points (lengths 1-4096 and 2^16: even, odd and
          Bluestein inner lengths; the parameter sets of refdata.FHT_PARAMS; standard normal values, and the
          physical r^(mu + 1) e^(-r^2 / 2), subnormal and huge values on four lengths) with Arb where the run
          is (refdata.arb_fht), judged by refdata.bound_fht. Regions by direction and by the bias: unbiased
          (|U_m| = 1) and biased. scipy is the comparator column.
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
    return "mixed" if refdata.is_mixed(n) else "bluestein"


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


_REAL_FORMS = ("pow2", "even", "odd", "bluestein")


def _real_form(n: int) -> str:
    from tests.fft import refdata

    if not refdata.is_mixed(refdata.real_inner(n)):
        return "bluestein"
    if n & (n - 1) == 0:
        return "pow2"
    return "even" if n % 2 == 0 else "odd"


def _whole(half, n):
    return np.concatenate([half, np.conj(half[1 : (n + 1) // 2][::-1])])


def _real_lines(family):
    """(method, signal id, x, n, hi, lo) of the real or hermitian family: the stored lines, then the long
    ones (Arb here)."""
    from tests.fft import refdata
    from tests.fft import signals as S

    if family == "real":
        r = refdata.ref("real")
        m = r.methods["forward"]
        for j, k in enumerate(m.index):
            x, n, hi, lo = r.inputs["x"][k], int(r.inputs["n"][k]), m.hi[j], m.lo[j]
            sid = str(r.inputs["signal"][k])
            yield "rfft", sid, x, n, hi, lo
            yield "fft", sid, x, n, _whole(hi, n), _whole(lo, n)
        for sid in refdata.LONG_REAL:
            s = S.get(sid)
            hi, lo = refdata.arb_dft(s.x.astype(np.complex128))
            yield "rfft", sid, s.x, s.n, hi[: s.n // 2 + 1], lo[: s.n // 2 + 1]
            yield "fft", sid, s.x, s.n, hi, lo
        return
    r = refdata.ref("hermitian")
    m = r.methods["c2r"]
    for j, k in enumerate(m.index):
        yield "irfft", str(r.inputs["signal"][k]), r.inputs["x"][k], int(r.inputs["n"][k]), m.hi[j], m.lo[j]
    for sid in refdata.LONG_REAL:  # the exact half spectrum of a long real line (rounded), back to n reals
        s = S.get(sid)
        hi, _ = refdata.arb_dft(s.x.astype(np.complex128))
        half = hi[: s.n // 2 + 1]
        bhi, blo = refdata.arb_dft(_whole(half, s.n), inverse=True)
        yield "irfft", "h/" + sid[2:], half, s.n, bhi.real, blo.real


def _real_compute(lib, method, x, n):
    if lib == "sd":
        import stardust as sd

        f = {"rfft": sd.fft.rfft, "fft": sd.fft.fft}.get(method)
        if f is not None:
            return np.asarray(f(sd.asarray(x)))
        return np.asarray(sd.fft.irfft(sd.asarray(x), n=n, norm="forward"))
    if lib in ("numpy", "scipy"):
        if lib == "numpy":
            mod = np.fft
        else:
            import scipy.fft as mod
        if method == "irfft":
            return mod.irfft(x, n=n, norm="forward")
        return getattr(mod, method)(x)
    return None


def _real_regions(family):
    def measure(lib):
        from tests.fft import refdata

        pts = {}
        for method, sid, x, n, hi, lo in _real_lines(family):
            got = _real_compute(lib, method, x, n)
            if got is None:
                return None
            err = refdata.error(got, hi, lo)
            ynorm = refdata.norm2(hi)
            if family == "real":
                b = refdata.bound("real", n, refdata.norm2(x))
            else:
                b = refdata.bound("hermitian", n, refdata.hermitian_norm(x, n))
            p = pts.setdefault(f"{method}/{_real_form(n)}", {"where": [], "contract": [], "rel": []})
            p["where"].append(sid)
            p["contract"].append(refdata.units(err, b))
            p["rel"].append(err / ynorm if ynorm else (0.0 if err == 0 else np.inf))
        out = {}
        for method in ("rfft", "fft", "irfft"):
            for form in _REAL_FORMS:
                p = pts.get(f"{method}/{form}")
                if p:
                    out[f"{method}/{form}"] = region(p["where"], rel=np.array(p["rel"]),
                                                     contract=np.array(p["contract"]), bound=1.0)  # fmt: skip
        return out or None

    return measure


def _nd_lines():
    """(method, signal id, x, s, hi, lo) of every n-dimensional point: the stored arrays, then refdata.nd_points
    (Arb here). For irfftn, x is the half spectrum as the transform sees it and s the lengths."""
    from tests.fft import refdata

    r = refdata.ref("nd")
    for method, name in (("forward", "fftn"), ("backward", "ifftn")):
        m = r.methods[method]
        for j, k in enumerate(m.index):
            shape = tuple(int(v) for v in r.inputs["shape"][k])
            hi, lo = m.hi[j].reshape(shape), m.lo[j].reshape(shape)
            yield name, str(r.inputs["signal"][k]), r.inputs["x"][k].reshape(shape), shape, hi, lo
    names = {"c2c": "fftn", "c2c-backward": "ifftn", "real": "rfftn", "hermitian": "irfftn"}
    for family, sid, x, shape, hi, lo in refdata.nd_points():
        yield names[family], sid, x, shape, hi, lo


def _nd_form(method, s) -> str:
    from tests.fft import refdata

    real = method in ("rfftn", "irfftn")
    smooth = all(refdata.is_mixed(refdata.real_inner(v) if real and i == len(s) - 1 else v) for i, v in enumerate(s))
    return "smooth" if smooth else "bluestein"


def _nd_compute(lib, method, x, s):
    if lib == "sd":
        import stardust as sd

        f = sd.fft
        sx = sd.asarray(x)
    elif lib == "numpy":
        f = np.fft
        sx = x
    elif lib == "scipy":
        import scipy.fft as f

        sx = x
    else:
        return None
    axes = list(range(x.ndim))
    if method == "fftn":
        return np.asarray(f.fftn(sx))
    if method == "ifftn":
        return np.asarray(f.ifftn(sx, norm="forward"))
    if method == "rfftn":
        return np.asarray(f.rfftn(sx))
    return np.asarray(
        f.irfftn(sx, s=s, axes=axes, norm="forward") if lib == "numpy" else f.irfftn(sx, s=s, norm="forward")
    )


def _nd_regions(lib):
    from tests.fft import refdata

    pts = {}
    for method, sid, x, s, hi, lo in _nd_lines():
        got = _nd_compute(lib, method, x, s)
        if got is None:
            return None
        err = refdata.error(got, hi, lo)
        ynorm = refdata.norm2(hi)
        if method == "irfftn":
            b = refdata.bound_nd(s, refdata.hermitian_norm_nd(x, s[-1]), real=True)
        else:
            b = refdata.bound_nd(s, refdata.norm2(x), real=method == "rfftn")
        p = pts.setdefault(f"{method}/{_nd_form(method, s)}", {"where": [], "contract": [], "rel": []})
        p["where"].append(sid)
        p["contract"].append(refdata.units(err, b))
        p["rel"].append(err / ynorm if ynorm else (0.0 if err == 0 else np.inf))
    out = {}
    for method in ("fftn", "ifftn", "rfftn", "irfftn"):
        for form in ("smooth", "bluestein"):
            p = pts.get(f"{method}/{form}")
            if p:
                out[f"{method}/{form}"] = region(p["where"], rel=np.array(p["rel"]),
                                                 contract=np.array(p["contract"]), bound=1.0)  # fmt: skip
    return out or None


def _r2r_form(method: str, n: int) -> str:
    from tests.fft import refdata

    if not refdata.is_mixed(refdata.r2r_inner(method, n)):
        return "bluestein"
    return "even" if n % 2 == 0 else "odd"


def _r2r_lines():
    """(method, signal id, x, n, hi, lo): the stored lines of every method, then the long ones (Arb here)."""
    from tests.fft import refdata
    from tests.fft import signals as S

    r = refdata.ref("r2r")
    for method in refdata.R2R:
        m = r.methods[method]
        for j, k in enumerate(m.index):
            yield method, str(r.inputs["signal"][k]), r.inputs["x"][k], int(r.inputs["n"][k]), m.hi[j], m.lo[j]
    for sid in refdata.LONG_R2R:
        s = S.get(sid)
        for method in refdata.R2R:
            hi, lo = refdata.arb_r2r(s.x, method)
            yield method, sid, s.x, s.n, hi, lo


def _r2r_regions(lib):
    from tests.fft import refdata

    if lib == "sd":
        import stardust as sd

        mod, conv = sd.fft, sd.asarray
    elif lib == "scipy":
        import scipy.fft as mod

        conv = np.asarray
    else:
        return None
    pts = {}
    for method, sid, x, n, hi, lo in _r2r_lines():
        f = getattr(mod, method[:3])
        got = np.asarray(f(conv(x), type=int(method[-1])))
        err = refdata.error(got, hi, lo)
        ynorm = refdata.norm2(hi)
        b = refdata.bound("r2r", n, refdata.norm2(x), method=method)
        p = pts.setdefault(f"{method}/{_r2r_form(method, n)}", {"where": [], "contract": [], "rel": []})
        p["where"].append(sid)
        p["contract"].append(refdata.units(err, b))
        p["rel"].append(err / ynorm if ynorm else (0.0 if err == 0 else np.inf))
    out = {}
    for method in refdata.R2R:
        for form in ("even", "odd", "bluestein"):
            p = pts.get(f"{method}/{form}")
            if p:
                out[f"{method}/{form}"] = region(p["where"], rel=np.array(p["rel"]),
                                                 contract=np.array(p["contract"]), bound=1.0)  # fmt: skip
    return out or None


def _fht_points():
    from tests.fft import refdata

    yield from refdata.fht_points(refdata.FHT_LENGTHS + (1 << 16,))
    yield from refdata.fht_points((16, 63, 64, 1000), ("gauss", "tiny", "huge"))


def _fftlog_regions(lib):
    import warnings

    from tests.fft import refdata

    if lib == "sd":
        import stardust as sd

        mod, conv = sd.fft, sd.asarray
    elif lib == "scipy":
        import scipy.fft as mod

        conv = np.asarray
    else:
        return None
    pts = {}
    for pid, a, dln, mu, offset, bias in _fht_points():
        for name in ("fht", "ifht"):
            hi, lo = refdata.arb_fht(a, dln, mu, offset, bias, name == "ifht")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # scipy warns on the singular sets
                got = np.asarray(getattr(mod, name)(conv(a), dln, mu, offset=offset, bias=bias))
            err = refdata.error(got, hi, lo)
            ynorm = refdata.norm2(hi)
            b = refdata.bound_fht(a, dln, mu, offset, bias, name == "ifht")
            key = f"{name}/{'biased' if bias else 'unbiased'}"
            p = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
            p["where"].append(pid)
            p["contract"].append(refdata.units(err, b))
            p["rel"].append(err / ynorm if ynorm else (0.0 if err == 0 else np.inf))
    return {k: region(p["where"], rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0)
            for k, p in sorted(pts.items())} or None  # fmt: skip


def sweeps():
    real, herm = _real_regions("real"), _real_regions("hermitian")
    return [
        Custom("fft", "c2c", lambda: _regions("sd"), compare_libs(_regions, ("numpy", "scipy", "R"), seconds=600.0)),
        Custom("fft", "real", lambda: real("sd"), compare_libs(real, ("numpy", "scipy"), seconds=600.0)),
        Custom("fft", "hermitian", lambda: herm("sd"), compare_libs(herm, ("numpy", "scipy"), seconds=600.0)),
        Custom("fft", "nd", lambda: _nd_regions("sd"), compare_libs(_nd_regions, ("numpy", "scipy"), seconds=600.0)),
        Custom("fft", "r2r", lambda: _r2r_regions("sd"), compare_libs(_r2r_regions, ("scipy",), seconds=600.0)),
        Custom(
            "fft", "fftlog", lambda: _fftlog_regions("sd"), compare_libs(_fftlog_regions, ("scipy",), seconds=600.0)
        ),
    ]
