"""savgol_coeffs and savgol_filter (docs/api/smooth.md §6; docs/design/smooth.md §3.7, §4.6; decisions SM24, SM25,
SM32, SM70-SM76): the exact-coefficient contract against the rational references and against Arb at orders up to
the window, the filter's contract in every mode, scipy's meaning (modes, even windows, positions, interp's ends,
axes), complex input, NaN, the inputs, the errors, and the same bits whatever the layout and the thread count."""

import math
from fractions import Fraction

import numpy as np
import pytest
from flint import arb, ctx

import stardust as sd

from . import refdata
from . import samples as S
from .conftest import LIVE

MODES = S.SAVGOL_MODES


def per_set(r, method, sid):
    pos, idx = refdata.at(r, method, sid)
    m = r.methods[method]
    order = np.argsort(idx, kind="stable")
    pos, idx = pos[order], idx[order]
    return idx, m.hi[pos], m.lo[pos], {k: v[pos] for k, v in m.fields.items()}


def npa(a):
    return np.asarray(a)


def arb_coeffs(n, d, deriv, pos, prec):
    """The coefficients (use='dot', delta = 1) by the orthonormal Gram recurrence in Arb balls at prec bits:
    (midpoints as Fractions, the largest radius)."""
    old = ctx.prec
    ctx.prec = prec
    try:
        c = arb(n - 1) / 2
        s = [arb(i) - c for i in range(n)]
        sp = (arb(2 * (n // 2) - 1) / 2 if n % 2 == 0 else arb(n // 2)) if pos is None else arb(pos)
        sp = sp - c
        n2 = arb(n) * n
        sqb = [arb(0)] + [((arb(k) * k * (n2 - k * k)) / (4 * (4 * arb(k) * k - 1))).sqrt() for k in range(1, d + 1)]
        q0 = 1 / arb(n).sqrt()
        prev, cur = [arb(0)] * n, [q0] * n
        dprev, dcur = [arb(0)] * (deriv + 1), [q0] + [arb(0)] * deriv
        out = [arb(0)] * n
        for k in range(d + 1):
            if k >= deriv:
                out = [o + dcur[deriv] * q for o, q in zip(out, cur)]
            if k == d:
                break
            nxt = [(si * q - sqb[k] * pq) / sqb[k + 1] for si, q, pq in zip(s, cur, prev)]
            dn = [
                (sp * dcur[v] + (v * dcur[v - 1] if v else 0) - sqb[k] * dprev[v]) / sqb[k + 1]
                for v in range(deriv + 1)
            ]
            prev, cur, dprev, dcur = cur, nxt, dcur, dn
        mids = []
        for o in out:
            man, exp = o.mid().man_exp()
            mids.append(Fraction(int(man)) * Fraction(2) ** int(exp))
        return mids, max(float(o.rad()) for o in out)
    finally:
        ctx.prec = old


# ---- the coefficient contract ------------------------------------------------------------------


def test_coefficients_within_the_contract_of_the_references():
    r = refdata.ref("savgol")
    worst = 0.0
    for case in S.SAVGOL:
        n, d, deriv, pos = case
        sid = S.savgol_id(case)
        idx, hi, lo, _ = per_set(r, "coeffs", sid)
        g = r.group_inputs
        norm = float(g["norm_inf"][refdata.group_of(r, sid)])
        got = npa(sd.smooth.savgol_coeffs(n, d, deriv=deriv, pos=pos, use="dot"))
        assert got.shape == (n,) and got.dtype == np.float64
        bound = refdata.savgol_bound(norm) if norm > 0 else refdata.U
        for i, h, l_ in zip(idx, hi, lo):
            u = refdata.units(float(got[i]), float(h), float(l_), bound)
            worst = max(worst, u)
            assert u <= 1.0, (case, int(i), u)
        conv = npa(sd.smooth.savgol_coeffs(n, d, deriv=deriv, pos=pos))
        assert np.array_equal(conv, got[::-1]), case
    assert worst <= 1.0


@pytest.mark.parametrize(
    "case",
    [
        (64, 63, 0, None),  # an even window at full degree: the Lagrange weights at the half-step centre
        (101, 100, 0, None),  # interpolation: the unit vector
        (101, 99, 0, 0),
        (101, 90, 1, 3),
        (121, 80, 2, 0),
        (201, 150, 0, 0),
        (201, 200, 1, 100),
        (60, 50, 4, 7),
    ],
)
def test_coefficients_at_orders_up_to_the_window_against_arb(case):
    """Where the degree recurrence at the window's points has lost every digit (d beyond ~N/4 at 106 bits;
    scipy's Vandermonde too), the coefficients are still within 8 u max|h| (decision SM70)."""
    n, d, deriv, pos = case
    ref, rad = arb_coeffs(n, d, deriv, pos, 6 * n + 300)
    ref = np.array([float(v) for v in ref])
    assert rad <= 2.0**-80 * np.abs(ref).max()
    got = npa(sd.smooth.savgol_coeffs(n, d, deriv=deriv, pos=pos, use="dot"))
    err = np.abs(got - ref).max() / (refdata.C_SAVGOL * refdata.U * np.abs(ref).max())
    assert err <= 1.0 / 8, (case, err)  # one rounding: 0.5 u max|h| at most, a sixteenth of the contract


def test_coefficients_reproduce_polynomials_and_their_derivatives():
    for n, d, deriv, pos in [(9, 4, 0, None), (9, 4, 2, 1), (12, 5, 1, None), (31, 6, 3, 30), (7, 6, 6, 0)]:
        h = npa(sd.smooth.savgol_coeffs(n, d, deriv=deriv, pos=pos, use="dot"))
        p = (n - 1) / 2 if pos is None else pos
        t = np.arange(n) - p
        for j in range(d + 1):
            want = math.factorial(deriv) if j == deriv else 0.0
            got = math.fsum(h * t**j)
            assert abs(got - want) <= 64 * refdata.U * math.fsum(np.abs(h * t**j)), (n, d, deriv, pos, j)


def test_delta_divides_by_its_power_exactly_once():
    h1 = npa(sd.smooth.savgol_coeffs(11, 4, deriv=2, use="dot"))
    h = npa(sd.smooth.savgol_coeffs(11, 4, deriv=2, delta=0.5, use="dot"))
    assert np.array_equal(h, h1 * 4.0)  # a power of two: the same rounding
    h3 = npa(sd.smooth.savgol_coeffs(11, 4, deriv=3, delta=0.1, use="dot"))
    ref = [Fraction(v) / Fraction(0.1) ** 3 for v in arb_coeffs(11, 4, 3, None, 300)[0]]
    top = max(abs(float(q)) for q in ref)
    assert max(abs(float(Fraction(g) - q)) for g, q in zip(h3, ref)) <= 0.5 * refdata.U * top
    # delta^3 under- and overflows on its own; the coefficients do not
    big = npa(sd.smooth.savgol_coeffs(7, 3, deriv=3, delta=2.0**-120, use="dot"))
    h3 = npa(sd.smooth.savgol_coeffs(7, 3, deriv=3, use="dot"))
    assert np.array_equal(big, h3 * 2.0**360)
    small = npa(sd.smooth.savgol_coeffs(7, 3, deriv=3, delta=2.0**130, use="dot"))
    assert np.array_equal(small, h3 * 2.0**-390)


def test_positions_mirror_bit_for_bit():
    for n, d, deriv in [(11, 3, 0), (11, 3, 1), (10, 4, 2), (51, 30, 3)]:
        for p in range(n):
            a = npa(sd.smooth.savgol_coeffs(n, d, deriv=deriv, pos=p, use="dot"))
            b = npa(sd.smooth.savgol_coeffs(n, d, deriv=deriv, pos=n - 1 - p, use="dot"))
            assert np.array_equal(a, (-1) ** deriv * b[::-1]), (n, d, deriv, p)
    c = npa(sd.smooth.savgol_coeffs(11, 4, deriv=1, use="dot"))
    assert c[5] == 0 and np.array_equal(c, -c[::-1])


def test_high_derivatives_and_small_windows():
    assert np.array_equal(npa(sd.smooth.savgol_coeffs(7, 3, deriv=4)), np.zeros(7))
    assert npa(sd.smooth.savgol_coeffs(1, 0)).tolist() == [1.0]
    assert npa(sd.smooth.savgol_coeffs(2, 1, use="dot")).tolist() == [0.5, 0.5]
    assert npa(sd.smooth.savgol_coeffs(2, 1, deriv=1, use="dot")).tolist() == [-1.0, 1.0]
    # interpolation (polyorder = window_length - 1) at a point of the window: the unit vector, its zeros to the
    # double-double sum's rounding (no snapping to 0; decision SM70)
    e = npa(sd.smooth.savgol_coeffs(3, 2, pos=0, use="dot"))
    assert e[0] == 1.0 and np.all(np.abs(e[1:]) <= 2.0**-100)


@LIVE
def test_scipy_agrees_at_low_orders_and_goes_wrong_at_high_ones():
    from scipy.signal import savgol_coeffs

    for n, d, deriv, pos in [(5, 2, 0, None), (11, 3, 1, 2), (10, 3, 0, None), (25, 5, 2, None), (101, 3, 0, None)]:
        a = npa(sd.smooth.savgol_coeffs(n, d, deriv=deriv, pos=pos))
        b = savgol_coeffs(n, d, deriv=deriv, pos=pos)
        assert np.abs(a - b).max() <= 2e4 * refdata.U * np.abs(a).max(), (n, d)
    exact = npa(sd.smooth.savgol_coeffs(101, 10))
    assert abs(math.fsum(exact) - 1.0) <= 1e-15
    assert abs(math.fsum(savgol_coeffs(101, 10)) - 1.0) > 0.5  # scipy: 1.1e-4, no warning


# ---- the filter ---------------------------------------------------------------------------------


def test_filter_within_the_contract_in_every_mode():
    r = refdata.ref("savgol_filter")
    worst = 0.0
    for sid in r.group_inputs["set"].tolist():
        sig, f = sid.split("|")
        n, d, deriv = map(int, f.split(","))
        x = S.signal(sig).x
        for mode in MODES:
            got = npa(sd.smooth.savgol_filter(x, n, d, deriv=deriv, mode=mode, cval=S.SAVGOL_CVAL))
            idx, hi, lo, fields = per_set(r, mode, sid)
            for i, h, l_, sc in zip(idx, hi, lo, fields["scale"]):
                u = refdata.units(float(got[i]), float(h), float(l_), refdata.savgol_filter_bound(n, float(sc)))
                worst = max(worst, u)
                assert u <= 1.0, (sid, mode, int(i), u)
    assert worst <= 1.0


@LIVE
@pytest.mark.parametrize("mode", MODES)
def test_scipy_meaning_in_every_mode(mode):
    from scipy.signal import savgol_filter

    g = np.random.default_rng(7)
    for n in (1, 2, 5, 11, 40):
        x = g.standard_normal(n)
        for N, d, deriv in [(1, 0, 0), (3, 1, 0), (4, 2, 1), (5, 2, 0), (7, 3, 2), (10, 3, 0), (11, 4, 5), (12, 5, 1)]:
            if mode == "interp" and N > n:
                with pytest.raises(ValueError, match="must not exceed the length of the axis"):
                    sd.smooth.savgol_filter(x, N, d, deriv=deriv, mode=mode)
                continue
            a = npa(sd.smooth.savgol_filter(x, N, d, deriv=deriv, delta=0.5, mode=mode, cval=-1.25))
            b = savgol_filter(x, N, d, deriv=deriv, delta=0.5, mode=mode, cval=-1.25)
            assert a.shape == b.shape
            assert np.abs(a - b).max() <= 1e-11 * max(1.0, np.abs(b).max()), (n, N, d, deriv)


@LIVE
def test_scipy_meaning_along_every_axis():
    from scipy.signal import savgol_filter

    x = np.random.default_rng(3).standard_normal((4, 9, 6))
    for axis in (0, 1, 2, -1, -3):
        for mode in MODES:
            N = 3 if mode == "interp" and axis in (0, -3) else 5
            a = npa(sd.smooth.savgol_filter(x, N, 2, axis=axis, mode=mode, deriv=1))
            b = savgol_filter(x, N, 2, axis=axis, mode=mode, deriv=1)
            assert np.abs(a - b).max() <= 1e-12, (axis, mode)


def test_polynomials_come_back_in_interp_mode():
    """A cubic and its derivatives come back from every filter of polyorder >= 3: at the samples for an odd window
    and at the ends, half a step after the sample in the middle of an even window (its centre)."""
    k = np.arange(40.0)
    x = S.signal("cubic/40").x
    fs = (
        lambda t: 0.125 * t**3 - 2.0 * t * t + 3.0 * t - 7.0,
        lambda t: 0.375 * t * t - 4 * t + 3,
        lambda t: 0.75 * t - 4,
        lambda t: np.full(t.shape, 0.75),
        lambda t: np.zeros(t.shape),
    )
    for N, d in [(7, 3), (9, 4), (11, 5), (8, 3), (40, 3)]:
        half = N // 2
        t = k.copy()
        if N % 2 == 0:
            t[half : 40 - half] += 0.5
        for deriv in range(5):
            got = npa(sd.smooth.savgol_filter(x, N, d, deriv=deriv))
            tol = 1e-9 * max(1.0, np.abs(x).max())
            assert np.abs(got - fs[deriv](t)).max() <= tol, (N, d, deriv)


def test_layouts_give_the_same_bits():
    g = np.random.default_rng(11)
    X = g.standard_normal((7, 50, 5))
    for mode in MODES:
        for N, d, deriv in [(7, 3, 0), (10, 4, 1), (11, 2, 2)]:
            for axis in (0, 1, 2):
                if mode == "interp" and X.shape[axis] < N:
                    continue
                full = npa(sd.smooth.savgol_filter(X, N, d, deriv=deriv, axis=axis, mode=mode, cval=2.0))
                moved = np.moveaxis(X, axis, -1)
                for idx in np.ndindex(moved.shape[:-1]):
                    line = npa(sd.smooth.savgol_filter(np.ascontiguousarray(moved[idx]), N, d, deriv=deriv,
                                                       mode=mode, cval=2.0))  # fmt: skip
                    assert np.array_equal(np.moveaxis(full, axis, -1)[idx], line), (mode, axis, idx)
    T = sd.asarray(X)[:, ::2, :]  # a strided view
    assert np.array_equal(npa(sd.smooth.savgol_filter(T, 5, 2, axis=1)),
                          npa(sd.smooth.savgol_filter(np.ascontiguousarray(X[:, ::2, :]), 5, 2, axis=1)))  # fmt: skip


def test_complex_filters_both_parts():
    z = S.signal("complex/40").x
    for mode in MODES:
        for N, d, deriv in [(7, 3, 0), (8, 3, 1)]:
            got = npa(sd.smooth.savgol_filter(z, N, d, deriv=deriv, mode=mode, cval=0.5))
            assert got.dtype == np.complex128
            re = npa(sd.smooth.savgol_filter(z.real.copy(), N, d, deriv=deriv, mode=mode, cval=0.5))
            im = npa(sd.smooth.savgol_filter(z.imag.copy(), N, d, deriv=deriv, mode=mode, cval=0.0))
            assert np.array_equal(got.real, re) and np.array_equal(got.imag, im), (mode, N)


@pytest.mark.parametrize("mode", MODES)
def test_nan_reaches_exactly_the_windows_that_hold_it(mode):
    n, N = 30, 7
    for where in (0, 2, 14, 29):
        x = np.linspace(-1, 2, n) ** 2
        x[where] = np.nan
        got = npa(sd.smooth.savgol_filter(x, N, 3, mode=mode))
        ext = np.arange(n)
        want = np.zeros(n, bool)
        for i in range(n):
            if mode == "interp" and i < N // 2:
                win = range(0, N)
            elif mode == "interp" and i >= n - N // 2:
                win = range(n - N, n)
            else:
                win = []
                for j in range(N):
                    k = i - N // 2 + j
                    if 0 <= k < n:
                        win.append(k)
                    elif mode == "nearest":
                        win.append(0 if k < 0 else n - 1)
                    elif mode == "wrap":
                        win.append(k % n)
                    elif mode == "mirror":
                        r = k % (2 * n - 2)
                        win.append(r if r < n else 2 * n - 2 - r)
            want[i] = where in [ext[k] for k in win]
        assert np.array_equal(np.isnan(got), want), (mode, where)


def test_inputs_and_dtypes():
    ref = npa(sd.smooth.savgol_filter(np.arange(20.0), 5, 2))
    assert np.array_equal(npa(sd.smooth.savgol_filter(list(range(20)), 5, 2)), ref)
    assert np.array_equal(npa(sd.smooth.savgol_filter(np.arange(20), 5, 2)), ref)
    assert np.array_equal(npa(sd.smooth.savgol_filter(sd.arange(20), 5, 2)), ref)
    out = sd.smooth.savgol_filter(sd.frame.Series(np.arange(20.0)), 5, 2)
    assert isinstance(out, sd.ndarray) and np.array_equal(npa(out), ref)
    b = npa(sd.smooth.savgol_filter([True, False, True, True, False], 3, 1))
    assert b.dtype == np.float64
    assert npa(sd.smooth.savgol_filter(np.zeros((3, 0)), 5, 2, mode="mirror")).shape == (3, 0)
    assert npa(sd.smooth.savgol_filter(np.zeros((0, 8)), 5, 2)).shape == (0, 8)


def test_errors_and_messages():
    f, c = sd.smooth.savgol_filter, sd.smooth.savgol_coeffs
    for call, exc, msg in [
        (lambda: c(5, 5), ValueError, r"savgol_coeffs\(\): 'polyorder' must be less than 'window_length' \(5 >= 5\)"),
        (lambda: c(0, 0), ValueError, "'window_length' must be positive"),
        (lambda: c(5, -1), ValueError, "'polyorder' must be nonnegative"),
        (lambda: c(5, 2, deriv=-1), ValueError, "'deriv' must be nonnegative"),
        (lambda: c(5, 2, pos=5), ValueError, "'pos' must be nonnegative and less than 'window_length'"),
        (lambda: c(5, 2, pos=1.5), TypeError, "'pos' must be an int, not float"),
        (lambda: c(5.0, 2), TypeError, "'window_length' must be an int, not float"),
        (lambda: c(True, 0), TypeError, "'window_length' must be an int, not bool"),
        (lambda: c(5, 2, use="full"), ValueError, "'use' must be 'conv' or 'dot', not 'full'"),
        (lambda: c(5, 2, delta=0.0), ValueError, "'delta' must be finite and nonzero"),
        (lambda: c(5, 2, delta=math.inf), ValueError, "'delta' must be finite and nonzero"),
        (lambda: f(np.ones(5), 7, 3), ValueError, r"savgol_filter\(\): 'window_length' must not exceed .* \(7 > 5\)"),
        (lambda: f(np.ones(9), 7, 3, mode="reflect"), ValueError, "'mode' must be 'interp', 'mirror', "),
        (lambda: f(np.ones(9), 7, 3, mode=1), TypeError, "'mode' must be"),
        (lambda: f(np.ones(9), 7, 3, axis=1), ValueError, "axis 1 is out of bounds for an array of dimension 1"),
        (lambda: f(np.ones(9), 7, 3, cval="a"), TypeError, "'cval' must be an int or a float"),
        (lambda: f(3.0, 1, 0), ValueError, "'x' must have at least 1 dimension"),
        (lambda: f("abc", 1, 0), TypeError, "'x' must be an sd.ndarray"),
        (lambda: f(np.array(["a", "b"]), 1, 0), TypeError, "'x' must be an sd.ndarray"),
        (lambda: f(np.ones(9), 7, 3, 1), TypeError, "positional"),
    ]:
        with pytest.raises(exc, match=msg):
            call()


@pytest.mark.usefixtures("restore_threads")
def test_threads_give_the_same_bytes():
    g = np.random.default_rng(5)
    cases = [
        (g.standard_normal(300_001), (101, 3), {}),
        (g.standard_normal(200_000), (11, 3), dict(mode="mirror", deriv=1)),
        (g.standard_normal((700, 600)), (11, 3), dict(axis=0)),
        (g.standard_normal((600, 701)), (25, 5), dict(axis=1, mode="wrap")),
        (g.standard_normal((60, 50, 40)) + 1j * g.standard_normal((60, 50, 40)), (9, 2), dict(axis=1)),
        (g.standard_normal((3, 20_000)), (1001, 5), {}),
    ]
    sd.set_num_threads(1)
    base = [npa(sd.smooth.savgol_filter(x, *wp, **kw)).tobytes() for x, wp, kw in cases]
    for t in (2, 3, 7, 10):
        sd.set_num_threads(t)
        for (x, wp, kw), b in zip(cases, base):
            assert npa(sd.smooth.savgol_filter(x, *wp, **kw)).tobytes() == b, (t, wp, kw)
