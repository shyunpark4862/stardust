"""quad's weighted routes, QAWO, QAWF, QAWS and QAWC (docs/api/integrate.md §2; docs/design/integrate.md
§3.1, §7.11; decisions C71-C83), past the collection's ten (test_quadpack.py):

  - the sweep (tests/integrate/weights.py, 99 integrals with closed forms): every run calls the
    integrand at scipy 1.18.1's points and gives its integral, error, nfev, status and message to
    the bit, but for QAWF at the frequency 0 on [a, inf), a != 0, where scipy integrates from 0
    (C78); against the Arb references every run converges within its error estimate plus the
    rounding of the value (C83);
  - a generated set of 400 runs (weights, frequencies through the three kinds of first rule,
    tolerances, limits, max_p1, limlst) against live scipy, to the bit;
  - the differences on purpose: the frequency 0 of QAWF, the moments of dqc25f once max_p1 is
    reached (C77), QAWO's single interval (C80); and what follows scipy's C port rather than the
    netlib original (dgtsl's form C76, dqawoe's size test C79, the corrected dqc25s C75)."""

import math
import random
import warnings

import numpy as np
import pytest

import stardust as sd

from . import refdata
from . import weights as W
from .test_reference_data import live_scipy

I = sd.integrate  # noqa: E741
EPS = 2.0**-52
ZERO_FREQUENCY = {"weights/qawf/exp-cos/1.0/0.0/1.0"}  # C78: scipy integrates from 0


def _sd(f, a, b, **kw):
    pts = []
    r = I.quad(lambda x: (pts.append(x), f(x))[1], a, b, **kw)
    return r, np.array(pts, dtype=np.float64)


def _scipy(f, a, b, **kw):
    from scipy import integrate

    pts = []
    names = {"atol": "epsabs", "rtol": "epsrel", "max_p1": "maxp1"}
    kw = {names.get(k, k): v for k, v in kw.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = integrate.quad(lambda x: (pts.append(x), f(x))[1], a, b, full_output=1, **kw)
    return r, np.array(pts, dtype=np.float64)


def _bits(x):
    return np.float64(x).view(np.int64).item()


def same_as_scipy(f, a, b, **kw):
    """sd's quad and scipy's on the same call: the same points, integral, error, nfev, message."""
    r, pts = _sd(f, a, b, **kw)
    s, spts = _scipy(f, a, b, **kw)
    ok = pts.size == spts.size and np.array_equal(pts.view(np.int64), spts.view(np.int64))
    ok = ok and _bits(r.integral) == _bits(s[0]) and _bits(r.error) == _bits(s[1]) and r.nfev == s[2]["neval"]
    if len(s) > 3:
        ok = ok and not r.converged and r.message == s[3]
    else:
        ok = ok and r.converged
    return ok, r, s


# ---- the sweep


def test_sweep_references_are_current():
    import sys

    sys.path.insert(0, str(refdata.REF.parents[2] / "tools"))
    try:
        from integrate_ref import integral_digest
    finally:
        sys.path.pop(0)
    ref = refdata.weights()
    assert set(ref) == set(W.BY_ID)
    for p in W.SWEEP:
        assert ref[p.id]["digest"] == integral_digest(p), p.id
    counts = {}
    for p in W.SWEEP:
        counts[W.routine(p)] = counts.get(W.routine(p), 0) + 1
    assert counts == {"qawo": 18, "qawf": 17, "qaws": 40, "qawc": 24}


@pytest.mark.parametrize("p", W.SWEEP, ids=lambda p: p.id)
def test_sweep_matches_scipy(p):
    live_scipy()
    ok, r, s = same_as_scipy(p.f, p.a, p.b, weight=p.weight, wvar=p.wvar)
    assert ok is (p.id not in ZERO_FREQUENCY)


def test_sweep_against_the_references():
    # every run converges; the true error is within the estimate but for 16 runs, all within it
    # plus 8 eps |I| (a Clenshaw-Curtis estimate has no rounding floor: polynomial integrands of
    # QAWC give 0, and dqc25f's moments are good to about 2e3 ulp; scipy gives the same bits)
    ref = refdata.weights()
    short = set()
    for p in W.SWEEP:
        r = I.quad(p.f, p.a, p.b, weight=p.weight, wvar=p.wvar)
        hi, lo = ref[p.id]["value"]
        err = abs((r.integral - hi) - lo)
        assert r.converged and err <= r.error + 8 * EPS * abs(hi), p.id
        if err > r.error:
            short.add(p.id)
    assert len(short) == 16 and {W.routine(W.BY_ID[i]) for i in short} == {"qawo", "qawc"}


# ---- a generated set against live scipy

_F = {
    "exp": lambda x: math.exp(min(x, 700.0)),
    "expm": lambda x: math.exp(min(-x, 700.0)),
    "log1p": lambda x: math.log1p(abs(x)),
    "poly": lambda x: 1 + x * (2 - 3 * x * x),
    "rat": lambda x: 1 / (1 + x * x),
    "sqrt": lambda x: math.sqrt(abs(x)),
    "cos3": lambda x: math.cos(3 * x),
    "logabs": lambda x: math.log(abs(x)) if x != 0 else 0.0,
    "peak": lambda x: 1 / ((x - 0.3) ** 2 + 1e-3),
    "gauss": lambda x: math.exp(-x * x),
    "osc": lambda x: math.sin(30 * x) * math.exp(min(-x, 700.0)),
    "step": lambda x: 1.0 if x < 0.37 else 0.5,
}


def _generated(seed=5, n=400):
    """(name, f, a, b, kw): quad's weighted options drawn over the routes and the first rules
    (|omega (b - a)/2| <= 2, <= 24, > 24), the tolerances and limits; max_p1 50 or 100 (not
    reached) and frequency 0 only from 0 (the differences of C77, C78 are tested below)."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        fn = rnd.choice(list(_F))
        kind = rnd.choice(["cos", "sin", "alg", "alg-loga", "alg-logb", "alg-log", "cauchy", "qawf"])
        atol, rtol = rnd.choice([(1.49e-8, 1.49e-8), (1e-12, 0.0), (0.0, 1e-10), (1e-4, 1e-4), (1e-14, 1e-14)])
        limit = rnd.choice([50, 50, 3, 10, 200])
        a = rnd.choice([0.0, -1.0, 0.5, 2.0, -3.3])
        b = a + rnd.choice([1.0, 0.25, 7.0, 30.0, 1e-3])
        kw = {"atol": atol, "rtol": rtol, "limit": limit}
        if kind in ("cos", "sin"):
            kw.update(weight=kind, wvar=rnd.choice([0.0, 0.5, 3.0, 10 * math.pi, 37.0, 300.0, 2e3, -7.0, 1e5]),
                      max_p1=rnd.choice([50, 100]))  # fmt: skip
        elif kind == "qawf":
            w = rnd.choice([0.5, 1.0, 3.0, 10.0, 0.01, -2.0, 100.0])
            a = rnd.choice([0.0, 1.0, -2.0, 5.0])
            if rnd.random() < 0.3:
                a, b, fn = -math.inf, a, rnd.choice(["exp", "rat", "gauss"])
            else:
                b, fn = math.inf, rnd.choice(["expm", "rat", "gauss", "osc"])
            if rnd.random() < 0.1:
                w, a, b = 0.0, 0.0, math.inf
            kw = {"atol": rnd.choice([1.49e-8, 1e-12, 1e-4]), "limit": limit, "weight": rnd.choice(["cos", "sin"]),
                  "wvar": w, "max_p1": rnd.choice([50, 100]), "limlst": rnd.choice([50, 3, 10])}  # fmt: skip
        elif kind.startswith("alg"):
            kw.update(weight=kind, wvar=(rnd.choice([0.0, -0.5, 0.5, -0.9, 2.0, -0.999, 1.5]),
                      rnd.choice([0.0, -0.5, 0.3, -0.75, 3.0])), limit=max(limit, 2))  # fmt: skip
        else:
            kw.update(weight="cauchy", wvar=a + (b - a) * rnd.choice([0.5, 0.31, 0.999, -0.3, 1.7, 0.05, 1.04, -0.02]))
        out.append((f"{i}-{fn}-{kind}", _F[fn], a, b, kw))
    return out


GENERATED = _generated()


def test_generated_set_matches_scipy():
    live_scipy()
    bad = [name for name, f, a, b, kw in GENERATED if not same_as_scipy(f, a, b, **kw)[0]]
    statuses = {I.quad(f, a, b, **kw).status for _, f, a, b, kw in GENERATED}
    assert bad == [] and len(GENERATED) == 400 and statuses == {0, 1, 2, 7}


# ---- the differences on purpose and what follows scipy's port


def test_qawf_zero_frequency():
    # C78: QUADPACK's dqawfe calls dqagie(f, 0, 1, ...) for omega = 0 whatever a is (scipy too:
    # it gives the integral from 0); here it is QAGI on [a, inf) with dqawfe's epsrel = 0
    f = lambda x: math.exp(-x)  # noqa: E731
    r = I.quad(f, 1.0, math.inf, weight="cos", wvar=0.0)
    g = I.quad(f, 1.0, math.inf, atol=1.49e-8, rtol=0.0)
    assert (r.integral, r.error, r.nfev) == (g.integral, g.error, g.nfev) and abs(r.integral - math.exp(-1)) < 1e-15
    assert r.n_cycles == 1 and r.cycle_integrals.tolist() == [r.integral] and r.converged
    s = I.quad(f, 1.0, math.inf, weight="sin", wvar=0.0)
    assert tuple(s) == (0.0, 0.0) and s.nfev == 0 and s.n_cycles == 1
    # from 0 it is scipy's (the left half line too, x -> -x onto [-b, inf))
    live_scipy()
    assert same_as_scipy(f, 0.0, math.inf, weight="cos", wvar=0.0)[0]
    assert not same_as_scipy(f, 1.0, math.inf, weight="cos", wvar=0.0)[0]
    left = I.quad(lambda x: math.exp(x), -math.inf, -1.0, weight="cos", wvar=0.0)
    assert abs(left.integral - math.exp(-1)) < 1e-15


def test_moments_past_max_p1():
    # C77: once max_p1 - 1 levels of moments are kept, dqc25f recomputes the last row for each new
    # level and the second half of a bisection (ksave = 1) takes the row of the call before it
    # (the Fortran's m, kept between calls); scipy's C port starts m at row 1 on every call and
    # gives the second half the moments of level 0: wrong values reported as converged
    live_scipy()
    f = lambda x: math.log(x) if x > 0 else 0.0  # noqa: E731
    want = I.quad(f, 0.0, 1.0, weight="cos", wvar=300.0)
    assert want.converged and want.n_intervals == 13
    for p in (1, 2, 3, 5, 8):
        r = I.quad(f, 0.0, 1.0, weight="cos", wvar=300.0, max_p1=p)
        assert r.converged and abs(r.integral - want.integral) <= 2 * (r.error + want.error), p
        s, _ = _scipy(f, 0.0, 1.0, weight="cos", wvar=300.0, max_p1=p)
        assert (abs(s[0] - want.integral) > 1e-3) is (p in (2, 3, 5)), p  # scipy: off by 1e-2
    assert same_as_scipy(f, 0.0, 1.0, weight="cos", wvar=300.0, max_p1=8)[0]  # never reached: the same


def test_qawo_single_interval():
    # C80: dqawoe returns last = 0 when its first rule ends it (scipy's infodict 'last' 0); the
    # result has the one interval
    r = I.quad(lambda x: math.exp(x) * x, 0.0, 1.0, weight="cos", wvar=300.0)
    assert r.converged and r.nfev == 25 and r.n_intervals == 1 and r.intervals.tolist() == [[0.0, 1.0]]
    assert r.interval_integrals.tolist() == [r.integral] and r.interval_errors.tolist() == [r.error]
    r = I.quad(math.exp, 0.0, 1.0, weight="sin", wvar=1.0, limit=1)
    assert r.n_intervals == 1 and r.nfev == 15 and r.status == 1


def test_ports_differences_from_the_original():
    live_scipy()
    # C79: dqawoe's test of a too small interval has 1000 epmach where QUADPACK has 1000 uflow:
    # bisecting toward 0.37 stops at a width of 2e-13 with status 3, as scipy
    ok, r, _ = same_as_scipy(lambda x: 1.0 if x < 0.37 else 0.5, -3.3, 3.7, weight="sin", wvar=300.0, atol=1e-14,
                             rtol=1e-14)  # fmt: skip
    assert ok and r.status == 3 and r.n_intervals == 46
    # C76: dgtsl in the form of scipy's port (the last row not eliminated); |omega h| near 24
    # takes it with moments of order 25 that the last row moves by ulps
    for w in (38.8, 40.0, 44.0, 47.5):
        assert same_as_scipy(lambda x: math.exp(-x), 0.0, 1.0, weight="cos", wvar=w, limit=1, atol=1e-14)[0], w
        assert same_as_scipy(lambda x: math.exp(-x), 0.0, 1.0, weight="sin", wvar=w, limit=1, atol=1e-14)[0], w
    # C75: dqc25s without the netlib typo (res24 = res12 + ...): the QAWS log example's value is
    # scipy's, 4e-9 from the Fortran original's (tests/integrate/data/fortran_vs_scipy.json)
    c = refdata.data("fortran_vs_scipy.json")["quadpack/qaws"]
    assert c["routine"] == "dqawse" and c["same_calls"] and not c["same"]["integral"]


def test_huge_frequency():
    # |omega| >= 2^30: scipy's 32-bit 2 floor(|omega|) + 1 overflows (its cycle comes out
    # negative); here it is 64-bit and the cycle (2 floor|omega| + 1) pi/|omega|, about 2 pi
    w = 2.0**31 + 0.5
    r = I.quad(lambda x: math.exp(-x), 0.0, math.inf, weight="cos", wvar=w)
    assert r.converged and r.n_cycles == 6 and abs(r.integral - 1 / (1 + w * w)) < 1e-18


def test_complex_and_nested_weights_match_scipy():
    live_scipy()
    from scipy import integrate

    f = lambda x: complex(math.exp(-x), x * x)  # noqa: E731
    for kw in ({"weight": "cos", "wvar": 3.0}, {"weight": "alg-loga", "wvar": (-0.5, 0.25)},
               {"weight": "cauchy", "wvar": 0.4}):  # fmt: skip
        r = I.quad(f, 0.0, 1.0, complex_func=True, **kw)
        s = integrate.quad(f, 0.0, 1.0, complex_func=True, full_output=1, **kw)
        assert r.integral == s[0] and r.error == s[1] and r.converged
    g = lambda x: complex(math.exp(-x), math.exp(-2 * x))  # noqa: E731
    r = I.quad(g, 0.0, math.inf, weight="sin", wvar=2.0, complex_func=True)
    s = integrate.quad(g, 0.0, math.inf, weight="sin", wvar=2.0, complex_func=True, full_output=1)
    assert r.integral == s[0] and r.error == s[1] and r.real.n_cycles == s[2]["real"][0]["lst"]
    assert "cycles" in r.summary()
    # nquad: a level with a weight, given as a dict and as a function of the outer variable
    h = lambda x, y: math.exp(-x) * (1 + y)  # noqa: E731
    seen_sd, seen_sp = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = integrate.nquad(lambda *v: (seen_sp.append(v), h(*v))[1], [[0, math.inf], [0, 1]],
                            opts=[lambda y: {"weight": "cos", "wvar": 1 + y}, {"weight": "alg", "wvar": (0.5, -0.5)}],
                            full_output=True)  # fmt: skip
    r = I.nquad(lambda *v: (seen_sd.append(v), h(*v))[1], [[0, math.inf], [0, 1]],
                opts=[lambda y: {"weight": "cos", "wvar": 1 + y}, {"weight": "alg", "wvar": (0.5, -0.5)}])  # fmt: skip
    assert seen_sd == seen_sp and (r.integral, r.error, r.nfev) == (s[0], s[1], s[2]["neval"])


def test_qawf_result_and_messages():
    live_scipy()
    # the cycles are scipy's lst, rslst, erlst, ierlst; ier 1 (limlst) has QAWF's own text
    f = lambda x: 1 / math.sqrt(x) if x > 0 else 0.0  # noqa: E731
    ok, r, s = same_as_scipy(f, 0.0, math.inf, weight="cos", wvar=1.0, limlst=3)
    assert ok and r.status == 1 and r.message.startswith("The maximum number of cycles allowed has been achieved.")
    assert r.n_cycles == s[2]["lst"] == 3 and r.cycle_status.tolist() == s[2]["ierlst"][:3].tolist()
    for name in ("intervals", "n_intervals"):
        assert not hasattr(r, name)
    with pytest.raises(ValueError):
        r.cycle_integrals[0] = 1.0
    txt = r.summary()
    assert "Cycles:" in txt and "cycles allowed" in txt and "[2]" in txt
    # reversed limits change the sign only
    a, b = I.quad(f, 0.0, math.inf, weight="sin", wvar=2.0), I.quad(f, math.inf, 0.0, weight="sin", wvar=2.0)
    assert b.integral == -a.integral and b.error == a.error and b.cycle_integrals.tolist() == a.cycle_integrals.tolist()


@pytest.mark.parametrize("pid", sorted(W.BY_ID))
def test_sweep_matches_the_scipy_records(pid):
    """Every weighted integral of the sweep against scipy 1.18.1's record (tests/integrate/data/
    scipy_weights.json, tools/gen_integrate_scipy.py weights): the same points in call order,
    integral, error, nfev, status, message and intervals (QAWF: cycles), every alg weight among them."""
    import hashlib

    rec = refdata.data("scipy_weights.json")[pid]
    p = W.BY_ID[pid]
    kw = {"weight": p.weight, "wvar": p.wvar}
    r, pts = _sd(p.f, p.a, p.b, **kw)
    if pid in ZERO_FREQUENCY:  # C78: scipy integrates from 0 whatever a is; sd is QAGI on [a, inf)
        assert _bits(r.integral) != _bits(rec["integral"])
        return
    assert hashlib.sha256(np.ascontiguousarray(pts).tobytes()).hexdigest() == rec["points_sha256"]
    assert _bits(r.integral) == _bits(rec["integral"]) and _bits(r.error) == _bits(rec["error"])
    assert r.nfev == rec["nfev"] and r.status == rec["status"] and (r.message if r.status else None) == rec["message"]
    info = rec["info"]
    if "lst" in info:  # QAWF: its cycles
        assert r.n_cycles == info["lst"]
    elif "last" in info:
        assert r.n_intervals == max(int(info["last"]), 1)
