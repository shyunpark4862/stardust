"""The weighted integrals of quad's sweep (docs/design/integrate.md §7.11, milestone C5): families of
f(x) w(x) for each of QUADPACK's weighted routines with closed forms in Arb, past the collection's
ten weighted problems (tests/integrate/problems.py). Each is a problems.Integral (the integrand f
without the weight, the weight and wvar as quad takes them), its reference in ref/weights.json
(tools/integrate_ref.py weights).

  qawo   e^(cx) cos(wx), e^(cx) sin(wx) on finite ranges, the frequencies chosen so that the
         first rule is the 15-point Kronrod one (|w (b - a)/2| <= 2), the Clenshaw-Curtis rule
         with the moments of the tridiagonal system (<= 24) or of the recurrence (> 24); and
         log(x) cos(wx), log(x) sin(wx) on [0, 1] (Si, Ci), which bisect toward 0
  qawf   e^(-cx) cos(wx), e^(-cx) sin(wx) on [a, inf) and e^(cx) cos(wx), sin(wx) on (-inf, b],
         1/(1 + x^2) cos(wx) and x^-1/2 cos(wx), x^-1/2 sin(wx) on [0, inf), and the frequency
         0 (QAGI on [a, inf))
  qaws   1 and (x - a) + 2 with (x - a)^alpha (b - x)^beta and each of its logarithms on [0, 1]
         and [-1, 2] (Beta functions and their derivatives: digamma, trigamma)
  qawc   the principal values of 1/(x - c), x^3/(x - c), 1/((1 + x^2)(x - c)) and e^x/(x - c)
         with c inside, near an end and outside (Ei)

The constants are doubles of the code; a reference is the exact integral of the problem as the
doubles pose it.
"""

from __future__ import annotations

import math

from .problems import INF, Integral

# ---- closed forms (Arb), at the doubles of the problem


def _arb(v):
    from flint import arb

    return arb(v)


def _acb(v):
    from flint import acb

    return acb(v)


def _exp_trig(c, w, a, b, part):
    """int_a^b e^(cx) cos(wx) (part 'cos') or sin (part 'sin'): [e^((c + iw) x)/(c + iw)]_a^b,
    with -inf or inf ends where the exponential vanishes."""
    z = _acb(c) + _acb(w) * 1j

    def at(x):
        if math.isinf(x):
            return _acb(0)
        return (z * _arb(x)).exp() / z

    v = at(b) - at(a)
    return v.real if part == "cos" else v.imag


def _log_trig(w, part):
    """int_0^1 log(x) cos(wx) = -Si(w)/w; int_0^1 log(x) sin(wx) = -(gamma + log w - Ci(w))/w."""
    from flint import arb

    W = _arb(w)
    if part == "cos":
        return -W.si() / W
    return -(arb.const_euler() + W.log() - W.ci()) / W


def _beta_logs(p, q, la, lb):
    """int_0^1 s^p (1 - s)^q log(s)^la log(1 - s)^lb ds for la, lb in {0, 1}."""
    from flint import acb, arb

    P, Q = _arb(p) + 1, _arb(q) + 1
    N = P + Q
    beta = (P.lgamma() + Q.lgamma() - N.lgamma()).exp()
    dp, dq = P.digamma() - N.digamma(), Q.digamma() - N.digamma()
    if la and lb:
        return beta * (dp * dq - acb(N).polygamma(1).real)
    if la:
        return beta * dp
    if lb:
        return beta * dq
    return beta + arb(0)


def _qaws_exact(a, b, alfa, beta, weight, shifted):
    """int_a^b f(x) (x - a)^alfa (b - x)^beta [log(x - a)] [log(b - x)], f = 1 or (x - a) + 2:
    x = a + L s, L = b - a, so (x - a) = L s, (b - x) = L (1 - s) and the logs split."""
    L = _arb(b) - _arb(a)
    la = weight in ("alg-loga", "alg-log")
    lb = weight in ("alg-logb", "alg-log")
    logL = L.log()

    def part(p):  # int_0^1 s^p (1 - s)^beta times the logs, the logs of L split out
        terms = _beta_logs(p, beta, 0, 0)
        if la and lb:
            terms = (logL * logL * _beta_logs(p, beta, 0, 0) + logL * (_beta_logs(p, beta, 1, 0)
                     + _beta_logs(p, beta, 0, 1)) + _beta_logs(p, beta, 1, 1))  # fmt: skip
        elif la:
            terms = logL * _beta_logs(p, beta, 0, 0) + _beta_logs(p, beta, 1, 0)
        elif lb:
            terms = logL * _beta_logs(p, beta, 0, 0) + _beta_logs(p, beta, 0, 1)
        return terms

    scale = (L.log() * (_arb(alfa) + _arb(beta) + 1)).exp()
    v = 2 * part(alfa) + L * part(alfa + 1) if shifted else part(alfa)
    return scale * v


def _qawc_exact(a, b, c, kind):
    """The principal value of int_a^b f(x)/(x - c), c not a or b."""
    A, B, C = _arb(a), _arb(b), _arb(c)
    log_ratio = abs((B - C) / (A - C)).log()
    if kind == "one":
        return log_ratio
    if kind == "cube":  # x^3/(x - c) = x^2 + c x + c^2 + c^3/(x - c)
        return (B**3 - A**3) / 3 + C * (B**2 - A**2) / 2 + C**2 * (B - A) + C**3 * log_ratio
    if kind == "rational":  # 1/((1 + x^2)(x - c)) = k/(x - c) - k (x + c)/(1 + x^2), k = 1/(1 + c^2)
        k = 1 / (1 + C**2)
        return k * log_ratio - k / 2 * ((1 + B**2) / (1 + A**2)).log() - k * C * (B.atan() - A.atan())
    # e^x/(x - c): e^c (Ei(b - c) - Ei(a - c))
    return C.exp() * ((B - C).ei() - (A - C).ei())


# ---- the problems


def _qawo():
    out = []
    for c, w, a, b in ((1.0, 0.5, 0.0, 1.0), (-2.0, 10.0, 0.0, 1.0), (-1.0, 40.0, 0.0, 1.0), (0.3, 45.0, -1.0, 2.0),
                       (1.0, 1000.0, 0.0, 1.0), (0.5, -7.0, 0.0, 3.0), (-0.25, 3.0, -3.3, 26.7)):  # fmt: skip
        for part in ("cos", "sin"):
            out.append(
                Integral(
                    id=f"weights/qawo/exp-{part}/{c}/{w}/{a}/{b}",
                    expr=lambda x, M, c=c: M.exp(M.const(c) * x),
                    a=a,
                    b=b,
                    weight=part,
                    wvar=w,
                    exact=lambda c=c, w=w, a=a, b=b, part=part: _exp_trig(c, w, a, b, part),
                    numeric=True,
                    source=f"e^({c} x) {part}({w} x) on [{a}, {b}]",
                    tanhsinh=False,
                )
            )
    for w in (10.0, 300.0):
        for part in ("cos", "sin"):
            out.append(
                Integral(
                    id=f"weights/qawo/log-{part}/{w}",
                    expr=lambda x, M: M.log(x) if x != 0 else M.const(0.0),
                    a=0.0,
                    b=1.0,
                    weight=part,
                    wvar=w,
                    exact=lambda w=w, part=part: _log_trig(w, part),
                    source=f"log(x) {part}({w} x) on [0, 1]",
                    tanhsinh=False,
                )
            )
    return out


def _rsqrt(x, M):
    return 1 / M.sqrt(x) if x != 0 else M.const(0.0)


def _qawf():
    out = []
    for c, w, a in ((1.0, 1.0, 0.0), (0.5, 3.0, 1.0), (2.0, 10.0, -1.0), (0.1, 0.5, 0.0), (1.0, 0.0, 1.0)):
        for part in ("cos", "sin"):
            if w == 0.0 and part == "sin":
                continue
            out.append(
                Integral(
                    id=f"weights/qawf/exp-{part}/{c}/{w}/{a}",
                    expr=lambda x, M, c=c: M.exp(-M.const(c) * x),
                    a=a,
                    b=INF,
                    weight=part,
                    wvar=w,
                    exact=lambda c=c, w=w, a=a, part=part: _exp_trig(-c, w, a, INF, part),
                    source=f"e^(-{c} x) {part}({w} x) on [{a}, inf)",
                    tanhsinh=False,
                )
            )
    for c, w, b in ((1.0, 2.0, 0.0), (0.5, 1.5, 2.0)):
        for part in ("cos", "sin"):
            out.append(
                Integral(
                    id=f"weights/qawf/left-exp-{part}/{c}/{w}/{b}",
                    expr=lambda x, M, c=c: M.exp(M.const(c) * x),
                    a=-INF,
                    b=b,
                    weight=part,
                    wvar=w,
                    exact=lambda c=c, w=w, b=b, part=part: _exp_trig(c, w, -INF, b, part),
                    source=f"e^({c} x) {part}({w} x) on (-inf, {b}]",
                    tanhsinh=False,
                )
            )
    for w in (1.0, 5.0):
        out.append(
            Integral(
                id=f"weights/qawf/lorentz-cos/{w}",
                expr=lambda x, M: 1 / (1 + x * x),
                a=0.0,
                b=INF,
                weight="cos",
                wvar=w,
                exact=lambda w=w: _lorentz(w),
                source=f"1/(1 + x^2) cos({w} x) on [0, inf) = pi e^-w/2",
                tanhsinh=False,
            )
        )
    for part in ("cos", "sin"):
        out.append(
            Integral(
                id=f"weights/qawf/rsqrt-{part}/3.0",
                expr=_rsqrt,
                a=0.0,
                b=INF,
                weight=part,
                wvar=3.0,
                exact=lambda: (_pi() / (2 * _arb(3.0))).sqrt(),
                source=f"x^-1/2 {part}(3 x) on [0, inf) = sqrt(pi/6)",
                tanhsinh=False,
            )
        )
    return out


def _pi():
    from flint import arb

    return arb.pi()


def _lorentz(w):
    return _pi() / 2 * (-_arb(w)).exp()


def _qaws():
    out = []
    cases = ((0.0, 1.0, -0.5, -0.5), (0.0, 1.0, 0.3, -0.7), (-1.0, 2.0, 2.0, 0.0), (-1.0, 2.0, -0.9, 1.5),
             (0.0, 1.0, 0.0, 0.0))  # fmt: skip
    for a, b, al, be in cases:
        for weight in ("alg", "alg-loga", "alg-logb", "alg-log"):
            for shifted in (False, True):
                name = "shifted" if shifted else "one"
                out.append(
                    Integral(
                        id=f"weights/qaws/{weight}/{name}/{a}/{b}/{al}/{be}",
                        expr=(lambda x, M, a=a: x - M.const(a) + 2) if shifted else (lambda x, M: M.const(1.0) + 0 * x),
                        a=a,
                        b=b,
                        weight=weight,
                        wvar=(al, be),
                        exact=lambda a=a, b=b, al=al, be=be, weight=weight, shifted=shifted: _qaws_exact(
                            a, b, al, be, weight, shifted
                        ),
                        source=f"{'(x - a) + 2' if shifted else '1'} with {weight} ({al}, {be}) on [{a}, {b}]",
                        tanhsinh=False,
                    )
                )
    return out


def _qawc():
    out = []
    fs = {
        "one": lambda x, M: M.const(1.0) + 0 * x,
        "cube": lambda x, M: x * x * x,
        "rational": lambda x, M: 1 / (1 + x * x),
        "exp": lambda x, M: M.exp(x),
    }
    for a, b, c in ((-1.0, 5.0, 0.0), (0.0, 1.0, 0.5), (0.0, 1.0, 0.999), (0.0, 1.0, 1.5), (-2.0, 2.0, 0.3),
                    (0.0, 1.0, -0.02)):  # fmt: skip
        for kind, f in fs.items():
            out.append(
                Integral(
                    id=f"weights/qawc/{kind}/{a}/{b}/{c}",
                    expr=f,
                    a=a,
                    b=b,
                    weight="cauchy",
                    wvar=c,
                    exact=lambda a=a, b=b, c=c, kind=kind: _qawc_exact(a, b, c, kind),
                    source=f"the principal value of {kind}/(x - {c}) on [{a}, {b}]",
                    tanhsinh=False,
                )
            )
    return out


SWEEP = tuple(_qawo() + _qawf() + _qaws() + _qawc())
BY_ID = {p.id: p for p in SWEEP}
assert len(BY_ID) == len(SWEEP)


def routine(p):
    """The QUADPACK routine quad calls for p."""
    if p.weight in ("cos", "sin"):
        return "qawf" if math.isinf(p.a) or math.isinf(p.b) else "qawo"
    return "qawc" if p.weight == "cauchy" else "qaws"
