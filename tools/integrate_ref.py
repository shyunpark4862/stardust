"""The Arb references of sd.integrate (docs/design/integrate.md §7.2), in tests/integrate/ref/:

    .venv/bin/python tools/integrate_ref.py                    # all seven
    .venv/bin/python tools/integrate_ref.py integrals series   # only these

  integrals   integrals.json: per one-dimensional integral of tests/integrate/problems.py
              (INTEGRALS: the QUADPACK book's families, the routines' examples, scipy's tests,
              the hard integrals), the exact integral of the problem as the double code poses it
  series      series.json: per series of problems.SERIES, its exact sum
  multiple    multiple.json: per multiple integral of problems.MULTI, its exact value
  vector      vector.json: per vector-valued integrand of problems.VECTOR, its exact values
  samples     samples.json: per sample set of problems.samples(), each sample rule's formula
              (scipy's, docs/design/integrate.md §3.7) computed exactly in rationals from the
              doubles: the value of trapezoid, simpson and romb, the running values of
              cumulative_trapezoid and cumulative_simpson (C3)
  gauss       gauss.json: the Gauss-Legendre nodes and weights of fixed_quad past the table
              (n = 101 ... 1000, decision C42): Arb's, each settled and rounded to the nearest
              double (tools/gen_gauss_legendre.py, which writes the table of n <= 100)
  weights     weights.json: per weighted integral of tests/integrate/weights.py (SWEEP: quad's
              QAWO, QAWF, QAWS, QAWC families, C5), its exact integral like integrals'

A reference is an Arb ball settled to 2^-110 of its value (tools/arbref.py) and is stored as the
double-double [hi, lo]. It comes from up to three methods, and every method a problem allows is
computed and must agree with the others (the balls overlap), or the tool stops:

  closed form   the problem's exact() in Arb functions (atan, J0, Ci, Si, zeta, Gamma, erf, ...),
                at the doubles of the problem's constants and limits
  acb.integral  Arb's rigorous integration of f(x) w(x) on [a, b], for an integrand analytic on a
                neighbourhood of a finite [a, b] (problem.numeric; branch cuts by the `analytic`
                flag). acb.integral gives a non-finite ball where the integrand is unbounded at an
                end (x^-1/2, x^-0.9, log(x)/sqrt(x) on [0, 1]), hence the third method:
  substitution  acb.integral on the problem's pieces: a substitution that removes the endpoint
                singularity (x = t^2, x = e^-u) and, for a range made infinite by it, a cut with
                the rigorous bound of the rest added to the ball's radius (problem.tail)
The stored method is the first of these the problem has, and "checks" lists the others. The JSON
files carry "_provenance" (tests/integrate/refdata.py leaves it out) and, per entry, a digest of
the problem's definition (limits, break points, weight, source), so a changed problem shows as a
stale reference in tests/integrate/test_problems.py. Up to three processes (maxtasksperchild=8)
share the problems. An unknown name is a usage error (exit status 2); the console output also goes
to build/logs/integrate_ref/. No mpmath.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import arbref  # noqa: E402
from arbref import to_dd  # noqa: E402

OUT = ROOT / "tests" / "integrate" / "ref"
TOOL = "tools/integrate_ref.py"
PROCESSES = 3


def digest(*parts) -> str:
    """The sha256 of a problem's definition: what a reference depends on besides its code."""
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def integral_digest(p) -> str:
    return digest(p.id, p.a, p.b, p.points, p.weight, p.wvar, p.source)


def _acb_integral(g, lo, hi):
    """acb.integral of g(t, Acb) on [lo, hi] as an arb (its real part; the imaginary part is 0)."""
    from flint import acb, arb

    from tests.integrate.problems import Acb

    v = acb.integral(lambda t, analytic: g(t, Acb(analytic)), arb(lo), arb(hi))
    if not v.imag.contains(0):
        raise ArithmeticError("the integral of a real integrand has an imaginary part")
    return v.real


def _settle(fn):
    """fn() settled to 2^-110 at doubling precision (arbref.rigorous), as an arb."""
    v = arbref.rigorous(fn)
    if isinstance(v, float):
        from flint import arb

        v = arb(v)
    return v


def methods_of(p):
    """[(name, fn)] of the integral p, in the order the stored reference is chosen."""
    out = []
    if p.exact is not None:
        out.append(("closed form", p.exact))
    if p.numeric:
        out.append(("acb.integral", lambda: _acb_integral(p.full, p.a, p.b)))
    if p.pieces:

        def pieces(p=p):
            from flint import arb

            s = arb(0)
            for g, lo, hi in p.pieces:
                s += _acb_integral(g, lo, hi)
            return arbref.add_error(s, arb(p.tail)) if p.tail else s

        out.append(("substitution", pieces))
    return out


def _agree(pid, balls):
    names = list(balls)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not (balls[names[i]] - balls[names[j]]).contains(0):
                raise ArithmeticError(
                    f"{pid}: {names[i]} and {names[j]} disagree: {balls[names[i]]} vs {balls[names[j]]}"
                )


def _integral_task(pid):
    from tests.integrate import problems as P
    from tests.integrate import weights as W

    p = W.BY_ID[pid] if pid.startswith("weights/") else P.get(pid)
    ms = methods_of(p)
    if not ms:
        raise ValueError(f"{pid}: no reference method")
    t0 = time.perf_counter()
    try:
        balls = {name: _settle(fn) for name, fn in ms}
    except Exception as e:  # a plain error crosses the process boundary (the ball's function does not)
        raise RuntimeError(f"{pid}: {type(e).__name__}: {e}") from None
    _agree(pid, balls)
    first = ms[0][0]
    hi, lo = to_dd(balls[first])
    return pid, {
        "value": [hi, lo],
        "method": first,
        "checks": [name for name, _ in ms[1:]],
        "digest": integral_digest(p),
        "seconds": round(time.perf_counter() - t0, 3),
    }


def _series_task(sid):
    from tests.integrate import problems as P

    s = P.SERIES_BY_ID[sid]
    hi, lo = to_dd(_settle(s.exact))
    return sid, {"value": [hi, lo], "method": "closed form", "digest": digest(s.id, s.a, s.b, s.step, s.source)}


def _write(name, entries, n_requested, extra=None):
    prov = {**arbref.provenance(extra=extra), "tool": TOOL, "n_requested": n_requested, "dropped": []}
    lines = [f'"_provenance": {json.dumps(prov)}']
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in entries]
    (OUT / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"{len(entries)} references -> {(OUT / name).relative_to(ROOT)}")


def _pool():
    return mp.get_context("spawn").Pool(PROCESSES, maxtasksperchild=8)


def integrals():
    from tests.integrate import problems as P

    ids = [p.id for p in P.INTEGRALS]
    with _pool() as pool:
        rows = []
        for pid, entry in pool.imap(_integral_task, ids):
            rows.append((pid, entry))
            checked = f" (checked: {', '.join(entry['checks'])})" if entry["checks"] else ""
            print(f"  {pid}: {entry['method']}{checked}, {entry['seconds']} s")
    for _, e in rows:
        del e["seconds"]
    _write("integrals.json", rows, len(ids))


def weights():
    from tests.integrate import weights as W

    ids = [p.id for p in W.SWEEP]
    with _pool() as pool:
        rows = list(pool.imap(_integral_task, ids))
    for _, e in rows:
        del e["seconds"]
    _write("weights.json", rows, len(ids))


def series():
    from tests.integrate import problems as P

    rows = [_series_task(s.id) for s in P.SERIES]
    _write("series.json", rows, len(rows))


def multiple():
    from tests.integrate import problems as P

    rows = []
    for m in P.MULTI:
        hi, lo = to_dd(_settle(m.exact))
        rows.append(
            (m.id, {"value": [hi, lo], "method": "closed form", "digest": digest(m.id, m.kind, m.a, m.b, m.source)})
        )
    _write("multiple.json", rows, len(rows))


def vector():
    from tests.integrate import problems as P

    rows = []
    for v in P.VECTOR:
        n = len(v.exact())
        vals = [list(to_dd(_settle(lambda v=v, i=i: v.exact()[i]))) for i in range(n)]
        rows.append((v.id, {"value": vals, "method": "closed form", "digest": digest(v.id, v.a, v.b, v.source)}))
    _write("vector.json", rows, len(rows))


# ---- the sample rules, exactly (C3)


def _q(v):
    from fractions import Fraction

    return Fraction(v)


def _dd(q):
    """(hi, lo) of an exact rational: hi the nearest double, lo the nearest double to the rest."""
    from fractions import Fraction

    hi = float(q)
    return [hi, float(q - Fraction(hi))]


def _h(x, dx, i):
    return x[i + 1] - x[i] if x is not None else dx


def exact_trapezoid_terms(y, x, dx):
    return [_h(x, dx, i) * (y[i + 1] + y[i]) / 2 for i in range(len(y) - 1)]


def _simpson_basic(y, x, dx, stop):
    from fractions import Fraction

    total = Fraction(0)
    pairs = []
    for k in range(0, max(stop, 0), 2):
        if x is None:
            pairs.append(y[k] + 4 * y[k + 1] + y[k + 2])
            continue
        h0, h1 = x[k + 1] - x[k], x[k + 2] - x[k + 1]
        hsum, hprod = h0 + h1, h0 * h1
        r = h0 / h1 if h1 != 0 else Fraction(0)
        c0 = 2 - (1 / r if r != 0 else 0)
        c1 = hsum * (hsum / hprod if hprod != 0 else 0)
        total += hsum / 6 * (y[k] * c0 + y[k + 1] * c1 + y[k + 2] * (2 - r))
    if x is None:
        return dx / 3 * sum(pairs, Fraction(0))
    return total


def exact_simpson(y, x, dx):
    from fractions import Fraction

    n = len(y)
    if n <= 1:
        return Fraction(0)
    if n % 2 == 1:
        return _simpson_basic(y, x, dx, n - 2)
    if n == 2:
        return Fraction(1, 2) * _h(x, dx, 0) * (y[1] + y[0])
    res = _simpson_basic(y, x, dx, n - 3)
    h0, h1 = _h(x, dx, n - 3), _h(x, dx, n - 2)

    def ratio(num, den):
        return num / den if den != 0 else Fraction(0)

    alpha = ratio(2 * h1 * h1 + 3 * h0 * h1, 6 * (h1 + h0))
    beta = ratio(h1 * h1 + 3 * h0 * h1, 6 * h0)
    eta = ratio(h1**3, 6 * h0 * (h0 + h1))
    return res + alpha * y[n - 1] + beta * y[n - 2] - eta * y[n - 3]


def exact_romb(y, dx):
    n = len(y) - 1
    k = n.bit_length() - 1
    h = n * dx
    prev = [(y[0] + y[n]) / 2 * h]
    start = step = n
    for i in range(1, k + 1):
        start >>= 1
        s = sum(y[start:n:step])
        step >>= 1
        cur = [(prev[0] + h * s) / 2]
        for j in range(1, i + 1):
            cur.append(cur[j - 1] + (cur[j - 1] - prev[j - 1]) / ((1 << (2 * j)) - 1))
        h /= 2
        prev = cur
    return prev[k]


def exact_simpson_pieces(y, x, dx):
    n = len(y)
    if n < 3:
        return exact_trapezoid_terms(y, x, dx)
    out = []
    for k in range(n - 1):
        left = k % 2 == 0 and k != n - 2
        if left:
            f1, f2, f3, x21, x32 = y[k], y[k + 1], y[k + 2], _h(x, dx, k), _h(x, dx, k + 1)
        else:
            f1, f2, f3, x21, x32 = y[k + 1], y[k], y[k - 1], _h(x, dx, k), _h(x, dx, k - 1)
        if x is None:
            out.append(x21 / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4))
        else:
            x31 = x21 + x32
            a, b = x21 / x31, x21 / x32
            c = a * b
            out.append(x21 / 6 * ((3 - a) * f1 + (3 + c + a) * f2 - c * f3))
    return out


def _running(terms):
    out, s = [], 0
    for t in terms:
        s += t
        out.append(_dd(s))
    return out


def exact_rules(y, x, dx):
    """{rule: exact value} of one lane (y, x: lists of Fractions or x None; dx a Fraction)."""
    from fractions import Fraction

    r = {
        "trapezoid": _dd(sum(exact_trapezoid_terms(y, x, dx), Fraction(0))),
        "simpson": _dd(exact_simpson(y, x, dx)),
        "cumulative_trapezoid": _running(exact_trapezoid_terms(y, x, dx)),
        "cumulative_simpson": _running(exact_simpson_pieces(y, x, dx)),
    }
    n = len(y) - 1
    if x is None and n >= 1 and n & (n - 1) == 0:
        r["romb"] = _dd(exact_romb(y, dx))
    return r


def samples_digest(sid, y, x, dx) -> str:
    return digest(sid, y.tobytes(), None if x is None else x.tobytes(), dx)


def samples():
    from tests.integrate import problems as P

    rows = []
    for sid, (y, x, dx) in P.samples().items():
        lanes = y.reshape(-1, y.shape[-1])
        xs = None if x is None else [_q(v) for v in x.tolist()]
        per = [exact_rules([_q(v) for v in lane.tolist()], xs, _q(dx)) for lane in lanes]
        entry = {k: [p[k] for p in per] if y.ndim > 1 else per[0][k] for k in per[0]}
        entry.update(method="exact rationals", digest=samples_digest(sid, y, x, dx))
        rows.append((sid, entry))
    _write("samples.json", rows, len(rows), extra="exact rationals (fractions.Fraction) for the sample rules")


# ---- the Gauss-Legendre nodes and weights past the table (C42)

GAUSS_N = (101, 128, 150, 200, 256, 500, 1000)


def gauss():
    import gen_gauss_legendre

    rows = []
    for n in GAUSS_N:
        half = gen_gauss_legendre.half(n)
        rows.append(
            (
                str(n),
                {"x": [v.hex() for v, _ in half], "w": [w.hex() for _, w in half], "method": "arb.legendre_p_root"},
            )
        )
        print(f"  n = {n}: {len(half)} nodes")
    _write("gauss.json", rows, len(rows))


PARTS = {
    "integrals": integrals,
    "series": series,
    "multiple": multiple,
    "vector": vector,
    "samples": samples,
    "gauss": gauss,
    "weights": weights,
}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="tools/integrate_ref.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="NAME", help=", ".join(PARTS))
    a = ap.parse_args(argv)
    bad = [n for n in a.names if n not in PARTS]
    if bad:
        ap.error(f"unknown reference {bad[0]!r}")
    OUT.mkdir(parents=True, exist_ok=True)
    with arbref.logged("integrate_ref"):
        for n in a.names or PARTS:
            PARTS[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
