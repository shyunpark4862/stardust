"""Reference values for sd.spatial (docs/design/spatial.md §4, §7.2) on the test points of
tests/spatial/points.py: exact rationals and Arb (python-flint).

    .venv/bin/python tools/spatial_ref.py                     # every family, on 4 processes
    .venv/bin/python tools/spatial_ref.py distances           # only this family
    .venv/bin/python tools/spatial_ref.py --jobs 2 neighbors  # 2 processes
    .venv/bin/python tools/spatial_ref.py --list              # the families and their points

writes tests/spatial/ref/<family>.npz (layout: tests/reffile.py), read by tests/spatial/refdata.py.
Every point names its point set (input "set", the id in tests/spatial/points.py, or "edge/<id>",
"haversine/<id>" for the pair lists) and the inputs themselves; "digest" is points.digest of the
point's inputs, so a changed generator shows. The inputs are doubles taken as exact rationals:
every difference, sum and product of them is exact, and a reference is the true value of the
distance of those inputs, rounded once (a double-double hi + lo).

    distances  one point per pair (u, v): consecutive rows of the random, adversarial and lattice
               sets, and the finite edge pairs. Methods euclidean, sqeuclidean, cityblock,
               chebyshev, minkowski_p<p> (p = 0.5, 1.5, 3, 7.5), seuclidean (input V), mahalanobis
               (input VI, positive definite; n <= 16), cosine, correlation, canberra, braycurtis,
               and the weighted w_<metric> (input w, positive with some zeros: scipy's meaning,
               chebyshev the largest |z_i| over w_i > 0). A method is left out at a point where
               its value is NaN or not finite by definition (a zero vector's cosine, 0/0 of
               braycurtis, an overflowing value is kept: hi = inf)
    counting   hamming, jaccard and their weighted forms on the binary sets (bool, int64 with
               values past 2^53: inputs compared as they are), exact rationals; input "dtype" keeps
               each point's dtype (the file stores the ragged u and v in one int64 array)
    haversine  haversine (latitude, longitude in radians; sklearn's meaning: the central angle of
               the unit sphere) on random pairs, antipodes, poles, identical and close points
    neighbors  one point per query of a neighbour set: the distance from the query to every data
               point, in data order (methods p1, p2, pinf: cityblock, euclidean, chebyshev).
               On the lattice sets every computed distance is correctly rounded, so these are the
               exact answers of §4 neighbours 2; on the others they are the true distances §4
               neighbours 3 checks against

Settling: an exact rational is exact; an Arb value is settled to 2^-110 relative (a value that is
exactly 0 is recognised in rationals first). A point that does not settle within the precision
cap or its time limit is dropped and reported. Workers are forked, replaced every 8 tasks
(maxtasksperchild), each task under a time limit (SIGALRM). No mpmath (the project's rule). The
console output also goes to build/logs/spatial_ref/.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import signal
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import arbref  # noqa: E402
from arbref import NotSettled, rigorous, to_dd  # noqa: E402

OUT = ROOT / "tests" / "spatial" / "ref"
TOOL = "tools/spatial_ref.py"
SECONDS = 300  # a task's time limit
P_VALUES = (0.5, 1.5, 3.0, 7.5)
MAHALANOBIS_MAX_N = 16
PAIRS = 16  # consecutive pairs per set (4 for n >= 512)


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def P():
    from tests.spatial import points

    return points


# ---------------------------------------------------------------- exact helpers


def fr(v) -> Fraction:
    return Fraction(float(v))


def dd_of(q: Fraction):
    """(hi, lo) of an exact rational: hi rounded to nearest (ties to even), lo the rest rounded;
    (inf, 0) past the double range."""
    try:
        hi = float(q)  # int / int true division rounds correctly
    except OverflowError:
        return (math.inf if q > 0 else -math.inf), 0.0
    if math.isinf(hi):
        return hi, 0.0
    return hi, float(q - Fraction(hi))


def arb_of(q: Fraction):
    """An exact arb of a rational with a power-of-two denominator (a double, a sum or product of
    doubles); other rationals as the quotient ball."""
    from flint import arb

    num, den = q.numerator, q.denominator
    if den & (den - 1) == 0:
        return arb(num) * arb(2) ** (-(den.bit_length() - 1))
    return arb(num) / arb(den)


def settle(fn):
    return to_dd(rigorous(fn))


def sqrt_dd(q: Fraction):
    """(hi, lo) of sqrt(q), q >= 0 exact."""
    if q == 0:
        return 0.0, 0.0
    return settle(lambda: arb_of(q).sqrt())


# ---------------------------------------------------------------- distances


def _metric_values(u, v, w, V, VI):
    """{method: (hi, lo)} of every distance defined at (u, v) (weights w, variances V, VI or None)."""
    n = len(u)
    U, Vv = [fr(a) for a in u], [fr(b) for b in v]
    Z = [a - b for a, b in zip(U, Vv)]
    out = {}
    S = sum((z * z for z in Z), Fraction(0))
    out["sqeuclidean"] = dd_of(S)
    out["euclidean"] = sqrt_dd(S)
    out["cityblock"] = dd_of(sum((abs(z) for z in Z), Fraction(0)))
    out["chebyshev"] = dd_of(max((abs(z) for z in Z), default=Fraction(0)))
    absz = [abs(z) for z in Z]
    for p in P_VALUES:
        out[f"minkowski_p{p:g}"] = _minkowski(absz, None, p)
    if V is not None:
        out["seuclidean"] = sqrt_dd(sum((z * z / fr(s) for z, s in zip(Z, V)), Fraction(0)))
    if VI is not None:
        M = [[fr(a) for a in row] for row in VI]
        q = sum((Z[i] * M[i][j] * Z[j] for i in range(n) for j in range(n)), Fraction(0))
        if q >= 0:
            out["mahalanobis"] = sqrt_dd(q)
    c = _cosine(U, Vv, None)
    if c is not None:
        out["cosine"] = c
    c = _correlation(U, Vv, None)
    if c is not None:
        out["correlation"] = c
    out["canberra"] = dd_of(_canberra(U, Vv, None))
    b = _braycurtis(U, Vv, None)
    if b is not None:
        out["braycurtis"] = b
    if w is not None:
        W = [fr(a) for a in w]
        Sw = sum((a * z * z for a, z in zip(W, Z)), Fraction(0))
        out["w_sqeuclidean"] = dd_of(Sw)
        out["w_euclidean"] = sqrt_dd(Sw)
        out["w_cityblock"] = dd_of(sum((a * abs(z) for a, z in zip(W, Z)), Fraction(0)))
        out["w_chebyshev"] = dd_of(max((abs(z) for a, z in zip(W, Z) if a > 0), default=Fraction(0)))
        out["w_minkowski_p3"] = _minkowski(absz, W, 3.0)
        c = _cosine(U, Vv, W)
        if c is not None:
            out["w_cosine"] = c
        c = _correlation(U, Vv, W)
        if c is not None:
            out["w_correlation"] = c
        out["w_canberra"] = dd_of(_canberra(U, Vv, W))
        b = _braycurtis(U, Vv, W)
        if b is not None:
            out["w_braycurtis"] = b
    return out


def _minkowski(absz, W, p):
    """(sum w_i |z_i|^p)^(1/p) in Arb."""
    from flint import arb

    terms = [(z, a) for z, a in zip(absz, W if W is not None else [Fraction(1)] * len(absz)) if z != 0 and a != 0]
    if not terms:
        return 0.0, 0.0
    if len(terms) == 1:  # (w |z|^p)^(1/p) = w^(1/p) |z|
        z, a = terms[0]
        if a == 1:
            return dd_of(z)

    def f():
        pa = arb(p)
        s = arb(0)
        for z, a in terms:
            s += arb_of(a) * arb_of(z) ** pa
        return s ** (1 / pa)

    return settle(f)


def _cosine_of(xy: Fraction, xx: Fraction, yy: Fraction):
    """1 - xy / sqrt(xx yy), or None when a norm is 0 (NaN)."""
    if xx == 0 or yy == 0:
        return None
    if xy * xy == xx * yy:  # parallel: 0 or 2 exactly
        return (0.0, 0.0) if xy > 0 else (2.0, 0.0)
    return settle(lambda: 1 - arb_of(xy) / (arb_of(xx) * arb_of(yy)).sqrt())


def _cosine(U, Vv, W):
    W = W if W is not None else [Fraction(1)] * len(U)
    xy = sum((a * x * y for a, x, y in zip(W, U, Vv)), Fraction(0))
    xx = sum((a * x * x for a, x in zip(W, U)), Fraction(0))
    yy = sum((a * y * y for a, y in zip(W, Vv)), Fraction(0))
    return _cosine_of(xy, xx, yy)


def _correlation(U, Vv, W):
    if not U:
        return None
    W = W if W is not None else [Fraction(1)] * len(U)
    sw = sum(W, Fraction(0))
    if sw == 0:
        return None
    mu = sum((a * x for a, x in zip(W, U)), Fraction(0)) / sw
    mv = sum((a * y for a, y in zip(W, Vv)), Fraction(0)) / sw
    return _cosine([x - mu for x in U], [y - mv for y in Vv], W)


def _canberra(U, Vv, W):
    W = W if W is not None else [Fraction(1)] * len(U)
    s = Fraction(0)
    for a, x, y in zip(W, U, Vv):
        d = abs(x) + abs(y)
        if d != 0:
            s += a * abs(x - y) / d
    return s


def _braycurtis(U, Vv, W):
    W = W if W is not None else [Fraction(1)] * len(U)
    num = sum((a * abs(x - y) for a, x, y in zip(W, U, Vv)), Fraction(0))
    den = sum((a * abs(x + y) for a, x, y in zip(W, U, Vv)), Fraction(0))
    if den == 0:
        return None  # 0/0 is NaN, num/0 is inf: no reference needed
    return dd_of(num / den)


def distance_points():
    pts = []
    M = P()
    for sid in M.ids():
        s = M.get(sid)
        if "binary" in s.tags:
            continue
        u, v = M.pairs(s, PAIRS if s.n < 512 else 4)
        for k in range(len(u)):
            pts.append(_dpoint(sid, k, u[k], v[k]))
    for eid, u, v in M.edge_pairs():
        if np.all(np.isfinite(u)) and np.all(np.isfinite(v)):
            pts.append(_dpoint(f"edge/{eid}", 0, u, v))
    return pts


def _dpoint(sid, k, u, v):
    M = P()
    n = len(u)
    w = M.weights(n, "zeros" if k % 2 else "positive")
    V = M.variances(n)
    VI = M.inverse_covariances(n) if 1 <= n <= MAHALANOBIS_MAX_N else None
    return {"set": sid, "pair": k, "u": np.asarray(u, float), "v": np.asarray(v, float), "w": w, "V": V, "VI": VI}


def distance_task(p):
    return _metric_values(p["u"], p["v"], p["w"], p["V"], p["VI"])


# ---------------------------------------------------------------- counting


def counting_points():
    M = P()
    pts = []
    for sid in M.ids("binary"):
        s = M.get(sid)
        u, v = M.pairs(s, PAIRS)
        for k in range(len(u)):
            pts.append(
                {"set": sid, "pair": k, "u": u[k], "v": v[k], "w": M.weights(s.n, "zeros" if k % 2 else "positive")}
            )
    return pts


def counting_task(p):
    u, v = p["u"].tolist(), p["v"].tolist()
    W = [fr(a) for a in p["w"]]
    out = {}
    for prefix, ws in (("", [Fraction(1)] * len(u)), ("w_", W)):
        diff = sum((a for a, x, y in zip(ws, u, v) if x != y), Fraction(0))
        tot = sum(ws, Fraction(0))
        if tot != 0:
            out[prefix + "hamming"] = dd_of(diff / tot)
        nz = [(bool(x), bool(y)) for x, y in zip(u, v)]
        num = sum((a for a, (x, y) in zip(ws, nz) if x != y), Fraction(0))
        den = sum((a for a, (x, y) in zip(ws, nz) if x or y), Fraction(0))
        out[prefix + "jaccard"] = dd_of(num / den) if den != 0 else (0.0, 0.0)
    return out


# ---------------------------------------------------------------- haversine


def haversine_points():
    return [{"set": f"haversine/{hid}", "pair": 0, "u": u, "v": v} for hid, u, v in P().haversine_pairs()]


def haversine_task(p):
    from flint import arb

    (f1, l1), (f2, l2) = p["u"].tolist(), p["v"].tolist()
    if f1 == f2 and l1 == l2:
        return {"haversine": (0.0, 0.0)}

    def f():
        a1, b1, a2, b2 = arb(f1), arb(l1), arb(f2), arb(l2)
        h = ((a2 - a1) / 2).sin() ** 2 + a1.cos() * a2.cos() * ((b2 - b1) / 2).sin() ** 2
        return 2 * h.sqrt().asin()

    return {"haversine": settle(f)}


# ---------------------------------------------------------------- neighbours


def neighbor_points():
    M = P()
    pts = []
    for sid in M.ids("neighbors"):
        s = M.get(sid)
        for k in range(s.queries.shape[0]):
            pts.append({"set": sid, "pair": k, "query": s.queries[k]})
    return pts


def neighbor_task(p):
    M = P()
    s = M.get(p["set"])
    q = [fr(a) for a in p["query"]]
    out = {"p1": [], "p2": [], "pinf": []}
    for row in s.x:
        Z = [a - fr(b) for a, b in zip(q, row)]
        out["p1"].append(dd_of(sum((abs(z) for z in Z), Fraction(0))))
        out["p2"].append(sqrt_dd(sum((z * z for z in Z), Fraction(0))))
        out["pinf"].append(dd_of(max((abs(z) for z in Z), default=Fraction(0))))
    return {k: (np.array([h for h, _ in v]), np.array([lo for _, lo in v])) for k, v in out.items()}


# ---------------------------------------------------------------- driver

FAMILIES = {
    "distances": (distance_points, distance_task, ("u", "v", "w", "V", "VI")),
    "counting": (counting_points, counting_task, ("u", "v", "w")),
    "haversine": (haversine_points, haversine_task, ("u", "v")),
    "neighbors": (neighbor_points, neighbor_task, ("query",)),
}


def _point(task):
    fam, k, p = task
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SECONDS)
    t0 = time.monotonic()
    try:
        res = FAMILIES[fam][1](p)
        err = None
    except (_Timeout, NotSettled) as e:
        res, err = None, type(e).__name__
    finally:
        signal.alarm(0)
    return fam, k, res, time.monotonic() - t0, err


def _digest(p, names):
    return P().digest(*[p[n] if p.get(n) is not None else None for n in names])


def write_family(fam, plan, results, dropped):
    names = FAMILIES[fam][2]
    inputs = {"set": np.array([p["set"] for p in plan]), "pair": np.array([p["pair"] for p in plan], np.int64)}
    inputs["digest"] = np.array([_digest(p, names) for p in plan])
    for n in names:
        if n == "VI":
            inputs[n] = [p[n] if p[n] is not None else np.zeros((0, 0)) for p in plan]
        else:
            inputs[n] = [np.asarray(p[n]) for p in plan]
    if fam == "counting":  # ragged inputs share one dtype in the file: the points' own dtypes
        inputs["dtype"] = np.array([str(np.asarray(p["u"]).dtype) for p in plan])
    methods = {}
    order = []
    for k in sorted(results):
        for m in results[k]:
            if m not in order:
                order.append(m)
    for m in order:
        idx = [k for k in sorted(results) if m in results[k]]
        his = [results[k][m][0] for k in idx]
        los = [results[k][m][1] for k in idx]
        methods[m] = {"over": "point", "index": np.array(idx, np.int64), "hi": his, "lo": los,
                      "dropped": np.array(sorted(dropped), np.int64)}  # fmt: skip
    from tests import reffile

    reffile.write(
        OUT / f"{fam}.npz",
        **arbref.provenance(extra="exact rationals"),
        tool=TOOL,
        inputs=inputs,
        methods=methods,
        n_requested=len(plan),
    )


def build(names, jobs):
    import resource

    start = time.monotonic()
    plans = {f: FAMILIES[f][0]() for f in names}
    tasks = [(f, k, p) for f in names for k, p in enumerate(plans[f])]
    results = {f: {} for f in names}
    dropped = {f: {} for f in names}
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
        for fam, k, res, secs, err in pool.imap_unordered(_point, tasks, chunksize=4):
            if res is None:
                dropped[fam][k] = err
            else:
                results[fam][k] = res
            if secs > 30:
                print(f"  {fam} {plans[fam][k]['set']}: {secs:.0f} s", flush=True)
    for f in names:
        write_family(f, plans[f], results[f], dropped[f])
        print(f"{f}: {len(results[f])} of {len(plans[f])} points -> {(OUT / f'{f}.npz').relative_to(ROOT)}", flush=True)
        for k, err in sorted(dropped[f].items()):
            print(f"  dropped {plans[f][k]['set']} #{plans[f][k]['pair']}: {err}", flush=True)
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    print(f"{len(tasks)} points, {(time.monotonic() - start):.0f} s, largest worker RSS {rss:.2f} GB", flush=True)


def main(argv=None):
    from bench import cli

    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="FAMILY", help=", ".join(FAMILIES))
    ap.add_argument("--list", action="store_true", help="list the families and their points")
    cli.add_jobs(ap, default=4)
    a = ap.parse_args(argv)
    bad = [f for f in a.names if f not in FAMILIES]
    if bad:
        ap.error(f"unknown family {bad[0]!r} (known: {', '.join(FAMILIES)})")
    names = a.names or list(FAMILIES)
    if a.list:
        for f in names:
            pts = FAMILIES[f][0]()
            print(f"{f} ({len(pts)}): " + ", ".join(dict.fromkeys(p["set"] for p in pts)))
        return 0
    with arbref.logged("spatial_ref"):
        build(names, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
