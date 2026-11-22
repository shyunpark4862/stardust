"""Reference values for sd.graph (docs/design/graph.md §4, §7.2) on the test graphs of
tests/graph/graphs.py: exact rationals (Python fractions.Fraction; a double is an exact rational) and
Arb (python-flint) where a square root or a linear solve is in the way.

    .venv/bin/python tools/graph_ref.py                     # every family, on 2 processes
    .venv/bin/python tools/graph_ref.py paths yen           # only these families
    .venv/bin/python tools/graph_ref.py --jobs 1 pagerank   # one process
    .venv/bin/python tools/graph_ref.py --list              # the families and their points

writes tests/graph/ref/<family>.npz (layout: tests/reffile.py), read by tests/graph/refdata.py. Every
point names its graph (input "graph", the id in tests/graph/graphs.py) and "digest"
(graphs.graph_digest of the graph), so a changed generator shows. A reference is a true value rounded
once (a double-double hi + lo); per-node values are vectors (ragged across graphs).

    paths        one point per (graph, source). Methods directed, unweighted, undirected,
                 undirected_unweighted: the exact shortest distance to every node (inf when
                 unreachable), with fields pred (the predecessor of docs/design/graph.md §3.2 computed
                 exactly: tight edges, fewest edges, smallest index; -9999 for the source and the
                 unreachable), edges (that path's edge count; -1 unreachable) and abs_sum (the sum of
                 |w| along it, rounded). Methods negcycle and negcycle_undirected: 1 when a cycle of
                 negative exact sum is reachable from the source (then the distance methods leave the
                 point out). Field exact_pred: 1 when every path sum of the graph is exact in doubles
                 (integer weights below 2^53, dyadic weights), where sd's predecessors must equal pred
    yen          one point per (graph, source, sink) of the small graphs: every loopless path, sorted
                 by (exact length, edge count, node sequence); method lengths (the first K = 20) with
                 fields edges, paths (the nodes, each path closed by -1) and total (how many there are)
    assignment   one point per cost matrix (linear_sum_assignment: every entry a pair, +inf none) or
                 biadjacency matrix (min_weight_full_bipartite_matching: the stored entries, a stored 0
                 a pair of cost 0): methods min and max, the optimal exact cost over every assignment of
                 the smaller side, with fields count (the optimal assignments) and col_ind (the
                 lexicographically smallest optimal one); method feasible (1 or 0)
    pagerank     one point per (graph, options): the solution of (I - alpha M^T) x = (1 - alpha) v,
                 M the out-weight-normalised transition matrix with the dangling rows replaced
                 (Arb's arb_mat.solve, settled entry by entry)
    betweenness  one point per (graph, options): Brandes's definition in exact rationals (path
                 counts as Python integers), networkx's normalisation and endpoints
    laplacian    one point per (graph, options): the dense L = D - A (normed: D^-1/2 L D^-1/2, Arb's
                 square roots), the diagonal ignored, in-degree unless use_out_degree, A + A^T first
                 when symmetrized; the vector is L in row-major order

Workers are forked, replaced every 8 tasks (maxtasksperchild), each task under a time limit (SIGALRM).
No mpmath (the project's rule). The console output also goes to build/logs/graph_ref/.
"""

from __future__ import annotations

import argparse
import collections
import heapq
import itertools
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

OUT = ROOT / "tests" / "graph" / "ref"
TOOL = "tools/graph_ref.py"
SECONDS = 600  # a task's time limit
NONE = -9999  # the predecessor marker (scipy's)
YEN_K = 20
YEN_MAX_PATHS = 200_000


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def G():
    from tests.graph import graphs

    return graphs


# ---------------------------------------------------------------- exact helpers


def fr(v) -> Fraction:
    if isinstance(v, (bool, np.bool_)):
        return Fraction(int(v))
    if isinstance(v, (int, np.integer)):
        return Fraction(int(v))
    return Fraction(float(v))


def dd_of(q) -> tuple[float, float]:
    """(hi, lo) of an exact rational (None: inf)."""
    if q is None:
        return math.inf, 0.0
    try:
        hi = float(q)  # Fraction.__float__ rounds once
    except OverflowError:
        return (math.inf if q > 0 else -math.inf), 0.0
    if math.isinf(hi):
        return hi, 0.0
    return hi, float(q - Fraction(hi))


def adjacency(edges, n, unweighted=False):
    """[[(v, w)]] per node in index order, self loops left out (docs/design/graph.md §2.1)."""
    adj = [[] for _ in range(n)]
    for i, j, w in edges:
        if i != j:
            adj[i].append((j, Fraction(1) if unweighted else fr(w)))
    for a in adj:
        a.sort()
    return adj


def shortest(adj, s):
    """Exact distances from s by a queue Bellman-Ford: (dist list with None for unreachable, or None
    when a cycle of negative exact sum is reachable)."""
    n = len(adj)
    d = [None] * n
    d[s] = Fraction(0)
    q = collections.deque([s])
    inq = [False] * n
    inq[s] = True
    count = [0] * n
    while q:
        u = q.popleft()
        inq[u] = False
        count[u] += 1
        if count[u] > n:
            return None
        for v, w in adj[u]:
            t = d[u] + w
            if d[v] is None or t < d[v]:
                d[v] = t
                if not inq[v]:
                    inq[v] = True
                    q.append(v)
    return d


def predecessors(adj, s, d):
    """§3.2's rule on exact distances: T the tight edges, h the BFS edge count on T, pred(v) the
    smallest u with (u, v) in T and h(u) = h(v) - 1. Returns (pred, h)."""
    n = len(adj)
    tight = [[] for _ in range(n)]  # out-lists of T
    for u in range(n):
        if d[u] is None:
            continue
        for v, w in adj[u]:
            if v != u and d[v] is not None and d[u] + w == d[v]:
                tight[u].append(v)
    h = [-1] * n
    h[s] = 0
    q = collections.deque([s])
    while q:
        u = q.popleft()
        for v in tight[u]:
            if h[v] < 0:
                h[v] = h[u] + 1
                q.append(v)
    pred = [NONE] * n
    for u in range(n):
        if h[u] < 0:
            continue
        for v in tight[u]:
            if v != s and h[v] == h[u] + 1 and (pred[v] == NONE or u < pred[v]):
                pred[v] = u
    return pred, h


def undirected_edges(g):
    return [(i, j, w) for (i, j), w in sorted(G().undirected(g).items())]


# ---------------------------------------------------------------- paths


def paths_graphs():
    M = G()
    skip = {"nonfinite", "bipartite", "cost", "flow"}
    out = []
    for gid in M.ids():
        g = M.get(gid)
        if g.square and g.n >= 1 and not (skip & g.tags):
            out.append(gid)
    out.append("dense/special6")
    return list(dict.fromkeys(out))


def paths_points():
    M = G()
    pts = []
    for gid in paths_graphs():
        g = M.get(gid)
        n = g.n
        if n <= 16:
            sources = list(range(n))
        else:
            r = M.rng("sources", gid)
            sources = sorted({0, 1, n // 2, n - 1, *map(int, r.choice(n, 4, replace=False))})
        for s in sources:
            pts.append({"graph": gid, "source": s})
    return pts


def _exact_graph(g) -> bool:
    if "over" in g.id or "float" in g.tags and "exact" not in g.tags:
        return False
    return bool({"int", "exact", "bool"} & g.tags) or g.id.startswith(("empty", "isolated", "negative"))


def _one_reading(adj, s):
    d = shortest(adj, s)
    if d is None:
        return None
    pred, h = predecessors(adj, s, d)
    n = len(adj)
    absum = []
    w_of = [dict(a) for a in adj]
    for v in range(n):
        total, x = Fraction(0), v
        while pred[x] != NONE:
            total += abs(w_of[pred[x]][x])
            x = pred[x]
        absum.append(float(total))
    hi_lo = [dd_of(x) for x in d]
    return {
        "hi": np.array([a for a, _ in hi_lo]),
        "lo": np.array([b for _, b in hi_lo]),
        "pred": np.array(pred, np.int64),
        "edges": np.array(h, np.int64),
        "abs_sum": np.array(absum),
    }


def paths_task(p):
    g = G().get(p["graph"])
    s, n = p["source"], g.n
    out = {}
    readings = (("", g.edges()), ("undirected", undirected_edges(g)))
    for tag, edges in readings:
        for unweighted in (False, True):
            name = (tag or "directed") if not unweighted else ("unweighted" if not tag else "undirected_unweighted")
            r = _one_reading(adjacency(edges, n, unweighted), s)
            if r is None:
                out["negcycle" + ("_" + tag if tag else "")] = (1.0, 0.0, {})
                continue
            out[name] = (r.pop("hi"), r.pop("lo"), r)
            if not unweighted:
                out["negcycle" + ("_" + tag if tag else "")] = (0.0, 0.0, {})
    for v in out.values():
        if v[2]:
            v[2]["exact_pred"] = int(_exact_graph(g))
    return out


# ---------------------------------------------------------------- yen


YEN_GRAPHS = (
    "diamond/4",
    "grid2d/3x3",
    "path/10",
    "path/both10",
    "cycle/12",
    "star/9",
    "complete/8",
    "selfloop/10",
    "oneway/12",
    "negative/zerocycle10",
    "negative/dag30",
    "zero/15",
    "twoway/12",
    "absorb/20",
)


def yen_points():
    M = G()
    pts = []
    for gid in YEN_GRAPHS:
        g = M.get(gid)
        n = g.n
        pairs = [(0, n - 1), (n - 1, 0), (1, n // 2), (0, 0)]
        if gid == "negative/dag30":
            r = M.rng("yen", gid)
            pairs = [tuple(map(int, r.choice(n, 2, replace=False))) for _ in range(3)]
        for s, t in dict.fromkeys(pairs):
            pts.append({"graph": gid, "source": s, "sink": t})
    return pts


def yen_task(p):
    g = G().get(p["graph"])
    adj = adjacency(g.edges(), g.n)
    s, t = p["source"], p["sink"]
    found = []
    if s == t:
        found.append((Fraction(0), 0, (s,)))
    else:
        stack = [(s, (s,), Fraction(0))]
        while stack:
            u, path, length = stack.pop()
            for v, w in adj[u]:
                if v in path:
                    continue
                if v == t:
                    found.append((length + w, len(path), path + (v,)))
                    if len(found) > YEN_MAX_PATHS:
                        raise _Timeout
                else:
                    stack.append((v, path + (v,), length + w))
    found.sort()
    first = found[:YEN_K]
    hl = [dd_of(x[0]) for x in first]
    flat = [v for _, _, path in first for v in (*path, -1)]
    return {
        "lengths": (
            np.array([a for a, _ in hl]),
            np.array([b for _, b in hl]),
            {
                "edges": np.array([e for _, e, _ in first], np.int64),
                "paths": np.array(flat, np.int64),
                "total": len(found),
            },
        ),
    }


# ---------------------------------------------------------------- assignment


def assignment_points():
    M = G()
    return [{"graph": gid} for gid in M.ids("cost") + M.ids("bipartite")]


def assignment_task(p):
    g = G().get(p["graph"])
    r, c = g.shape
    cost = {(i, j): fr(w) for i, j, w in g.edges() if not (isinstance(w, float) and math.isinf(w) and w > 0)}
    has_neg_inf = any(isinstance(w, float) and w == -math.inf for _, _, w in g.edges())
    transpose = r > c
    small, large = (c, r) if transpose else (r, c)
    best = {"min": None, "max": None}
    count = {"min": 0, "max": 0}
    arg = {"min": None, "max": None}
    for perm in itertools.permutations(range(large), small):
        total = Fraction(0)
        ok = True
        for a, b in enumerate(perm):
            key = (b, a) if transpose else (a, b)
            if key not in cost:
                ok = False
                break
            total += cost[key]
        if not ok:
            continue
        if transpose:  # col_ind in the order of the ascending rows that are matched
            pairs = sorted((b, a) for a, b in enumerate(perm))
            cols = tuple(a for _, a in pairs)
        else:
            cols = tuple(perm)
        for kind, better in (("min", lambda x, y: x < y), ("max", lambda x, y: x > y)):
            if best[kind] is None or better(total, best[kind]):
                best[kind], count[kind], arg[kind] = total, 1, cols
            elif total == best[kind]:
                count[kind] += 1
                arg[kind] = min(arg[kind], cols)
    out = {"feasible": (1.0 if best["min"] is not None and not has_neg_inf else 0.0, 0.0, {})}
    if best["min"] is not None and not has_neg_inf:
        for kind in ("min", "max"):
            hi, lo = dd_of(best[kind])
            out[kind] = (hi, lo, {"count": count[kind], "col_ind": np.array(arg[kind], np.int64)})
    return out


# ---------------------------------------------------------------- pagerank


def pagerank_points():
    default = {"alpha": 0.85, "personalization": None, "dangling": None, "directed": True, "unweighted": False}
    graphs = (
        "dangling/50",
        "er/float100",
        "heavytail/120",
        "er/sym60",
        "grid2d/6x6",
        "star/9",
        "path/10",
        "components/40",
        "bool/30",
        "isolated/5",
        "selfloop/10",
    )
    pts = [{"graph": gid, **default} for gid in graphs]
    for gid in ("dangling/50", "er/float100"):
        n = G().get(gid).n
        r = G().rng("pagerank", gid)
        pers = r.uniform(0, 1, n)
        pers[r.choice(n, n // 5, replace=False)] = 0.0
        dang = r.uniform(0, 1, n)
        for change in (
            {"alpha": 0.5},
            {"alpha": 0.99},
            {"personalization": pers},
            {"dangling": dang},
            {"personalization": pers, "dangling": dang},
            {"directed": False},
            {"unweighted": True},
        ):
            pts.append({"graph": gid, **default, **change})
    return pts


def pagerank_task(p):
    from flint import arb, arb_mat

    g = G().get(p["graph"])
    n = g.n
    edges = g.edges() if p["directed"] else undirected_edges(g)
    pers = np.ones(n) if p["personalization"] is None else np.asarray(p["personalization"])
    dang = pers if p["dangling"] is None else np.asarray(p["dangling"])
    alpha = fr(p["alpha"])
    out_w = collections.defaultdict(Fraction)
    rows = collections.defaultdict(list)
    for i, j, w in edges:  # self loops are edges here (networkx keeps them)
        wq = Fraction(1) if p["unweighted"] else fr(w)
        rows[i].append((j, wq))
        out_w[i] += wq
    v = [fr(x) / sum(map(fr, pers)) for x in pers]
    dd = [fr(x) / sum(map(fr, dang)) for x in dang]
    # A = I - alpha M^T, b = (1 - alpha) v, entries exact rationals turned into balls at each precision

    def solve():
        A = arb_mat(n, n)
        for i in range(n):
            A[i, i] = arb(1)
        for i in range(n):
            if out_w[i] == 0:  # dangling: row i of M is dd
                for j in range(n):
                    A[j, i] -= (
                        arb(alpha.numerator) * arb(dd[j].numerator) / (arb(alpha.denominator) * arb(dd[j].denominator))
                    )
            else:
                for j, wq in rows[i]:
                    q = alpha * wq / out_w[i]
                    A[j, i] -= arb(q.numerator) / arb(q.denominator)
        b = arb_mat(n, 1)
        for i in range(n):
            q = (1 - alpha) * v[i]
            b[i, 0] = arb(q.numerator) / arb(q.denominator)
        return A.solve(b)

    p_ = arbref.START_PREC
    while p_ <= arbref.MAX_PREC:
        with arbref.precision(p_):
            x = solve()
        vals = [x[i, 0] for i in range(n)]
        if all(arbref.settled(val) or (val.is_exact() and val == 0) for val in vals):
            hl = [to_dd(val) if not (val.is_exact() and val == 0) else (0.0, 0.0) for val in vals]
            return {"x": (np.array([a for a, _ in hl]), np.array([b for _, b in hl]), {})}
        p_ *= 2
    raise NotSettled(p["graph"])


# ---------------------------------------------------------------- betweenness


def betweenness_points():
    graphs = (
        "grid2d/5x5",
        "ladder/20",
        "er/sym60",
        "er/int60",
        "star/9",
        "path/10",
        "components/40",
        "diamond/4",
        "dag/30",
        "heavytail/120",
        "complete/8",
        "isolated/5",
        "empty/1",
    )
    pts = []
    for gid in graphs:
        for directed, unweighted in ((True, True), (True, False), (False, True), (False, False)):
            for normalized, endpoints in ((True, False), (False, False), (True, True)):
                if (normalized, endpoints) != (True, False) and gid not in ("grid2d/5x5", "er/int60", "diamond/4"):
                    continue
                pts.append(
                    {
                        "graph": gid,
                        "directed": directed,
                        "unweighted": unweighted,
                        "normalized": normalized,
                        "endpoints": endpoints,
                    }
                )
    return pts


def betweenness_task(p):
    g = G().get(p["graph"])
    n = g.n
    edges = g.edges() if p["directed"] else undirected_edges(g)
    adj = adjacency(edges, n, p["unweighted"])
    bc = [Fraction(0)] * n
    for s in range(n):
        # exact distances (Dijkstra: weights positive) and the shortest-path DAG
        d = [None] * n
        d[s] = Fraction(0)
        heap = [(Fraction(0), s)]
        done = [False] * n
        order = []
        while heap:
            du, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            order.append(u)
            for v, w in adj[u]:
                t = du + w
                if d[v] is None or t < d[v]:
                    d[v] = t
                    heapq.heappush(heap, (t, v))
        sigma = [0] * n
        sigma[s] = 1
        preds = [[] for _ in range(n)]
        for u in order:  # nondecreasing distance: sigma of u is final before its successors
            for v, w in adj[u]:
                if d[u] + w == d[v] and v != s:
                    sigma[v] += sigma[u]
                    preds[v].append(u)
        delta = [Fraction(0)] * n
        for w_ in reversed(order):
            for u in preds[w_]:
                delta[u] += Fraction(sigma[u], sigma[w_]) * (1 + delta[w_])
            if w_ != s:
                bc[w_] += delta[w_] + (1 if p["endpoints"] else 0)
        if p["endpoints"]:
            bc[s] += len(order) - 1
    # networkx's _rescale
    scale = None
    if p["normalized"]:
        if p["endpoints"]:
            scale = None if n < 2 else Fraction(1, n * (n - 1))
        else:
            scale = None if n <= 2 else Fraction(1, (n - 1) * (n - 2))
    elif not p["directed"]:
        scale = Fraction(1, 2)
    if scale is not None:
        bc = [x * scale for x in bc]
    hl = [dd_of(x) for x in bc]
    return {"bc": (np.array([a for a, _ in hl]), np.array([b for _, b in hl]), {})}


# ---------------------------------------------------------------- laplacian


def laplacian_points():
    graphs = (
        "er/symfloat50",
        "er/float100",
        "star/9",
        "dangling/50",
        "negative/dag30",
        "int64/30",
        "bool/30",
        "selfloop/10",
        "isolated/5",
    )
    opts = (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),
        (True, False, True),
    )
    pts = []
    for gid in graphs:
        for normed, out_deg, sym in opts:
            if normed and gid == "negative/dag30":
                continue
            pts.append({"graph": gid, "normed": normed, "use_out_degree": out_deg, "symmetrized": sym})
    return pts


def laplacian_task(p):
    from flint import arb

    g = G().get(p["graph"])
    n = g.n
    A = {}
    for i, j, w in g.edges():
        if i != j:
            A[(i, j)] = A.get((i, j), Fraction(0)) + fr(w)
    if p["symmetrized"]:
        S = dict(A)
        for (i, j), w in A.items():
            S[(j, i)] = S.get((j, i), Fraction(0)) + w
        A = S
    deg = [Fraction(0)] * n
    for (i, j), w in A.items():
        deg[i if p["use_out_degree"] else j] += w
    hi, lo = np.zeros(n * n), np.zeros(n * n)
    for i in range(n):
        for j in range(n):
            if i == j:
                q = (Fraction(1) if deg[i] != 0 else Fraction(0)) if p["normed"] else deg[i]
                hi[i * n + j], lo[i * n + j] = dd_of(q)
                continue
            a = A.get((i, j))
            if a is None:
                continue
            if not p["normed"]:
                hi[i * n + j], lo[i * n + j] = dd_of(-a)
                continue
            di, dj = deg[i] or Fraction(1), deg[j] or Fraction(1)

            def val(a=a, di=di, dj=dj):
                return -(arb(a.numerator) / arb(a.denominator)) / (
                    (arb(di.numerator) / arb(di.denominator)).sqrt() * (arb(dj.numerator) / arb(dj.denominator)).sqrt()
                )

            hi[i * n + j], lo[i * n + j] = to_dd(rigorous(val))
    return {"L": (hi, lo, {})}


# ---------------------------------------------------------------- driver

FAMILIES = {
    "paths": (paths_points, paths_task, ("source",)),
    "yen": (yen_points, yen_task, ("source", "sink")),
    "assignment": (assignment_points, assignment_task, ()),
    "pagerank": (pagerank_points, pagerank_task, ("alpha", "personalization", "dangling", "directed", "unweighted")),
    "betweenness": (betweenness_points, betweenness_task, ("directed", "unweighted", "normalized", "endpoints")),
    "laplacian": (laplacian_points, laplacian_task, ("normed", "use_out_degree", "symmetrized")),
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


def _input(p, name):
    v = p[name]
    if name in ("personalization", "dangling"):
        return np.zeros(0) if v is None else np.asarray(v, np.float64)
    return v


def write_family(fam, plan, results, dropped):
    M = G()
    names = FAMILIES[fam][2]
    inputs = {"graph": np.array([p["graph"] for p in plan])}
    inputs["digest"] = np.array([M.graph_digest(M.get(p["graph"])) for p in plan])
    for n in names:
        vals = [_input(p, n) for p in plan]
        inputs[n] = vals if n in ("personalization", "dangling") else np.array(vals)
    order = []
    for k in sorted(results):
        for m in results[k]:
            if m not in order:
                order.append(m)
    methods = {}
    for m in order:
        idx = [k for k in sorted(results) if m in results[k]]
        entry = {
            "over": "point",
            "index": np.array(idx, np.int64),
            "hi": [results[k][m][0] for k in idx],
            "lo": [results[k][m][1] for k in idx],
            "dropped": np.array(sorted(dropped), np.int64),
        }
        fields = list(results[idx[0]][m][2]) if idx else []
        for f in fields:
            entry[f] = [results[k][m][2][f] for k in idx]
        methods[m] = entry
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
    tasks.sort(key=lambda t: -G().get(t[2]["graph"]).n)  # the large graphs first
    results = {f: {} for f in names}
    dropped = {f: {} for f in names}
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
        for fam, k, res, secs, err in pool.imap_unordered(_point, tasks, chunksize=1):
            if res is None:
                dropped[fam][k] = err
            else:
                results[fam][k] = res
            if secs > 30:
                print(f"  {fam} {plans[fam][k]['graph']}: {secs:.0f} s", flush=True)
    for f in names:
        write_family(f, plans[f], results[f], dropped[f])
        print(f"{f}: {len(results[f])} of {len(plans[f])} points -> {(OUT / f'{f}.npz').relative_to(ROOT)}", flush=True)
        for k, err in sorted(dropped[f].items()):
            print(f"  dropped {plans[f][k]['graph']} #{k}: {err}", flush=True)
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    print(f"{len(tasks)} points, {(time.monotonic() - start):.0f} s, largest worker RSS {rss:.2f} GB", flush=True)


def main(argv=None):
    from bench import cli

    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="FAMILY", help=", ".join(FAMILIES))
    ap.add_argument("--list", action="store_true", help="list the families and their points")
    cli.add_jobs(ap, default=2)
    a = ap.parse_args(argv)
    bad = [f for f in a.names if f not in FAMILIES]
    if bad:
        ap.error(f"unknown family {bad[0]!r} (known: {', '.join(FAMILIES)})")
    names = a.names or list(FAMILIES)
    if a.list:
        for f in names:
            pts = FAMILIES[f][0]()
            print(f"{f} ({len(pts)}): " + ", ".join(dict.fromkeys(p["graph"] for p in pts)))
        return 0
    with arbref.logged("graph_ref"):
        build(names, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
