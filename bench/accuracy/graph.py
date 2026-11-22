"""sd.graph's accuracy sweeps: one per reference family of tests/graph/ref (tools/graph_ref.py) once
the functions it measures are public (docs/design/graph.md §7.5).

GR2 brings

  paths   the general-weight points of paths.npz (the graphs whose path sums are not exact in doubles:
          random weights over [0.01, 1.01) and 1e-8 .. 1e8, absorbed weights, sums past 2^53), one region
          per graph and reading (directed, undirected) and per rounding: the definition's (sd's 'D',
          'BF', 'J' and 'auto' give the same bits; 'J' where a weight is negative) and the blocked
          Floyd-Warshall's ("<graph> <reading> FW"), one point per (source, reachable node). The
          contract (tests/graph/refdata.py, docs/design/graph.md §4): |d - d*| <= gamma_{H-1} S, H the
          larger edge count and S the larger sum of |w| of the returned path (the predecessor chain) and
          of the reference's shortest path, in units of that bound. The comparator scipy 1.18.1 runs
          the same methods ('D' or 'J', and 'FW') on the same graphs, judged with its own predecessors.

The graphs with exact sums (integer, dyadic and bool weights; unweighted) are exact and are the tests'
(tests/graph/test_paths_reference.py).

GR3 brings

  assignment   float costs: the float points of assignment.npz (cost/float7x7 by linear_sum_assignment,
               bipartite/float8x8 by min_weight_full_bipartite_matching; min and max) and random cost
               matrices over 10^-4 .. 10^4 of both signs (square and rectangular) and near ties (multiples
               of 1000 plus parts below 10^-10), one region per shape and min / max, the exact optimum of tests/graph/definitions.py assignment_exact, an exact-
               rational solver). One point per problem: the exact cost of the returned assignment above the
               optimum, in units of the contract gamma_{2r} (sum |c| over the returned and an optimal
               assignment; docs/design/graph.md §4), and its ulp and relative distance from the optimum.
               The comparator scipy 1.18.1 runs linear_sum_assignment (min_weight_full_bipartite_matching
               for the biadjacency point) on the same problems. Integer costs are exact (the tests).

GR4 brings

  laplacian    the float points of laplacian.npz (er/symfloat50, er/float100, star/9, dangling/50,
               negative/dag30 (unnormalized only), selfloop/10; every option of normed, use_out_degree and
               symmetrized; the integer, bool and empty graphs are exact: the tests), one region per graph
               and normed / unnormalized, one point per entry the rounding can touch (the degrees and,
               normed or symmetrized, the off-diagonal entries), in units of the contract
               (tests/graph/refdata.py laplacian_bound, docs/design/graph.md §4; decision GR55). The
               comparator scipy 1.18.1 runs laplacian on the same sparse inputs.

GR5 brings

  pagerank     every point of pagerank.npz (11 graphs; alpha 0.5, 0.85, 0.99, personalization, dangling,
               undirected, unweighted on dangling/50 and er/float100), one region per graph, one point per
               options: the l1 distance from Arb's solution in units of the contract tol + F (F = gamma_K /
               (1 - alpha), tests/graph/refdata.py pagerank_bound; decisions GR56, GR57) at the default
               tol = 1e-12 and, in the region "<graph> floor", at tol = 1e-15 (below the floor F of most
               points: the iteration then stops on its a priori count), with the entries' largest ulp and
               relative distance. The comparator networkx 3.7 runs pagerank at the region's tol / n (its
               successive-difference test), max_iter 10^4, judged by the same bound.
  betweenness  every point of betweenness.npz (13 graphs; directed and undirected, weighted and unweighted,
               normalized and endpoints on three of them), one region per graph and weighting, one point per
               node: the relative distance from the exact-rational Brandes in units of gamma_K
               (refdata.betweenness_k over the computed DAG, decision GR58; a reference 0 must be 0). The
               comparator networkx 3.7 runs betweenness_centrality on the same graphs.
"""

from __future__ import annotations

import collections
import math

import numpy as np

from .common import Custom, compare_libs, region, ulps

READINGS = (("directed", True), ("undirected", False))


def _all_pairs(lib, g, directed, fw, negative):
    """(dist, pred) of every pair as numpy arrays, by lib's FW (fw) or its definition's method."""
    if lib == "sd":
        import stardust as sd
        from tests.graph.forms import csr

        A = csr(g)
        if fw:
            r = sd.graph.floyd_warshall(A, directed=directed, return_predecessors=True)
        else:
            m = "J" if negative else "D"
            r = sd.graph.shortest_path(A, method=m, directed=directed, return_predecessors=True)
        return np.asarray(r.dist_matrix), np.asarray(r.predecessors)
    if lib == "scipy":
        import scipy.sparse as S
        import scipy.sparse.csgraph as C

        if g.matrix is not None:
            a = C.csgraph_from_dense(g.matrix)
        else:
            a = S.csr_array((g.data.astype(np.float64), g.indices, g.indptr), shape=g.shape)
        m = "FW" if fw else ("J" if negative else "D")
        d, p = C.shortest_path(a, method=m, directed=directed, return_predecessors=True)
        return d, p
    return None


def _walk(w, pred, s, v):
    """(edge count, sum of |w|) of the path the predecessor row draws to v (None: it does not reach s)."""
    nodes = [v]
    while nodes[-1] != s:
        p = int(pred[nodes[-1]])
        if p < 0 or len(nodes) > pred.size:
            return None
        nodes.append(p)
    return len(nodes) - 1, sum(abs(float(w[(a, b)])) for b, a in zip(nodes, nodes[1:]))


def _paths_regions(lib):
    from tests.graph import graphs as G
    from tests.graph import refdata

    ref = refdata.ref("paths")
    out = {}
    for reading, directed in READINGS:
        m = ref.methods[reading]
        by_graph = collections.defaultdict(list)
        for j, k in enumerate(m.index):
            if not m.fields["exact_pred"][j]:
                by_graph[str(ref.inputs["graph"][k])].append((j, int(ref.inputs["source"][k])))
        for gid, pts in sorted(by_graph.items()):
            g = G.get(gid)
            w = {(i, j): float(x) for i, j, x in g.edges()} if directed else G.undirected(g)
            w = {k: float(v) for k, v in w.items() if k[0] != k[1]}
            negative = any(x < 0 for x in w.values())
            for fw in (False, True):
                got = _all_pairs(lib, g, directed, fw, negative)
                if got is None:
                    continue
                D, P = got
                where, units, ul, rel = [], [], [], []
                for j, s in pts:
                    hi, lo = m.hi[j], m.lo[j]
                    for v in range(g.n):
                        if math.isinf(hi[v]) or v == s:
                            continue
                        x = float(D[s, v])
                        mine = _walk(w, P[s], s, v)
                        H = max(int(m.fields["edges"][j][v]), mine[0] if mine else 0)
                        S = max(float(m.fields["abs_sum"][j][v]), mine[1] if mine else 0.0)
                        where.append(f"{gid} s={s} v={v}")
                        units.append(refdata.units(x, float(hi[v]), float(lo[v]), H, S))
                        ul.append(float(ulps(x, float(hi[v]))))
                        rel.append(abs((x - hi[v]) - lo[v]) / abs(hi[v]) if hi[v] else abs(x))
                if where:
                    key = f"{gid} {reading}" + (" FW" if fw else "")
                    out[key] = region(where, ulp=np.array(ul), rel=np.array(rel), contract=np.array(units), bound=1.0)
    return out or None


ASSIGN_SHAPES = ((10, 10), (25, 25), (40, 40), (12, 30))
ASSIGN_SEEDS = 4


def _assign(lib, matrix, sparse, maximize):
    """(row_ind, col_ind) of lib's assignment of a dense cost matrix (sparse: its stored pairs)."""
    if lib == "sd":
        import stardust as sd

        if sparse:
            r = sd.graph.min_weight_full_bipartite_matching(sd.sparse.csr_array(matrix), maximize=maximize)
        else:
            r = sd.graph.linear_sum_assignment(matrix, maximize=maximize)
        return np.asarray(r.row_ind), np.asarray(r.col_ind)
    if lib == "scipy":
        if sparse:
            import scipy.sparse as S
            import scipy.sparse.csgraph as C

            return C.min_weight_full_bipartite_matching(S.csr_array(matrix), maximize=maximize)
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(matrix, maximize=maximize)
    return None


def _assign_point(lib, matrix, sparse, maximize, best, opt_abs, r):
    """(units, ulp, rel) of lib's assignment against the exact optimum best (a Fraction of the minimized
    costs: -matrix with maximize) whose sum of |c| is opt_abs."""
    from fractions import Fraction

    from tests.graph import refdata

    got = _assign(lib, matrix, sparse, maximize)
    if got is None:
        return None
    ri, ci = got
    c = -matrix if maximize else matrix
    exact = sum((Fraction(float(c[i, j])) for i, j in zip(ri, ci)), Fraction(0))
    abs_sum = float(np.abs(c[ri, ci]).sum()) + opt_abs
    hi = float(best)
    value = float(exact)
    return (
        refdata.assignment_units(float(exact - best), r, abs_sum),
        float(ulps(value, hi)),
        abs(float(exact - best)) / abs(hi) if hi else abs(value),
    )


def _assign_regions(lib):
    from tests.graph import definitions as D
    from tests.graph import graphs as G
    from tests.graph import refdata

    out = {}
    ref = refdata.ref("assignment")
    for k in range(ref.points):
        gid = str(ref.inputs["graph"][k])
        g = G.get(gid)
        if "float" not in g.tags:
            continue
        sparse = "cost" not in g.tags
        M = g.matrix if not sparse else g.dense(np.nan)
        if sparse:  # the stored pairs only: no pair is +inf for the dense exact solver
            M = np.where(np.isnan(M), np.inf, M)
        for kind in ("min", "max"):
            maximize = kind == "max"
            c = -M if maximize else M
            best_exact = D.assignment_exact([[None if not np.isfinite(x) else float(x) for x in row] for row in c])
            best, opt = best_exact
            opt_abs = float(np.abs(c[np.arange(len(opt)), opt]).sum())
            matrix = M if not sparse else np.where(np.isinf(M), 0.0, M)
            pt = _assign_point(lib, matrix, sparse, maximize, best, opt_abs, min(g.shape))
            if pt is None:
                continue
            out[f"{gid} {kind}"] = region([gid], contract=np.array([pt[0]]), bound=1.0, ulp=np.array([pt[1]]),
                                          rel=np.array([pt[2]]))  # fmt: skip
    for family, shape in [("random", s) for s in ASSIGN_SHAPES] + [("near ties", (30, 30))]:
        for maximize in (False, True):
            where, units, ul, rel = [], [], [], []
            for seed in range(ASSIGN_SEEDS):
                rng = np.random.Generator(np.random.PCG64([seed, *shape]))
                if family == "random":
                    M = rng.uniform(-1, 1, shape) * 10.0 ** rng.integers(-4, 5, shape)
                else:  # many optima of the integer part, told apart only in the last bits
                    M = rng.integers(0, 4, shape) * 1000.0 + rng.uniform(0, 1e-10, shape)
                c = -M if maximize else M
                best, opt = D.assignment_exact(c.tolist())
                opt_abs = float(np.abs(c[np.arange(shape[0]), opt]).sum())
                pt = _assign_point(lib, M, False, maximize, best, opt_abs, shape[0])
                if pt is None:
                    return None
                where.append(f"seed {seed}")
                units.append(pt[0])
                ul.append(pt[1])
                rel.append(pt[2])
            key = f"{family} {shape[0]}x{shape[1]} {'max' if maximize else 'min'}"
            out[key] = region(where, contract=np.array(units), bound=1.0, ulp=np.array(ul), rel=np.array(rel))
    return out or None


LAPLACIAN_OPTIONS = tuple((out, sym) for out in (False, True) for sym in (False, True))


def _laplacian(lib, g, normed, out, sym):
    """The dense laplacian of a test graph's sparse form by lib (None when lib has none)."""
    if lib == "sd":
        import stardust as sd
        from tests.graph.forms import csr

        return np.asarray(sd.graph.laplacian(csr(g), normed=normed, use_out_degree=out, symmetrized=sym).toarray())
    if lib == "scipy":
        import scipy.sparse.csgraph as C

        with np.errstate(all="ignore"):
            return C.laplacian(g.scipy().copy(), normed=normed, use_out_degree=out, symmetrized=sym).toarray()
    return None


def _laplacian_regions(lib):
    from tests.graph import graphs as G
    from tests.graph import refdata

    ref = refdata.ref("laplacian")
    m = ref.methods["L"]
    by = collections.defaultdict(list)
    for j, k in enumerate(m.index):
        gid = str(ref.inputs["graph"][k])
        if G.get(gid).data.dtype != np.float64 or gid == "isolated/5":
            continue
        by[(gid, bool(ref.inputs["normed"][k]))].append((j, bool(ref.inputs["use_out_degree"][k]),
                                                         bool(ref.inputs["symmetrized"][k])))  # fmt: skip
    out = {}
    for (gid, normed), pts in sorted(by.items()):
        g = G.get(gid)
        n = g.n
        absA = {(i, j): abs(float(w)) for i, j, w in g.edges() if i != j}
        where, units, ul, rel = [], [], [], []
        for j, uo, sym in pts:
            got = _laplacian(lib, g, normed, uo, sym)
            if got is None:
                return None
            hi, lo = m.hi[j].reshape(n, n), m.lo[j].reshape(n, n)
            terms = refdata.laplacian_terms(n, g.edges(), sym, uo)
            for a in range(n):
                for b in range(n):
                    touched = a == b if not normed else a != b
                    touched = touched or (sym and a != b and (a, b) in absA and (b, a) in absA)
                    if not touched or (hi[a, b] == 0 and lo[a, b] == 0 and a != b):
                        continue
                    pair = absA.get((a, b), 0.0) + absA.get((b, a), 0.0)
                    bd = refdata.laplacian_bound(normed, sym, a, b, terms, hi[a, b], pair)
                    x = float(got[a, b])
                    where.append(f"out={uo} sym={sym} ({a}, {b})")
                    units.append(refdata.laplacian_units(x, hi[a, b], lo[a, b], bd))
                    ul.append(float(ulps(x, float(hi[a, b]))))
                    rel.append(abs((x - hi[a, b]) - lo[a, b]) / abs(hi[a, b]) if hi[a, b] else abs(x))
        key = f"{gid} {'normed' if normed else 'unnormalized'}"
        out[key] = region(where, ulp=np.array(ul), rel=np.array(rel), contract=np.array(units), bound=1.0)
    return out or None


def _pagerank_lib(lib, g, opts, tol):
    """lib's PageRank of a test graph (numpy), or None when lib has none."""
    n = g.n
    if lib == "sd":
        import stardust as sd
        from tests.graph.forms import csr

        return np.asarray(sd.graph.pagerank(csr(g), tol=tol, **opts))
    if lib == "networkx":
        import networkx as nx

        from tests.graph import graphs as G

        D = nx.DiGraph()
        D.add_nodes_from(range(n))
        if opts["directed"]:
            edges = [(i, j, float(w)) for i, j, w in g.edges()]
        else:
            edges = [(i, j, float(w)) for (i, j), w in G.undirected(g).items()]
        D.add_weighted_edges_from(edges)
        kw = {"weight": None if opts["unweighted"] else "weight", "alpha": opts["alpha"]}
        for key in ("personalization", "dangling"):
            if opts[key] is not None:
                kw[key] = dict(enumerate(np.asarray(opts[key], dtype=np.float64)))
        x = nx.pagerank(D, tol=tol / n, max_iter=10**4, **kw)
        return np.array([x[v] for v in range(n)])
    return None


def _pagerank_regions(lib):
    from tests.graph import graphs as G
    from tests.graph import refdata
    from tests.graph.test_centrality import pagerank_k, reading

    ref = refdata.ref("pagerank")
    m = ref.methods["x"]
    by = collections.defaultdict(list)
    for j, k in enumerate(m.index):
        by[str(ref.inputs["graph"][k])].append((j, k))
    out = {}
    for gid, pts in sorted(by.items()):
        g = G.get(gid)
        for floor in (False, True):
            tol = 1e-15 if floor else 1e-12
            where, units, ul, rel = [], [], [], []
            for j, k in pts:
                pers, dang = ref.inputs["personalization"][k], ref.inputs["dangling"][k]
                opts = {
                    "alpha": float(ref.inputs["alpha"][k]),
                    "personalization": None if pers.size == 0 else pers,
                    "dangling": None if dang.size == 0 else dang,
                    "directed": bool(ref.inputs["directed"][k]),
                    "unweighted": bool(ref.inputs["unweighted"][k]),
                }
                x = _pagerank_lib(lib, g, opts, tol)
                if x is None:
                    return None
                hi, lo = m.hi[j], m.lo[j]
                err = math.fsum(abs((float(a) - b) - c) for a, b, c in zip(x, hi, lo))
                kk = pagerank_k(g.n, reading(g, opts["directed"]), opts["unweighted"])
                tag = ", ".join(f"{a}={'given' if isinstance(b, np.ndarray) else b}" for a, b in opts.items())
                where.append(tag)
                units.append(err / refdata.pagerank_bound(tol, opts["alpha"], kk))
                ul.append(float(np.max(ulps(x, hi))) if g.n else 0.0)
                rel.append(err)
            key = f"{gid} floor" if floor else gid
            out[key] = region(where, ulp=np.array(ul), rel=np.array(rel), contract=np.array(units), bound=1.0)
    return out or None


def _betweenness_lib(lib, g, directed, unweighted, normalized, endpoints):
    if lib == "sd":
        import stardust as sd
        from tests.graph.forms import csr

        return np.asarray(
            sd.graph.betweenness_centrality(
                csr(g), directed=directed, unweighted=unweighted, normalized=normalized, endpoints=endpoints
            )
        )
    if lib == "networkx":
        import networkx as nx

        from tests.graph import graphs as G

        D = nx.DiGraph() if directed else nx.Graph()
        D.add_nodes_from(range(g.n))
        if directed:
            D.add_weighted_edges_from((i, j, float(w)) for i, j, w in g.edges() if i != j)
        else:
            D.add_weighted_edges_from((i, j, float(w)) for (i, j), w in G.undirected(g).items() if i != j)
        x = nx.betweenness_centrality(
            D, normalized=normalized, endpoints=endpoints, weight=None if unweighted else "weight"
        )
        return np.array([x[v] for v in range(g.n)])
    return None


def _betweenness_regions(lib):
    from fractions import Fraction

    from tests.graph import definitions as D
    from tests.graph import graphs as G
    from tests.graph import refdata

    ref = refdata.ref("betweenness")
    m = ref.methods["bc"]
    by = collections.defaultdict(list)
    for j, k in enumerate(m.index):
        by[(str(ref.inputs["graph"][k]), bool(ref.inputs["unweighted"][k]))].append((j, k))
    out = {}
    stats = {}
    for (gid, unweighted), pts in sorted(by.items()):
        g = G.get(gid)
        where, units, ul, rel = [], [], [], []
        for j, k in pts:
            directed = bool(ref.inputs["directed"][k])
            normalized, endpoints = bool(ref.inputs["normalized"][k]), bool(ref.inputs["endpoints"][k])
            if (gid, directed, unweighted) not in stats:
                _, _, st = D.betweenness_dag(g.n, g.edges(), directed, unweighted)
                stats[(gid, directed, unweighted)] = refdata.betweenness_k(st["L"], st["dout"], st["din"],
                                                                           st["sigma"], g.n)  # fmt: skip
            gk = refdata.gamma(stats[(gid, directed, unweighted)])
            x = _betweenness_lib(lib, g, directed, unweighted, normalized, endpoints)
            if x is None:
                return None
            for v in range(g.n):
                want = Fraction(float(m.hi[j][v])) + Fraction(float(m.lo[j][v]))
                e = abs(Fraction(float(x[v])) - want)
                r = float(e / want) if want else (0.0 if x[v] == 0 else math.inf)
                where.append(f"directed={directed} normalized={normalized} endpoints={endpoints} v={v}")
                units.append(r / gk)
                ul.append(float(ulps(x[v], m.hi[j][v])))
                rel.append(r)
        key = f"{gid} {'unweighted' if unweighted else 'weighted'}"
        out[key] = region(where, ulp=np.array(ul), rel=np.array(rel), contract=np.array(units), bound=1.0)
    return out or None


def sweeps():
    return [
        Custom("graph", "paths", lambda: _paths_regions("sd"), compare_libs(_paths_regions, ("scipy",), seconds=300.0)),
        Custom(
            "graph",
            "assignment",
            lambda: _assign_regions("sd"),
            compare_libs(_assign_regions, ("scipy",), seconds=300.0),
        ),
        Custom(
            "graph",
            "laplacian",
            lambda: _laplacian_regions("sd"),
            compare_libs(_laplacian_regions, ("scipy",), seconds=300.0),
        ),
        Custom(
            "graph",
            "pagerank",
            lambda: _pagerank_regions("sd"),
            compare_libs(_pagerank_regions, ("networkx",), seconds=300.0),
        ),
        Custom(
            "graph",
            "betweenness",
            lambda: _betweenness_regions("sd"),
            compare_libs(_betweenness_regions, ("networkx",), seconds=300.0),
        ),
    ]
