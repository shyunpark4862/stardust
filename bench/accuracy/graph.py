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

Later: `laplacian` with GR4, `pagerank` and `betweenness` with GR5.
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


def sweeps():
    return [
        Custom("graph", "paths", lambda: _paths_regions("sd"), compare_libs(_paths_regions, ("scipy",), seconds=300.0)),
        Custom(
            "graph",
            "assignment",
            lambda: _assign_regions("sd"),
            compare_libs(_assign_regions, ("scipy",), seconds=300.0),
        ),
    ]
