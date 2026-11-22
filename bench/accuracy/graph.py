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
(tests/graph/test_paths_reference.py). Later: `assignment` with GR3, `laplacian` with GR4, `pagerank` and
`betweenness` with GR5.
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


def sweeps():
    return [
        Custom("graph", "paths", lambda: _paths_regions("sd"), compare_libs(_paths_regions, ("scipy",), seconds=300.0))
    ]
