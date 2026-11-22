"""yen against the enumeration of every loopless path (tests/graph/ref/yen.npz; docs/design/graph.md §3.2
Yen, §4; milestone GR2): on integer weights the first K paths in the order (exact length, edge count,
node sequence) exactly, elsewhere each length its path's left-to-right sum and within the contract;
s == t, K = 0, unweighted, undirected, negative weights and cycles, the result's form, the errors, and
scipy 1.18.1 (same paths where nothing ties; its own tie order on the diamond)."""

import math

import numpy as np
import pytest
import scipy.sparse as S
import scipy.sparse.csgraph as CG

import stardust as sd

from . import definitions as D
from . import graphs as G
from . import refdata
from .conftest import LIVE
from .forms import csr

GR = sd.graph
REF = refdata.ref("yen")
NONE = refdata.NONE
K = refdata.YEN_K


def ref_points():
    m = REF.methods["lengths"]
    for j, k in enumerate(m.index):
        flat = list(m.fields["paths"][j])
        paths = []
        while flat:
            cut = flat.index(-1)
            paths.append(tuple(flat[:cut]))
            flat = flat[cut + 1 :]
        yield (
            str(REF.inputs["graph"][k]),
            int(REF.inputs["source"][k]),
            int(REF.inputs["sink"][k]),
            m.hi[j],
            m.lo[j],
            m.fields["edges"][j],
            paths,
        )


def rows_to_paths(pred, s, t):
    out = []
    for row in pred:
        nodes = [t]
        while nodes[-1] != s:
            nodes.append(int(row[nodes[-1]]))
        out.append(tuple(reversed(nodes)))
    return out


def test_yen_against_the_enumeration():
    seen = 0
    for gid, s, t, hi, lo, edges, paths in ref_points():
        g = G.get(gid)
        A = csr(g)
        r = GR.yen(A, s, t, K, return_predecessors=True)
        dist, pred = np.asarray(r.dist_array), np.asarray(r.predecessors)
        assert dist.shape == (len(paths),) and pred.shape == (len(paths), g.n), (gid, s, t)
        got = rows_to_paths(pred, s, t) if s != t else [(s,)] * len(paths)
        w = D.weights(g.n, g.edges(), True)
        sums = []
        for p in got:
            x, absum = 0.0, 0.0
            for a, b in zip(p, p[1:]):
                x += float(w[(a, b)])
                absum += abs(float(w[(a, b)]))
            sums.append((x, len(p) - 1, absum))
        assert [x for x, _, _ in sums] == dist.tolist(), (gid, s, t)
        if all(lo == 0) and all(float(x).is_integer() for x in hi):  # exact sums: the very paths
            assert got == paths and dist.tolist() == hi.tolist(), (gid, s, t)
        else:
            H = max([e for _, e, _ in sums] + list(edges))
            Smax = max([a for _, _, a in sums] + [0.0])
            for k in range(len(paths)):
                assert refdata.units(float(dist[k]), hi[k], lo[k], H, Smax) <= 1, (gid, s, t, k)
        seen += 1
    assert seen > 40


def test_source_is_sink_and_k_zero():
    A = csr(G.get("diamond/4"))
    assert np.asarray(GR.yen(A, 2, 2, 3)).tolist() == [0.0]
    r = GR.yen(A, 2, 2, 3, return_predecessors=True)
    assert np.asarray(r.predecessors).tolist() == [[NONE] * 4]
    assert np.asarray(GR.yen(A, 0, 3, 0)).shape == (0,)
    assert np.asarray(GR.yen(A, 0, 3, 0, return_predecessors=True).predecessors).shape == (0, 4)
    assert np.asarray(GR.yen(A, 3, 0, 5)).shape == (0,)  # no path


def test_result_form():
    A = csr(G.get("grid2d/3x3"))
    d = GR.yen(A, 0, 8, 4)
    assert isinstance(d, sd.ndarray) and d.dtype == sd.float64 and d._flags()["writeable"]
    r = GR.yen(A, 0, 8, 4, return_predecessors=True)
    assert type(r).__name__ == "YenResult" and r._fields == ("dist_array", "predecessors")
    assert r.predecessors.dtype == sd.int64 and not r.predecessors._flags()["writeable"]
    d2, p2 = r
    assert np.asarray(d2).tolist() == np.asarray(d).tolist()


def test_unweighted_and_undirected_against_the_definition():
    for gid in ("oneway/12", "twoway/12", "zero/15", "star/9"):
        g = G.get(gid)
        A = csr(g)
        for directed in (True, False):
            for unweighted in (False, True):
                adj, _ = D.exact_weights(g.n, g.edges(), directed, unweighted)
                for s, t in ((0, g.n - 1), (1, 2)):
                    want = D.loopless_paths(adj, s, t)[:8]
                    r = GR.yen(A, s, t, 8, directed=directed, unweighted=unweighted, return_predecessors=True)
                    got = rows_to_paths(np.asarray(r.predecessors), s, t)
                    if unweighted:
                        assert got == [p for _, _, p in want], (gid, directed, s, t)
                    assert len(got) == len(want)


def test_negative_weights_and_cycles():
    g = G.get("negative/dag30")
    A = csr(g)
    adj, _ = D.exact_weights(g.n, g.edges(), True)
    want = D.loopless_paths(adj, 0, 29)[:10]
    got = rows_to_paths(np.asarray(GR.yen(A, 0, 29, 10, return_predecessors=True).predecessors), 0, 29)
    assert got == [p for _, _, p in want]
    reach = csr(G.get("negative/reach12"))
    with pytest.raises(GR.NegativeCycleError, match=r"^yen\(\): negative cycle through node \d+$"):
        GR.yen(reach, 0, 5, 3)
    unreach = csr(G.get("negative/unreach12"))
    assert np.asarray(GR.yen(unreach, 0, 7, 2)).size >= 1  # a cycle the source does not reach is no error
    with pytest.raises(GR.NegativeCycleError):
        GR.yen(unreach, 8, 9, 2)
    with pytest.raises(GR.NegativeCycleError):  # an undirected negative edge is a cycle
        GR.yen(csr(G.get("negative/undirected6")), 0, 3, 2, directed=False)


def test_errors():
    A = csr(G.get("diamond/4"))
    with pytest.raises(TypeError, match=r"^yen\(\): 'K' must be an int, not float$"):
        GR.yen(A, 0, 3, 2.0)
    with pytest.raises(TypeError, match=r"^yen\(\): 'K' must be an int, not bool$"):
        GR.yen(A, 0, 3, True)
    with pytest.raises(ValueError, match=r"^yen\(\): 'K' must be 0 or more, not -1$"):
        GR.yen(A, 0, 3, -1)
    with pytest.raises(ValueError, match=r"^yen\(\): 'sink' must be a node of the graph \(0 \.\. 3\), not 4$"):
        GR.yen(A, 0, 4, 1)
    with pytest.raises(TypeError, match=r"^yen\(\): 'source' must be an int"):
        GR.yen(A, 0.0, 3, 1)
    with pytest.raises(TypeError, match=r"^yen\(\): takes 4 positional arguments"):
        GR.yen(A, 0, 3, 1, True)
    bad = sd.sparse.csr_array(np.array([[0, np.nan], [0, 0]]))
    with pytest.raises(ValueError, match=r"^yen\(\): stored edge weights must be finite$"):
        GR.yen(bad, 0, 1, 1)
    assert np.asarray(GR.yen(A, 0, 3, 10**30)).size == 2  # a huge K is as many paths as there are


@LIVE
def test_scipy_same_paths_without_ties():
    for gid in ("er/float100", "heavytail/120"):
        g = G.get(gid)
        a = S.csr_array((g.data.astype(float), g.indices, g.indptr), shape=g.shape)
        for s, t in ((0, g.n - 1), (3, 17)):
            want = CG.yen(a, s, t, 6, return_predecessors=True)
            got = GR.yen(csr(g), s, t, 6, return_predecessors=True)
            assert np.allclose(np.asarray(got.dist_array), want[0], rtol=1e-14, atol=0), (gid, s, t)
            assert np.asarray(got.predecessors).tolist() == want[1].tolist(), (gid, s, t)


def test_scipy_tie_order_differs_on_the_diamond():
    rec = refdata.data("scipy")["rule/yen"]
    scipy_pred = rec["K3"]["value"][1]["array"]
    assert scipy_pred == [-9999, -9999, 0, 2, -9999, 0, -9999, 1]  # scipy: 0-2-3 first
    r = GR.yen(csr(G.get("diamond/4")), 0, 3, 3, return_predecessors=True)
    assert np.asarray(r.predecessors).ravel().tolist() == [NONE, 0, NONE, 1, NONE, NONE, 0, 2]  # sd: 0-1-3 first
    assert np.asarray(r.dist_array).tolist() == [2.0, 2.0] and math.isfinite(r.dist_array[1])
