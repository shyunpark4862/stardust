"""articulation_points, bridges, biconnected_components and core_number (docs/api/graph.md §6,
docs/design/graph.md §2.6, §3.7, §4; milestone GR5): exactly the definitions of tests/graph/definitions.py
(nodes whose removal splits their component, edges whose removal disconnects their ends, blocks as the
edges no node separates, the repeated removal of low degrees) on every test graph and input form and on
random graphs; networkx 3.7 gives the same sets, the same partition and the same core numbers (its
records, tests/graph/data/networkx.json, and live)."""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as S

import stardust as sd

from . import definitions as D
from . import graphs as G
from . import refdata
from .conftest import LIVE
from .forms import csr, forms, square_ids

GR = sd.graph


def pairs_of(g):
    return {(i, j) for i, j, _ in g.edges()}


def rows(x):
    return [tuple(r) for r in np.asarray(x).tolist()]


def check(A, n, pairs, where):
    ap = GR.articulation_points(A)
    assert isinstance(ap, sd.ndarray) and ap.dtype == sd.int64 and ap.ndim == 1, where
    assert np.asarray(ap).tolist() == D.articulation_points(n, pairs), where
    br = GR.bridges(A)
    assert br.dtype == sd.int64 and br.ndim == 2 and br.shape[1] == 2, where
    assert rows(br) == D.bridges(n, pairs), where
    r = GR.biconnected_components(A)
    edges, labels = D.biconnected_components(n, pairs)
    assert type(r).__name__ == "BiconnectedComponentsResult" and isinstance(r.n_components, int), where
    assert r.edges.dtype == sd.int64 and r.edges.shape == (len(edges), 2) and r.labels.dtype == sd.int64, where
    assert rows(r.edges) == edges and np.asarray(r.labels).tolist() == labels, where
    assert r.n_components == (max(labels) + 1 if labels else 0), where
    cn = GR.core_number(A)
    assert cn.dtype == sd.int64 and cn.shape == (n,), where
    assert np.asarray(cn).tolist() == D.core_number(n, pairs), where


@pytest.mark.parametrize("gid", square_ids())
def test_definitions_for_every_form(gid):
    g = G.get(gid)
    pairs = pairs_of(g) if g.matrix is None else {(i, j) for i, j, _ in g.edges()}
    for form, A in forms(g).items():
        check(A, g.n, pairs, (gid, form))


def _random(n, d, seed, loops=False):
    r = np.random.Generator(np.random.PCG64([seed, n, int(d * 10)]))
    m = max(1, int(n * d))
    rows_, cols = r.integers(0, n, m), r.integers(0, n, m)
    keep = loops | (rows_ != cols)
    a = S.coo_array((r.uniform(0.5, 2, int(keep.sum())), (rows_[keep], cols[keep])), shape=(n, n)).tocsr()
    a.sum_duplicates()
    a.sort_indices()
    return a


@pytest.mark.parametrize("n", [1, 2, 3, 7, 20, 60])
def test_random_graphs(n):
    for seed in range(8):
        for d in (0.5, 1, 1.5, 3):
            a = _random(n, d, seed, loops=seed % 2 == 1)
            pairs = set(zip(*(x.tolist() for x in a.nonzero())))
            check(sd.sparse.csr_array(a), n, pairs, (seed, d))


def test_shapes():
    # a cycle has no articulation point; two triangles sharing a node have it; a path is all bridges
    cyc = np.zeros((5, 5))
    for i in range(5):
        cyc[i, (i + 1) % 5] = 1
    assert GR.articulation_points(cyc).size == 0 and GR.bridges(cyc).shape == (0, 2)
    assert GR.biconnected_components(cyc).n_components == 1
    assert np.asarray(GR.core_number(cyc)).tolist() == [2] * 5
    bow = np.zeros((5, 5))
    for i, j in ((0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)):
        bow[i, j] = 1
    assert np.asarray(GR.articulation_points(bow)).tolist() == [2]
    r = GR.biconnected_components(bow)
    assert rows(r.edges) == [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
    assert np.asarray(r.labels).tolist() == [0, 0, 0, 1, 1, 1]
    # n = 0 and 1, no edges, self loops only: empty answers of the right shapes
    for A in (np.zeros((0, 0)), np.zeros((1, 1)), np.eye(4), sd.sparse.csr_array((3, 3))):
        n = A.shape[0]
        assert GR.articulation_points(A).shape == (0,)
        assert GR.bridges(A).shape == (0, 2)
        r = GR.biconnected_components(A)
        assert r.n_components == 0 and r.edges.shape == (0, 2) and r.labels.shape == (0,)
        assert np.asarray(GR.core_number(A)).tolist() == [0] * n


def test_pattern_only_and_undirected():
    # a stored 0 and a stored NaN are edges; one direction is enough; the diagonal is left out
    A = sd.sparse.coo_array((np.array([0.0, np.nan, 1.0, 5.0]), (np.array([0, 2, 2, 3]), np.array([1, 1, 3, 3]))),
                            shape=(4, 4))  # fmt: skip
    assert rows(GR.bridges(A)) == [(0, 1), (1, 2), (2, 3)]
    assert np.asarray(GR.articulation_points(A)).tolist() == [1, 2]
    B = GR.biconnected_components(sd.sparse.csr_array(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]])))
    C = GR.biconnected_components(sd.sparse.csr_array(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]])))
    assert rows(B.edges) == rows(C.edges) and np.asarray(B.labels).tolist() == np.asarray(C.labels).tolist()


def test_results_are_new_arrays_and_read_only_tuples():
    g = csr(G.get("components/40"))
    a = GR.articulation_points(g)
    a[0] = 7  # a plain result is writable
    assert np.asarray(GR.articulation_points(g))[0] != 7
    r = GR.biconnected_components(g)
    for x in (r.edges, r.labels):
        with pytest.raises(ValueError):
            x[0] = 1


def test_networkx_records():
    data = refdata.data("networkx")
    seen = 0
    for key, v in data.items():
        if not key.endswith("/structure"):
            continue
        gid = key[len("graph/") : -len("/structure")]
        g = G.get(gid)
        assert v["digest"] == G.graph_digest(g), gid
        A = csr(g)
        assert np.asarray(GR.articulation_points(A)).tolist() == sorted(v["articulation_points"]), gid
        assert rows(GR.bridges(A)) == sorted(tuple(sorted(e)) for e in v["bridges"]), gid
        r = GR.biconnected_components(A)
        E, L = np.asarray(r.edges), np.asarray(r.labels)
        mine = sorted(sorted(map(tuple, E[L == c].tolist())) for c in range(r.n_components))
        theirs = sorted(sorted(tuple(sorted(e)) for e in comp) for comp in v["biconnected_edges"])
        assert mine == theirs, gid
        assert np.asarray(GR.core_number(A)).tolist() == v["core_number"], gid
        seen += 1
    assert seen >= 12


@LIVE
@pytest.mark.parametrize("seed", range(6))
def test_networkx_live(seed):
    for n, d in ((30, 1.2), (200, 1.5), (500, 3)):
        a = _random(n, d, seed)
        a.setdiag(0)
        a.eliminate_zeros()
        g = nx.from_scipy_sparse_array(a, create_using=nx.Graph)
        A = sd.sparse.csr_array(a)
        assert np.asarray(GR.articulation_points(A)).tolist() == sorted(nx.articulation_points(g))
        assert rows(GR.bridges(A)) == sorted(tuple(sorted(e)) for e in nx.bridges(g))
        r = GR.biconnected_components(A)
        E, L = np.asarray(r.edges), np.asarray(r.labels)
        mine = sorted(sorted(map(tuple, E[L == c].tolist())) for c in range(r.n_components))
        theirs = sorted(sorted(tuple(sorted(e)) for e in c) for c in nx.biconnected_component_edges(g))
        assert mine == theirs
        core = nx.core_number(g)
        assert np.asarray(GR.core_number(A)).tolist() == [core[v] for v in range(n)]


def test_core_number_selfloop_rule():
    # networkx refuses self loops (its record); sd leaves the diagonal out (docs/design/graph.md §2.1)
    assert refdata.data("networkx")["rule/core_number_selfloop"]["error"] == "NetworkXNotImplemented"
    A = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1.0]])
    assert np.asarray(GR.core_number(A)).tolist() == [1, 1, 1]


def test_large_graph_matches_networkx_structure():
    """A large sparse graph (past the GIL threshold): the bridges are the edges whose removal splits, the
    articulation points agree with the bridges' and blocks' counts (networkx on the same graph)."""
    from .graphs import random_graph

    ip, ix, w = random_graph(20000, 1, seed=31)
    A = sd.sparse.csr_array((w, ix, ip), shape=(20000, 20000))
    a = S.csr_array((w, ix, ip), shape=(20000, 20000))
    g = nx.from_scipy_sparse_array(a, create_using=nx.Graph)
    assert np.asarray(GR.articulation_points(A)).tolist() == sorted(nx.articulation_points(g))
    assert rows(GR.bridges(A)) == sorted(tuple(sorted(e)) for e in nx.bridges(g))
    core = nx.core_number(g)
    assert np.asarray(GR.core_number(A)).tolist() == [core[v] for v in range(20000)]
    assert GR.biconnected_components(A).n_components == sum(1 for _ in nx.biconnected_components(g))


@pytest.mark.parametrize("name", ["articulation_points", "bridges", "biconnected_components", "core_number"])
def test_errors(name):
    f = getattr(GR, name)
    with pytest.raises(ValueError, match=rf"^{name}\(\): 'csgraph' must be a square matrix"):
        f(np.zeros((2, 3)))
    with pytest.raises(TypeError, match=rf"^{name}\(\): 'csgraph' must be real"):
        f(np.eye(2) * 1j)
    with pytest.raises(TypeError, match=rf"^{name}\(\).*scipy.sparse"):
        f(S.eye_array(3))
    with pytest.raises(TypeError):
        f(np.eye(2), True)
