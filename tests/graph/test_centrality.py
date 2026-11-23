"""pagerank and betweenness_centrality (docs/api/graph.md §6, docs/design/graph.md §2.6, §3.7, §4; decisions
GR19, GR20, GR28, GR56-GR58; milestone GR5).

PageRank: within the contract tol + F of Arb's solution (tests/graph/ref/pagerank.npz) for every option, F =
gamma_K / (1 - alpha) the rounding of one step (tests/graph/refdata.py pagerank_k, pagerank_bound); the
iteration count at most the a priori one; networkx 3.7 at a tight tolerance agrees (its records and live).

Betweenness: within gamma_K of the exact-rational Brandes (relative, every term non-negative;
refdata.betweenness_k) over the computed shortest-path DAG (tests/graph/definitions.py betweenness_dag), on
every test graph of positive weights; the exact references of betweenness.npz (the DAG of the exact
distances) the same; networkx 3.7 agrees to 1e-12; path counts past the double range (a chain of diamonds)
are exact."""

import math
from fractions import Fraction

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
REF = {name: refdata.ref(name) for name in ("pagerank", "betweenness")}


# ---- pagerank ----------------------------------------------------------------------------------------


def reading(g, directed):
    """[(i, j, w)] of the graph a function reads: its stored entries or the undirected reading."""
    if directed:
        return [(i, j, float(w)) for i, j, w in g.edges()]
    return [(i, j, float(w)) for (i, j), w in G.undirected(g).items()]


def pagerank_k(n, edges, unweighted):
    """refdata.pagerank_k of a reading [(i, j, w)]."""
    indeg, outdeg, outw = [0] * n, [0] * n, [0.0] * n
    for i, j, w in edges:
        indeg[j] += 1
        outdeg[i] += 1
        outw[i] += 1.0 if unweighted else w
    dangling = sum(1 for v in range(n) if outw[v] == 0)
    return refdata.pagerank_k(n, max(indeg, default=0), max(outdeg, default=0), dangling, not unweighted)


def pagerank_points():
    r = REF["pagerank"]
    m = r.methods["x"]
    for j, k in enumerate(m.index):
        yield (
            G.get(str(r.inputs["graph"][k])),
            float(r.inputs["alpha"][k]),
            r.inputs["personalization"][k],
            r.inputs["dangling"][k],
            bool(r.inputs["directed"][k]),
            bool(r.inputs["unweighted"][k]),
            m.hi[j],
            m.lo[j],
        )


def l1(x, hi, lo):
    return math.fsum(abs((float(a) - b) - c) for a, b, c in zip(np.asarray(x), hi, lo))


@pytest.mark.parametrize("tol", [1e-12, 1e-8, 1e-3])
def test_pagerank_reference(tol):
    seen = 0
    for g, alpha, pers, dang, directed, unweighted, hi, lo in pagerank_points():
        opts = {
            "alpha": alpha,
            "directed": directed,
            "unweighted": unweighted,
            "personalization": None if pers.size == 0 else pers,
            "dangling": None if dang.size == 0 else dang,
            "tol": tol,
        }
        x = GR.pagerank(csr(g), **opts)
        assert x.dtype == sd.float64 and x.shape == (g.n,)
        k = pagerank_k(g.n, reading(g, directed), unweighted)
        assert l1(x, hi, lo) <= refdata.pagerank_bound(tol, alpha, k), (g.id, opts)
        seen += 1
    assert seen == REF["pagerank"].points


def test_pagerank_every_form():
    g = G.get("dangling/50")
    want = np.asarray(GR.pagerank(csr(g)))
    for form, A in forms(g).items():
        assert np.asarray(GR.pagerank(A)).tobytes() == want.tobytes(), form


def test_pagerank_iteration_count():
    """The a posteriori bound stops the iteration before the a priori count ceil(log(tol / 2) / log alpha)
    + 1 on these graphs, and never past it; alpha = 0 is v after one step."""
    from . import graphs

    ip, ix, w = graphs.random_graph(3000, 5, seed=9)
    A = sd.sparse.csr_array((w, ix, ip), shape=(3000, 3000))
    p = sd.ones(3000)
    for alpha in (0.0, 0.3, 0.85, 0.99):
        for tol in (1e-3, 1e-12):
            x, its = GR._centrality._core._pagerank(A.indptr, A.indices, A.data, 3000, True, alpha, p, p, tol)
            kmax = 1 if alpha == 0 else math.ceil(math.log(tol / 2) / math.log(alpha)) + 1
            assert 1 <= its <= kmax, (alpha, tol, its, kmax)
            if alpha == 0:
                assert its == 1 and np.all(np.asarray(x) == 1 / 3000)
            elif alpha >= 0.85 and tol == 1e-12:
                assert its < kmax, (alpha, its, kmax)


def test_pagerank_networkx_records():
    data = refdata.data("networkx")
    seen = 0
    for key, v in data.items():
        if not key.endswith("/pagerank"):
            continue
        gid = key[len("graph/") : -len("/pagerank")]
        g = G.get(gid)
        assert v["digest"] == G.graph_digest(g)
        got = np.asarray(GR.pagerank(csr(g)))
        assert np.abs(got - np.array(v["tight"]["value"])).sum() < 1e-11, gid
        seen += 1
    assert seen >= 6


def _random(n, d, seed, symmetric=False, zeros=False):
    r = np.random.Generator(np.random.PCG64([seed, n, d]))
    rows, cols = r.integers(0, n, n * d), r.integers(0, n, n * d)
    w = r.uniform(0.1, 3, n * d)
    if zeros:
        w[r.random(n * d) < 0.2] = 0.0
    a = S.coo_array((w, (rows, cols)), shape=(n, n)).tocsr()
    a.sum_duplicates()
    if symmetric:
        a = a.maximum(a.T).tocsr()
    a.sort_indices()
    return a


@LIVE
@pytest.mark.parametrize("seed", range(4))
def test_pagerank_networkx_live(seed):
    for n, d in ((40, 2), (300, 3)):
        a = _random(n, d, seed)  # self loops included: an edge for both (decision GR28)
        A = sd.sparse.csr_array(a)
        r = np.random.Generator(np.random.PCG64([seed, 77]))
        pers = r.uniform(0, 1, n) * (r.random(n) < 0.7)
        dang = r.uniform(0, 1, n)
        g = nx.from_scipy_sparse_array(a, create_using=nx.DiGraph)
        # networkx stops when successive iterates differ by less than n tol in l1: within about
        # alpha / (1 - alpha) n tol of the solution, sd within 1e-12
        close = 1e-12 + 6 * n * 1e-15
        for opts, nxopts in (
            ({}, {}),
            ({"alpha": 0.6}, {"alpha": 0.6}),
            (
                {"personalization": pers, "dangling": dang},
                {"personalization": dict(enumerate(pers)), "dangling": dict(enumerate(dang))},
            ),  # fmt: skip
        ):
            want = nx.pagerank(g, tol=1e-15, max_iter=10000, **nxopts)
            got = np.asarray(GR.pagerank(A, **opts))
            assert np.abs(got - np.array([want[v] for v in range(n)])).sum() < close, opts
        want = nx.pagerank(g, weight=None, tol=1e-15, max_iter=10000)
        got = np.asarray(GR.pagerank(A, unweighted=True))
        assert np.abs(got - np.array([want[v] for v in range(n)])).sum() < close
        s = _random(n, d, seed, symmetric=True)
        want = nx.pagerank(nx.from_scipy_sparse_array(s, create_using=nx.Graph), tol=1e-15, max_iter=10000)
        got = np.asarray(GR.pagerank(sd.sparse.csr_array(s), directed=False))
        assert np.abs(got - np.array([want[v] for v in range(n)])).sum() < close


def test_pagerank_rules():
    # a row whose stored weights are all 0 is dangling (as an empty row); unweighted counts it
    A = sd.sparse.csr_array((np.array([0.0, 1.0, 1.0]), np.array([1, 2, 0]), np.array([0, 1, 2, 3])), shape=(3, 3))
    B = sd.sparse.csr_array((np.array([1.0, 1.0]), np.array([2, 0]), np.array([0, 0, 1, 2])), shape=(3, 3))
    assert np.asarray(GR.pagerank(A)).tobytes() == np.asarray(GR.pagerank(B)).tobytes()
    assert np.asarray(GR.pagerank(A, unweighted=True)).tobytes() != np.asarray(GR.pagerank(A)).tobytes()
    # a self loop is an edge (decision GR28): a node with only a self loop keeps its mass
    loop = np.array([[1.0, 0], [1.0, 0]])
    x = np.asarray(GR.pagerank(loop))
    want = nx.pagerank(nx.from_numpy_array(loop, create_using=nx.DiGraph), tol=1e-15, max_iter=1000)
    assert abs(x[0] - want[0]) + abs(x[1] - want[1]) < 1e-12 and x[0] > 0.9
    # n = 0, isolated nodes, personalization of ints and bools, a dense matrix
    assert GR.pagerank(np.zeros((0, 0))).shape == (0,)
    assert np.allclose(np.asarray(GR.pagerank(np.zeros((4, 4)))), 0.25, rtol=0, atol=1e-16)
    x = np.asarray(GR.pagerank(np.zeros((3, 3)), personalization=[0, 2, 2]))
    assert x[0] == 0 and x[1] == x[2] == 0.5
    assert np.asarray(GR.pagerank(np.zeros((2, 2)), personalization=[True, False])).tolist() == [1.0, 0.0]


def test_pagerank_errors():
    A = csr(G.get("er/float100"))
    for alpha in (1, 1.0, -0.1, math.nan, math.inf):
        with pytest.raises(ValueError, match=r"^pagerank\(\): 'alpha' must be >= 0 and < 1"):
            GR.pagerank(A, alpha=alpha)
    for tol in (0, -1e-3, math.inf, math.nan):
        with pytest.raises(ValueError, match=r"^pagerank\(\): 'tol' must be positive and finite"):
            GR.pagerank(A, tol=tol)
    for bad in ("0.85", None, True):
        with pytest.raises(TypeError, match=r"^pagerank\(\): 'alpha'"):
            GR.pagerank(A, alpha=bad)
    p = np.ones(100)
    with pytest.raises(ValueError, match=r"one entry per node \(100\), not 99"):
        GR.pagerank(A, personalization=p[:99])
    with pytest.raises(ValueError, match=r"'dangling' must be >= 0"):
        GR.pagerank(A, dangling=-p)
    with pytest.raises(ValueError, match=r"'personalization' must be finite"):
        GR.pagerank(A, personalization=np.where(np.arange(100) == 3, np.nan, 1.0))
    with pytest.raises(ValueError, match=r"'personalization' must have a positive sum"):
        GR.pagerank(A, personalization=np.zeros(100))
    with pytest.raises(ValueError, match=r"'personalization' must be 1-D"):
        GR.pagerank(A, personalization=np.ones((10, 10)))
    with pytest.raises(TypeError, match=r"'personalization' must be real"):
        GR.pagerank(A, personalization=np.ones(100) * 1j)
    with pytest.raises(TypeError, match=r"'personalization' must be a 1-D array of numbers"):
        GR.pagerank(A, personalization="abc")
    neg = csr(G.get("negative/dag30"))
    with pytest.raises(ValueError, match=r"^pagerank\(\): negative edge weights are not allowed"):
        GR.pagerank(neg)
    GR.pagerank(neg, unweighted=True)  # the pattern only
    with pytest.raises(ValueError, match=r"^pagerank\(\): stored edge weights must be finite"):
        GR.pagerank(csr(G.get("nonfinite/nan")))
    with pytest.raises(TypeError):
        GR.pagerank(A, 0.85)


# ---- betweenness -------------------------------------------------------------------------------------


def positive(g):
    return g.n > 0 and all(0 < float(w) < math.inf for _, _, w in g.edges()) and "cost" not in g.tags


def units(got, want, k):
    """|got - want| / (gamma_K want) per node (want exact; 0 must be exactly 0)."""
    out = 0.0
    for x, y in zip(np.asarray(got), want):
        if y == 0:
            assert x == 0
            continue
        out = max(out, float(abs(Fraction(float(x)) - y) / y) / refdata.gamma(k))
    return out


OPTIONS = ((True, False), (False, False), (True, True), (False, True))


@pytest.mark.parametrize("gid", [i for i in square_ids() if G.get(i).n <= 130])
def test_betweenness_definition(gid):
    g = G.get(gid)
    A = csr(g)
    for directed in (True, False):
        for unweighted in (True, False):
            if not unweighted and not positive(g):
                if g.nnz:
                    with pytest.raises(ValueError):
                        GR.betweenness_centrality(A, directed=directed)
                continue
            raw, ends, st = D.betweenness_dag(g.n, g.edges(), directed, unweighted)
            k = refdata.betweenness_k(st["L"], st["dout"], st["din"], st["sigma"], g.n)
            for normalized, endpoints in OPTIONS:
                want = D.betweenness_scaled(g.n, raw, ends, directed, normalized, endpoints)
                got = GR.betweenness_centrality(
                    A, directed=directed, unweighted=unweighted, normalized=normalized, endpoints=endpoints
                )
                assert got.dtype == sd.float64 and got.shape == (g.n,)
                assert units(got, want, k) <= 1, (directed, unweighted, normalized, endpoints)


def test_betweenness_reference():
    """betweenness.npz: Brandes over the exact distances in exact rationals. The computed DAG is the exact
    one on these graphs, so the same contract holds against them."""
    r = REF["betweenness"]
    m = r.methods["bc"]
    stats = {}
    for j, k in enumerate(m.index):
        g = G.get(str(r.inputs["graph"][k]))
        directed, unweighted = bool(r.inputs["directed"][k]), bool(r.inputs["unweighted"][k])
        key = (g.id, directed, unweighted)
        if key not in stats:
            raw, ends, st = D.betweenness_dag(g.n, g.edges(), directed, unweighted)
            stats[key] = refdata.betweenness_k(st["L"], st["dout"], st["din"], st["sigma"], g.n)
        got = GR.betweenness_centrality(
            csr(g),
            directed=directed,
            unweighted=unweighted,
            normalized=bool(r.inputs["normalized"][k]),
            endpoints=bool(r.inputs["endpoints"][k]),
        )
        want = [Fraction(float(a)) + Fraction(float(b)) for a, b in zip(m.hi[j], m.lo[j])]
        # the reference is a double-double: within 2^-104 relative of the exact value
        assert units(got, want, stats[key]) <= 1 + 1e-9, (key, j)


def test_betweenness_networkx_records():
    data = refdata.data("networkx")
    seen = 0
    for key, v in data.items():
        if not key.endswith("/betweenness"):
            continue
        g = G.get(key[len("graph/") : -len("/betweenness")])
        for opt, out in v.items():
            if opt == "digest":
                continue
            dch, wch, flags = opt.split("/")
            got = GR.betweenness_centrality(
                csr(g), directed=dch == "d", unweighted=wch == "u", normalized=flags[0] == "1",
                endpoints=flags[1] == "1",
            )  # fmt: skip
            assert np.allclose(np.asarray(got), out["value"], rtol=1e-12, atol=0), (key, opt)
            seen += 1
    assert seen >= 40
    small = data["rule/betweenness_small_n"]
    two = np.array([[0, 1], [1, 0.0]])
    got = GR.betweenness_centrality(two, directed=False)
    assert np.asarray(got).tolist() == list(small["n2_normalized"]["value"].values())
    got = GR.betweenness_centrality(two, directed=False, endpoints=True)
    assert np.asarray(got).tolist() == list(small["n2_endpoints"]["value"].values())


@LIVE
@pytest.mark.parametrize("seed", range(4))
def test_betweenness_networkx_live(seed):
    for n, d in ((30, 2), (150, 3)):
        a = _random(n, d, seed)
        a.setdiag(0)
        a.eliminate_zeros()
        A = sd.sparse.csr_array(a)
        g = nx.from_scipy_sparse_array(a, create_using=nx.DiGraph)
        s = _random(n, d, seed, symmetric=True)
        s.setdiag(0)
        s.eliminate_zeros()
        h = nx.from_scipy_sparse_array(s, create_using=nx.Graph)
        for normalized, endpoints in OPTIONS:
            for weight in (None, "weight"):
                for graph, B, directed in ((g, A, True), (h, sd.sparse.csr_array(s), False)):
                    want = nx.betweenness_centrality(graph, normalized=normalized, endpoints=endpoints, weight=weight)
                    got = GR.betweenness_centrality(
                        B, directed=directed, unweighted=weight is None, normalized=normalized, endpoints=endpoints
                    )
                    assert np.allclose(np.asarray(got), [want[v] for v in range(n)], rtol=1e-12, atol=1e-15)


def test_betweenness_counts_past_the_double_range():
    """A directed chain of m diamonds h_i -> a_i, b_i -> h_{i+1}: 2^m shortest paths end to end, past 2^1024
    for m = 1100. bc(h_i) = 9 i (m - i), bc(a_i) = (3i + 1)(3(m - i) - 2) / 2 (unnormalized), exactly: the
    path counts are powers of 2 and every ratio 1 or 1/2."""
    m = 1100
    n = 3 * m + 1
    hub = lambda i: 3 * i  # noqa: E731
    rows, cols = [], []
    for i in range(m):
        for x in (hub(i) + 1, hub(i) + 2):
            rows += [hub(i), x]
            cols += [x, hub(i + 1)]
    A = sd.sparse.csr_array(S.csr_array((np.ones(len(rows)), (rows, cols)), shape=(n, n)))
    want = np.zeros(n)
    for i in range(m + 1):
        want[hub(i)] = 9 * i * (m - i)
    for i in range(m):
        want[hub(i) + 1] = want[hub(i) + 2] = (3 * i + 1) * (3 * (m - i) - 2) / 2
    for unweighted in (True, False):
        got = np.asarray(GR.betweenness_centrality(A, unweighted=unweighted, normalized=False))
        assert np.array_equal(got, want)


def test_betweenness_absorbed_steps():
    """A weight absorbed by a large distance: the steps of equal computed distance are ordered by the edge
    count (decision GR58), so every reached node has a predecessor and no ratio is 0/0."""
    big, tiny = 1e17, 1.0
    W = np.zeros((4, 4))
    W[0, 3], W[3, 1], W[1, 2] = big, tiny, tiny  # d = 1e17 for 3, 1, 2: absorbed
    raw, ends, st = D.betweenness_dag(4, [(0, 3, big), (3, 1, tiny), (1, 2, tiny)], True, False)
    got = GR.betweenness_centrality(W, normalized=False)
    assert np.asarray(got).tolist() == [float(x) for x in raw] == [0.0, 2.0, 0.0, 2.0]


def test_betweenness_errors():
    A = csr(G.get("zero/15"))
    with pytest.raises(ValueError, match=r"^betweenness_centrality\(\): edge weights must be positive"):
        GR.betweenness_centrality(A)
    GR.betweenness_centrality(A, unweighted=True)
    with pytest.raises(ValueError, match=r"^betweenness_centrality\(\): edge weights must be positive"):
        GR.betweenness_centrality(csr(G.get("negative/dag30")), directed=False)
    with pytest.raises(ValueError, match=r"^betweenness_centrality\(\): stored edge weights must be finite"):
        GR.betweenness_centrality(csr(G.get("nonfinite/inf")))
    for opt in ("directed", "unweighted", "normalized", "endpoints"):
        with pytest.raises(TypeError, match=rf"^betweenness_centrality\(\): '{opt}' must be True or False"):
            GR.betweenness_centrality(A, **{opt: 1})
    with pytest.raises(TypeError):
        GR.betweenness_centrality(A, False)
    assert GR.betweenness_centrality(np.zeros((0, 0))).shape == (0,)
