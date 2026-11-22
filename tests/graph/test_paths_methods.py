"""The shortest paths on random graphs (docs/design/graph.md §1.3, §3.2, §4; milestone GR2; hypothesis):
'D', 'BF', 'J', 'auto' and dijkstra, bellman_ford, johnson give the same bytes (distances and
predecessors) from any sources and in either reading; integer weights give the exact distances and the
predecessor rule computed exactly; float weights stay within the contract; a negative cycle reachable
from the sources is a NegativeCycleError for every method; min_only is the smallest of the rows; limit
cuts the distances and keeps the predecessors; construct_dist_matrix gives the distances back bit for
bit; the blocked Floyd-Warshall is within the contract; yen gives the first K loopless paths."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import stardust as sd

from . import definitions as D
from . import refdata

GR = sd.graph
NONE = refdata.NONE
KINDS = ("int", "dyadic", "float", "wide", "zero", "negative")


def make(n, density, kind, symmetric, seed):
    """(edges [(i, j, w)], csr_array) of a random graph of the kind."""
    g = np.random.default_rng(seed)
    mask = g.random((n, n)) < density
    if symmetric:
        mask |= mask.T
    rows, cols = np.nonzero(mask)
    m = rows.size
    if kind == "int":
        w = g.integers(1, 4, m).astype(float)
    elif kind == "dyadic":
        w = g.integers(1, 64, m) / 8.0
    elif kind == "float":
        w = g.uniform(0.01, 1.01, m)
    elif kind == "wide":
        w = 10.0 ** g.uniform(-8, 8, m)
    elif kind == "zero":
        w = g.choice([0.0, 0.0, 1.0, 2.0, -0.0], m)
    else:
        w = g.integers(-2, 6, m).astype(float)
    edges = [(int(i), int(j), float(x)) for i, j, x in zip(rows, cols, w)]
    A = sd.sparse.csr_array((w, (rows, cols)), shape=(n, n)) if m else sd.sparse.csr_array((n, n))
    return edges, A


def exact_kind(kind):
    return kind in ("int", "dyadic", "zero", "negative")


def run(fn, *args, **kw):
    """(dist, pred) as numpy, or the NegativeCycleError."""
    try:
        r = fn(*args, return_predecessors=True, **kw)
    except GR.NegativeCycleError as e:
        return e
    return np.asarray(r.dist_matrix), np.asarray(r.predecessors)


def path_walk(w, pred, s, v, unweighted):
    """(left-to-right sum, edge count, sum of |w|) of the path the predecessor row draws to v."""
    nodes = [v]
    while nodes[-1] != s:
        nodes.append(int(pred[nodes[-1]]))
        assert len(nodes) <= pred.size + 1
    nodes.reverse()
    total = absum = 0.0
    for a, b in zip(nodes, nodes[1:]):
        x = 1.0 if unweighted else float(w[(a, b)])
        total += x
        absum += abs(x)
    return total, len(nodes) - 1, absum


def check_row(edges, n, s, dist, pred, directed, unweighted, kind, fw=False):
    """One source's distances against the exact ones: equal (exact kinds, unweighted), else the contract."""
    adj, loops = D.exact_weights(n, edges, directed, unweighted)
    d = D.shortest_exact(adj, loops, [s])
    assert d is not None
    xp, xh, _ = D.predecessors_exact(adj, d, [s])
    w = D.weights(n, edges, directed)
    exact = exact_kind(kind) or unweighted
    for v in range(n):
        if d[v] is None:
            assert math.isinf(dist[v]) and pred[v] == NONE, (s, v)
            continue
        if exact:
            assert dist[v] == d[v], (s, v)
            continue
        total, edges_hat, abs_hat = path_walk(w, pred, s, v, unweighted)
        if not fw:
            assert total == dist[v]
        _, _, abs_star = path_walk(w, np.array(xp), s, v, unweighted)
        H, S = max(edges_hat, xh[v]), max(abs_hat, abs_star)
        hi = float(d[v])
        lo = float(d[v] - type(d[v])(hi))
        assert refdata.units(float(dist[v]), hi, lo, H, S) <= 1, (s, v)
    if exact and not fw:
        assert pred.tolist() == xp, s


@given(
    st.integers(1, 13),
    st.sampled_from([0.1, 0.25, 0.5]),
    st.sampled_from(KINDS),
    st.booleans(),
    st.booleans(),
    st.booleans(),
    st.integers(0, 2**32 - 1),
)
def test_methods_give_one_answer(n, density, kind, symmetric, directed, unweighted, seed):
    edges, A = make(n, density, kind, symmetric, seed)
    negative = not unweighted and any(w < 0 for _, _, w in edges)
    g = np.random.default_rng(seed + 1)
    for indices in (None, int(g.integers(0, n)), g.integers(0, n, 3)):
        kw = dict(directed=directed, unweighted=unweighted, indices=indices)
        runs = {m: run(GR.shortest_path, A, method=m, **kw) for m in ("BF", "J", "auto")}
        runs["bellman_ford"] = run(GR.bellman_ford, A, **kw)
        runs["johnson"] = run(GR.johnson, A, **kw)
        if not negative:
            runs["D"] = run(GR.shortest_path, A, method="D", **kw)
            runs["dijkstra"] = run(GR.dijkstra, A, **kw)
        first = runs["BF"]
        for m, r in runs.items():
            if isinstance(first, Exception):
                assert isinstance(r, Exception) and str(r).split(":", 1)[1] == str(first).split(":", 1)[1], m
            else:
                assert r[0].tobytes() == first[0].tobytes() and r[1].tobytes() == first[1].tobytes(), m
        srcs = range(n) if indices is None else np.atleast_1d(indices)
        adj, loops = D.exact_weights(n, edges, directed, unweighted)
        cycles = [D.shortest_exact(adj, loops, [int(s)]) is None for s in srcs]
        if isinstance(first, Exception):
            assert any(cycles)
            continue
        assert not any(cycles)
        dist, pred = (first[0].reshape(-1, n), first[1].reshape(-1, n))
        for row, s in enumerate(srcs):
            check_row(edges, n, int(s), dist[row], pred[row], directed, unweighted, kind)
        if indices is None:
            c = GR.construct_dist_matrix(A, first[1], directed=directed) if not unweighted else None
            if c is not None:
                assert np.asarray(c).tobytes() == first[0].tobytes()


@given(
    st.integers(1, 13),
    st.sampled_from([0.1, 0.25, 0.5]),
    st.sampled_from(KINDS),
    st.booleans(),
    st.booleans(),
    st.integers(0, 2**32 - 1),
)
def test_floyd_warshall_within_the_contract(n, density, kind, directed, unweighted, seed):
    edges, A = make(n, density, kind, False, seed)
    adj, loops = D.exact_weights(n, edges, directed, unweighted)
    cyc = any(D.shortest_exact(adj, loops, [s]) is None for s in range(n))
    r = run(GR.floyd_warshall, A, directed=directed, unweighted=unweighted)
    assert isinstance(r, Exception) == cyc
    if cyc:
        assert str(r) == str(run(GR.bellman_ford, A, directed=directed, unweighted=unweighted)).replace(
            "bellman_ford", "floyd_warshall"
        )
        return
    dist, pred = r
    assert (
        run(GR.shortest_path, A, method="FW", directed=directed, unweighted=unweighted)[0].tobytes() == dist.tobytes()
    )
    plain = np.asarray(GR.floyd_warshall(A, directed=directed, unweighted=unweighted))
    assert plain.tobytes() == dist.tobytes()  # the predecessors do not change the distances
    for s in range(n):
        check_row(edges, n, s, dist[s], pred[s], directed, unweighted, kind, fw=True)


@given(
    st.integers(1, 13),
    st.sampled_from([0.1, 0.25, 0.5]),
    st.sampled_from(("int", "float", "zero")),
    st.booleans(),
    st.integers(0, 2**32 - 1),
)
def test_min_only_and_limit(n, density, kind, unweighted, seed):
    edges, A = make(n, density, kind, False, seed)
    g = np.random.default_rng(seed + 2)
    src = sorted({int(x) for x in g.integers(0, n, 3)})
    full = GR.dijkstra(A, indices=src, unweighted=unweighted)
    r = GR.dijkstra(A, indices=src, unweighted=unweighted, min_only=True, return_predecessors=True)
    assert np.asarray(r.dist_matrix).tobytes() == np.min(np.asarray(full), axis=0).tobytes()
    adj, loops = D.exact_weights(n, edges, True, unweighted)
    d = D.shortest_exact(adj, loops, src)
    if exact_kind(kind) or unweighted:
        xp, _, xs = D.predecessors_exact(adj, d, src)
        assert np.asarray(r.predecessors).tolist() == xp and np.asarray(r.sources).tolist() == xs
    # limit: the distances up to it, and the predecessors of the nodes kept
    whole = GR.dijkstra(A, indices=src[0], unweighted=unweighted, return_predecessors=True)
    dist, pred = np.asarray(whole.dist_matrix), np.asarray(whole.predecessors)
    finite = dist[np.isfinite(dist)]
    for limit in sorted({0.0, *finite.tolist(), math.inf})[:6]:
        for lim in (limit, np.nextafter(limit, -math.inf) if 0 < limit < math.inf else limit):
            cut = GR.dijkstra(A, indices=src[0], unweighted=unweighted, limit=lim, return_predecessors=True)
            keep = dist <= lim
            assert np.asarray(cut.dist_matrix).tobytes() == np.where(keep, dist, math.inf).tobytes()
            assert np.asarray(cut.predecessors).tolist() == np.where(keep, pred, NONE).tolist()


@given(
    st.integers(2, 8),
    st.sampled_from([0.3, 0.5, 0.8]),
    st.sampled_from(("int", "zero", "negative", "float")),
    st.booleans(),
    st.integers(0, 2**32 - 1),
)
def test_yen_is_the_first_k_loopless_paths(n, density, kind, directed, seed):
    edges, A = make(n, density, kind, False, seed)
    g = np.random.default_rng(seed + 3)
    s, t = (int(x) for x in g.integers(0, n, 2))
    adj, loops = D.exact_weights(n, edges, directed)
    if D.shortest_exact(adj, loops, [s]) is None:
        with pytest.raises(GR.NegativeCycleError):
            GR.yen(A, s, t, 5, directed=directed)
        return
    want = D.loopless_paths(adj, s, t)
    K = 12
    r = GR.yen(A, s, t, K, directed=directed, return_predecessors=True)
    dist, pred = np.asarray(r.dist_array), np.asarray(r.predecessors)
    assert dist.shape == (min(K, len(want)),) and pred.shape == (dist.size, n)
    w = D.weights(n, edges, directed)
    got = []
    for row in range(dist.size):
        total, edges_hat, _ = path_walk(w, pred[row], s, t, False)
        nodes = [t]
        while nodes[-1] != s:
            nodes.append(int(pred[row][nodes[-1]]))
        got.append(tuple(reversed(nodes)))
        assert total == dist[row]  # the length is the path's left-to-right sum
    if exact_kind(kind):
        assert got == [p for _, _, p in want[:K]]
        assert dist.tolist() == [float(x) for x, _, _ in want[:K]]
    else:  # each path a distinct loopless path, in order of its length
        assert len(set(got)) == len(got) and all(len(set(p)) == len(p) for p in got)
        assert dist.tolist() == sorted(dist.tolist())
