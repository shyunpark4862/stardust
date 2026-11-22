"""The shortest paths against the exact references of tests/graph/ref/paths.npz (docs/design/graph.md §4,
§7.2; milestone GR2): on every test graph, source and reading (directed, unweighted, undirected,
undirected unweighted) 'D', 'BF', 'J' and 'auto' give the same bytes; where every path sum is exact
(integer and dyadic weights, unweighted) the distances are the exact shortest distances and the
predecessors are those of the rule computed exactly (tight edges, fewest edges, smallest node); elsewhere
a distance is its own predecessor path's left-to-right sum and within gamma_{H-1} S of the true one; a
negative cycle reachable from the source is a NegativeCycleError for every method and nothing else is;
the blocked Floyd-Warshall is within the same contract (its own path for p-hat)."""

import functools
import math

import numpy as np
import pytest

import stardust as sd

from . import graphs as G
from . import refdata
from .forms import csr

GR = sd.graph
REF = refdata.ref("paths")
NONE = refdata.NONE
READINGS = {
    "directed": (True, False),
    "unweighted": (True, True),
    "undirected": (False, False),
    "undirected_unweighted": (False, True),
}


@functools.cache
def graph(gid):
    return csr(G.get(gid))


@functools.cache
def weights(gid, directed):
    """{(u, v): weight} of the reading, self loops left out."""
    g = G.get(gid)
    w = {(i, j): float(x) for i, j, x in g.edges()} if directed else G.undirected(g)
    return {k: float(v) for k, v in w.items() if k[0] != k[1]}


def points(method):
    m = REF.methods[method]
    for j, k in enumerate(m.index):
        yield (
            str(REF.inputs["graph"][k]),
            int(REF.inputs["source"][k]),
            m.hi[j],
            m.lo[j],
            {f: m.fields[f][j] for f in m.fields},
        )


def flags(name):
    m = REF.methods[name]
    return {(str(REF.inputs["graph"][k]), int(REF.inputs["source"][k])): bool(m.hi[j]) for j, k in enumerate(m.index)}


NEGCYCLE = {"directed": flags("negcycle"), "undirected": flags("negcycle_undirected")}


def has_negative(gid, directed):
    return any(w < 0 for w in weights(gid, directed).values())


def path_of(pred, s, v):
    """The nodes s .. v the predecessor row draws (None when it does not reach s)."""
    out = [v]
    while out[-1] != s:
        p = int(pred[out[-1]])
        if p == NONE or len(out) > pred.size:
            return None
        out.append(p)
    return out[::-1]


def walk(w, nodes, unweighted):
    """(left-to-right sum, edge count, sum of |w|) of a path."""
    total, absum = 0.0, 0.0
    for a, b in zip(nodes, nodes[1:]):
        x = 1.0 if unweighted else w[(a, b)]
        total += x
        absum += abs(x)
    return total, len(nodes) - 1, absum


def run_all(A, s, directed, unweighted, negative):
    """{method: (dist, pred)} of every method that is not FW, as numpy arrays."""
    out = {}
    for m in ("D", "BF", "J", "auto"):
        if m == "D" and negative and not unweighted:
            continue
        r = GR.shortest_path(A, method=m, indices=s, directed=directed, unweighted=unweighted, return_predecessors=True)
        out[m] = (np.asarray(r.dist_matrix), np.asarray(r.predecessors))
    return out


@pytest.mark.parametrize("reading", list(READINGS))
def test_methods_agree_and_match_the_references(reading):
    directed, unweighted = READINGS[reading]
    checked = 0
    for gid, s, hi, lo, f in points(reading):
        A = graph(gid)
        negative = has_negative(gid, directed)
        runs = run_all(A, s, directed, unweighted, negative)
        dist, pred = runs["BF"]
        for m, (d2, p2) in runs.items():
            assert d2.tobytes() == dist.tobytes() and p2.tobytes() == pred.tobytes(), (gid, s, reading, m)
        at_s = np.arange(dist.size) == s
        assert ((pred == NONE) == (np.isinf(dist) | at_s)).all() and dist[s] == 0, (gid, s)
        exact = bool(f["exact_pred"]) or unweighted
        w = weights(gid, directed)
        for v in range(dist.size):
            if math.isinf(hi[v]):
                assert math.isinf(dist[v]), (gid, s, v)
                continue
            if exact:
                assert lo[v] == 0 and dist[v] == hi[v], (gid, s, reading, v)
                continue
            nodes = path_of(pred, s, v)
            total, edges, absum = walk(w, nodes, unweighted)
            assert total == dist[v], (gid, s, v)  # the distance is its own path's left-to-right sum
            H = max(edges, int(f["edges"][v]))
            S = max(absum, float(f["abs_sum"][v]))
            assert refdata.units(float(dist[v]), hi[v], lo[v], H, S) <= 1, (gid, s, reading, v)
        if exact:
            assert pred.tolist() == f["pred"].tolist(), (gid, s, reading)
        checked += 1
    assert checked > 100


@pytest.mark.parametrize("directed", [True, False])
def test_negative_cycles_reachable_from_the_source(directed):
    key = "directed" if directed else "undirected"
    seen = 0
    for (gid, s), cycle in NEGCYCLE[key].items():
        A = graph(gid)
        for m in ("BF", "J", "auto"):
            if cycle:
                with pytest.raises(
                    GR.NegativeCycleError, match=r"^shortest_path\(\): negative cycle through node \d+$"
                ):
                    GR.shortest_path(A, method=m, indices=s, directed=directed)
            else:
                GR.shortest_path(A, method=m, indices=s, directed=directed)
        seen += cycle
    assert seen > 0


def fw_contract(gid, directed, unweighted):
    """The blocked Floyd-Warshall of a graph against every reference row of it: within the contract,
    exact where the sums are."""
    reading = ("directed" if directed else "undirected") + ("_unweighted" if unweighted else "")
    reading = {"directed_unweighted": "unweighted"}.get(reading, reading)
    rows = [(s, hi, lo, f) for g2, s, hi, lo, f in points(reading) if g2 == gid]
    if not rows:
        return 0
    r = GR.floyd_warshall(graph(gid), directed=directed, unweighted=unweighted, return_predecessors=True)
    D, P = np.asarray(r.dist_matrix), np.asarray(r.predecessors)
    w = weights(gid, directed)
    for s, hi, lo, f in rows:
        exact = bool(f["exact_pred"]) or unweighted
        for v in range(D.shape[1]):
            if math.isinf(hi[v]):
                assert math.isinf(D[s, v]) and P[s, v] == NONE, (gid, s, v)
                continue
            if exact:
                assert D[s, v] == hi[v], (gid, s, v)
                continue
            nodes = path_of(P[s], s, v)
            _, edges, absum = walk(w, nodes, unweighted)
            H = max(edges, int(f["edges"][v]))
            S = max(absum, float(f["abs_sum"][v]))
            assert refdata.units(float(D[s, v]), hi[v], lo[v], H, S) <= 1, (gid, s, v)
    return len(rows)


def test_floyd_warshall_within_the_contract():
    graphs = sorted({str(x) for x in REF.inputs["graph"]})
    checked = 0
    for gid in graphs:
        for directed in (True, False):
            key = "directed" if directed else "undirected"
            if any(c for (g2, _), c in NEGCYCLE[key].items() if g2 == gid):
                with pytest.raises(GR.NegativeCycleError, match=r"^floyd_warshall\(\): negative cycle through node"):
                    GR.floyd_warshall(graph(gid), directed=directed)
                continue
            for unweighted in (False, True):
                checked += fw_contract(gid, directed, unweighted)
    assert checked > 100


def test_construct_dist_matrix_is_the_distances_bit_for_bit():
    graphs = sorted({str(x) for x in REF.inputs["graph"]})
    for gid in graphs:
        A = graph(gid)
        for directed in (True, False):
            if has_negative(gid, directed):
                continue
            r = GR.shortest_path(A, directed=directed, return_predecessors=True)
            c = GR.construct_dist_matrix(A, r.predecessors, directed=directed)
            assert np.asarray(c).tobytes() == np.asarray(r.dist_matrix).tobytes(), (gid, directed)
