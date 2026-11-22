"""The reference and comparison data of sd.graph (docs/design/graph.md §7): every file belongs to the
test graphs as they are (digests), the references pass checks of their own (exact optimality of the
distances, the predecessor rule, the enumerations, the PageRank equation), and the recorded scipy
1.18.1 and networkx 3.7 results agree with the references and the definitions and show the behaviours
the design departs from. The recorded files are compared, not the installed libraries, so the tests do
not change with them."""

import collections
import heapq
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from . import graphs as G
from . import refdata

REFS = {f: refdata.ref(f) for f in refdata.FAMILIES}
SCIPY = refdata.data("scipy")
NX = refdata.data("networkx")
NONE = refdata.NONE


def fr(v) -> Fraction:
    return Fraction(int(v)) if isinstance(v, (bool, int, np.integer, np.bool_)) else Fraction(float(v))


def arr(rec):
    """A recorded array as numpy (the probe writes inf and nan as their repr)."""
    vals = [float(x) if isinstance(x, str) else x for x in rec["array"]]
    return np.array(vals, dtype=rec["dtype"]).reshape(rec["shape"])


def adjacency(edges, n, unweighted=False):
    adj = [[] for _ in range(n)]
    for i, j, w in edges:
        if i != j:
            adj[i].append((j, Fraction(1) if unweighted else fr(w)))
    return [sorted(a) for a in adj]


def reading(g, method):
    edges = (
        g.edges()
        if method in ("directed", "unweighted")
        else sorted((i, j, w) for (i, j), w in G.undirected(g).items())
    )
    return adjacency(edges, g.n, method.endswith("unweighted"))


# ---- the files belong to the graphs ----------------------------------------------------------


def test_every_family_is_there_with_its_provenance():
    assert refdata.names() == sorted(refdata.FAMILIES)
    for f, r in REFS.items():
        assert r.meta["arb"].startswith("python-flint 0.9.0"), f
        assert r.meta["tool"] == "tools/graph_ref.py"
        assert r.meta["target_bits"] >= 106
        assert r.meta["n_requested"] == r.points, f
        for m in r.methods.values():
            assert m.dropped.size == 0, f


def test_points_are_the_generators_graphs():
    for f, r in REFS.items():
        for k in range(r.points):
            g = G.get(str(r.inputs["graph"][k]))
            assert r.inputs["digest"][k] == G.graph_digest(g), (f, g.id)


def test_records_are_the_generators_graphs():
    for rec in (SCIPY, NX):
        for key, v in rec.items():
            if key.startswith("graph/"):
                gid = key[len("graph/") :].rsplit("/", 1)[0]
                assert v["digest"] == G.graph_digest(G.get(gid)), key
    assert SCIPY["_provenance"]["version"] == "1.18.1"
    assert NX["_provenance"]["version"] == "3.7"


# ---- shortest paths --------------------------------------------------------------------------


def _paths_points(method):
    r = REFS["paths"]
    m = r.methods[method]
    for j, k in enumerate(m.index):
        g = G.get(str(r.inputs["graph"][k]))
        yield g, int(r.inputs["source"][k]), m.hi[j], m.lo[j], {f: m.fields[f][j] for f in m.fields}


@pytest.mark.parametrize("method", ["directed", "unweighted", "undirected", "undirected_unweighted"])
def test_distances_are_optimal(method):
    """d(s) = 0, no edge improves a distance, and every reachable node has a tight in-edge (exact
    rationals where the graph's sums are exact, 2^-100 relative elsewhere)."""
    for g, s, hi, lo, f in _paths_points(method):
        adj = reading(g, method)
        exact = bool(f["exact_pred"]) or method.endswith("unweighted")
        d = [None if math.isinf(h) else fr(h) + fr(l) for h, l in zip(hi, lo)]
        tol = [Fraction(0) if (exact or x is None) else abs(x) * Fraction(1, 2**100) + Fraction(1, 2**1000) for x in d]
        assert d[s] == 0, (g.id, s)
        for u in range(g.n):
            if d[u] is None:
                continue
            for v, w in adj[u]:
                assert d[v] is not None and d[v] <= d[u] + w + tol[v] + tol[u], (g.id, s, u, v)
        for v in range(g.n):
            if d[v] is None or v == s:
                continue
            assert any(
                d[u] is not None and abs(d[u] + w - d[v]) <= tol[u] + tol[v]
                for u in range(g.n)
                for x, w in adj[u]
                if x == v
            ), (g.id, s, v)


@pytest.mark.parametrize("method", ["directed", "unweighted", "undirected", "undirected_unweighted"])
def test_predecessors_follow_the_rule(method):
    """pred is a tree of tight edges, edge counts drop by one along it, and pred(v) is the smallest
    tight in-neighbour one edge closer (docs/design/graph.md §3.2), on the exact graphs."""
    for g, s, hi, lo, f in _paths_points(method):
        if not (f["exact_pred"] or method.endswith("unweighted")):
            continue
        adj = reading(g, method)
        d = [None if math.isinf(h) else fr(h) + fr(l) for h, l in zip(hi, lo)]
        pred, h = f["pred"], f["edges"]
        assert pred[s] == NONE and h[s] == 0
        for v in range(g.n):
            if d[v] is None:
                assert pred[v] == NONE and h[v] == -1
                continue
            if v == s:
                continue
            cands = [
                u
                for u in range(g.n)
                for x, w in adj[u]
                if x == v and d[u] is not None and d[u] + w == d[v] and h[u] == h[v] - 1 and u != v
            ]
            assert cands and pred[v] == min(cands), (g.id, s, v)


def test_unweighted_distances_are_edge_counts():
    for g, _, hi, lo, f in _paths_points("unweighted"):
        assert np.array_equal(np.where(np.isinf(hi), -1, hi), f["edges"]), g.id
        assert not lo.any()


def test_negative_cycles_are_where_the_graphs_put_them():
    r = REFS["paths"]
    flags = {}
    for name in ("negcycle", "negcycle_undirected"):
        m = r.methods[name]
        for j, k in enumerate(m.index):
            flags[(name, str(r.inputs["graph"][k]), int(r.inputs["source"][k]))] = m.hi[j]
    assert flags[("negcycle", "negative/reach12", 0)] == 1
    assert flags[("negcycle", "negative/reach12", 11)] == 0
    assert all(flags[("negcycle", "negative/unreach12", s)] == (1 if 8 <= s <= 10 else 0) for s in range(12))
    assert flags[("negcycle", "negative/zerocycle10", 0)] == 0  # an exact sum of 0 is no negative cycle
    assert flags[("negcycle", "negative/undirected6", 0)] == 0
    assert flags[("negcycle_undirected", "negative/undirected6", 0)] == 1
    assert flags[("negcycle", "negative/dag30", 0)] == 0


# ---- yen, assignment -------------------------------------------------------------------------


def test_yen_paths_are_sorted_loopless_paths():
    r = REFS["yen"]
    m = r.methods["lengths"]
    for j, k in enumerate(m.index):
        g = G.get(str(r.inputs["graph"][k]))
        s, t = int(r.inputs["source"][k]), int(r.inputs["sink"][k])
        w = {(a, b): fr(x) for a, b, x in g.edges() if a != b}
        flat = list(m.fields["paths"][j])
        paths = []
        while flat:
            cut = flat.index(-1)
            paths.append(tuple(flat[:cut]))
            flat = flat[cut + 1 :]
        assert len(paths) == len(m.hi[j]) == min(refdata.YEN_K, int(m.fields["total"][j]))
        keys = []
        for p, hi, e in zip(paths, m.hi[j], m.fields["edges"][j]):
            assert p[0] == s and p[-1] == t and len(set(p)) == len(p), (g.id, p)
            exact = sum((w[(a, b)] for a, b in zip(p, p[1:])), Fraction(0))
            assert fr(hi) == Fraction(float(exact)) and e == len(p) - 1, (g.id, p)
            keys.append((exact, e, p))
        assert keys == sorted(keys), g.id


def test_yen_diamond():
    r = REFS["yen"]
    m = r.methods["lengths"]
    for j, k in enumerate(m.index):
        if str(r.inputs["graph"][k]) == "diamond/4" and (int(r.inputs["source"][k]), int(r.inputs["sink"][k])) == (
            0,
            3,
        ):
            assert list(m.hi[j]) == [2.0, 2.0] and list(m.fields["paths"][j]) == [0, 1, 3, -1, 0, 2, 3, -1]


def test_assignments_are_optimal():
    r = REFS["assignment"]
    feas = {str(r.inputs["graph"][k]): r.methods["feasible"].hi[j] for j, k in enumerate(r.methods["feasible"].index)}
    assert feas["cost/infeasible4x4"] == 0 and feas["bipartite/nofull5x5"] == 0 and feas["cost/inf5x5"] == 1
    m = r.methods["min"]
    for j, k in enumerate(m.index):
        g = G.get(str(r.inputs["graph"][k]))
        if max(g.shape) > 6:
            continue
        cost = {(a, b): fr(x) for a, b, x in g.edges() if not (isinstance(x, float) and x == math.inf)}
        rows, cols = g.shape
        best = None
        for perm in itertools.permutations(range(max(rows, cols)), min(rows, cols)):
            pairs = [(a, b) if rows <= cols else (b, a) for a, b in enumerate(perm)]
            if all(p in cost for p in pairs):
                c = sum(cost[p] for p in pairs)
                best = c if best is None else min(best, c)
        assert fr(m.hi[j]) == Fraction(float(best)), g.id
    for j, k in enumerate(m.index):
        if str(r.inputs["graph"][k]) == "cost/ties6x6":
            assert m.fields["count"][j] == 720 and list(m.fields["col_ind"][j]) == list(range(6))


# ---- pagerank, betweenness, laplacian ----------------------------------------------------------


def test_pagerank_solves_its_equation():
    r = REFS["pagerank"]
    m = r.methods["x"]
    for j, k in enumerate(m.index):
        g = G.get(str(r.inputs["graph"][k]))
        x = m.hi[j]
        n = g.n
        alpha = float(r.inputs["alpha"][k])
        pers, dang = r.inputs["personalization"][k], r.inputs["dangling"][k]
        v = np.ones(n) / n if pers.size == 0 else pers / pers.sum()
        dd = v if dang.size == 0 else dang / dang.sum()
        edges = g.edges() if r.inputs["directed"][k] else [(a, b, w) for (a, b), w in G.undirected(g).items()]
        P = np.zeros((n, n))
        for a, b, w in edges:
            P[a, b] += 1.0 if r.inputs["unweighted"][k] else float(w)
        out = P.sum(axis=1)
        for a in range(n):
            P[a] = P[a] / out[a] if out[a] > 0 else dd
        assert abs(x.sum() - 1) < 1e-15 and (x >= 0).all(), g.id
        assert np.abs(alpha * P.T @ x + (1 - alpha) * v - x).sum() < 1e-14, g.id


def test_pagerank_agrees_with_networkx():
    r = REFS["pagerank"]
    m = r.methods["x"]
    default = {
        str(r.inputs["graph"][k]): m.hi[j]
        for j, k in enumerate(m.index)
        if r.inputs["personalization"][k].size == 0
        and r.inputs["dangling"][k].size == 0
        and float(r.inputs["alpha"][k]) == 0.85
        and r.inputs["directed"][k]
        and not r.inputs["unweighted"][k]
    }
    for key, v in NX.items():
        if key.endswith("/pagerank"):
            gid = key[len("graph/") : -len("/pagerank")]
            tight = np.array(v["tight"]["value"])
            assert np.abs(tight - default[gid]).sum() < 1e-12, gid
            assert np.abs(np.array(v["default"]["value"]) - default[gid]).sum() > 1e-12 or G.get(gid).n <= 9, gid


def test_betweenness_agrees_with_networkx():
    r = REFS["betweenness"]
    m = r.methods["bc"]
    ref = {}
    for j, k in enumerate(m.index):
        key = (
            str(r.inputs["graph"][k]),
            bool(r.inputs["directed"][k]),
            bool(r.inputs["unweighted"][k]),
            bool(r.inputs["normalized"][k]),
            bool(r.inputs["endpoints"][k]),
        )
        ref[key] = m.hi[j]
    seen = 0
    for key, v in NX.items():
        if not key.endswith("/betweenness"):
            continue
        gid = key[len("graph/") : -len("/betweenness")]
        for opt, out in v.items():
            if opt == "digest":
                continue
            dch, wch, flags = opt.split("/")
            k = (gid, dch == "d", wch == "u", flags[0] == "1", flags[1] == "1")
            if k in ref:
                got = np.array(out["value"])
                assert np.allclose(got, ref[k], rtol=1e-13, atol=0), (key, opt)
                seen += 1
    assert seen >= 40


def test_laplacian_entries():
    r = REFS["laplacian"]
    m = r.methods["L"]
    for j, k in enumerate(m.index):
        g = G.get(str(r.inputs["graph"][k]))
        n = g.n
        L = m.hi[j].reshape(n, n)
        A = np.zeros((n, n))
        for a, b, w in g.edges():
            if a != b:
                A[a, b] += float(w)
        if r.inputs["symmetrized"][k]:
            A = A + A.T
        if not r.inputs["normed"][k]:
            off = ~np.eye(n, dtype=bool)
            assert np.array_equal(L[off], -A[off]), g.id
            deg = A.sum(axis=1 if r.inputs["use_out_degree"][k] else 0)
            assert np.allclose(np.diag(L), deg, rtol=1e-14, atol=0), g.id
        else:
            deg = A.sum(axis=1 if r.inputs["use_out_degree"][k] else 0)
            assert np.array_equal(np.diag(L), (deg != 0).astype(float)), g.id
            if r.inputs["symmetrized"][k]:
                assert np.array_equal(L, L.T), g.id


# ---- the records show what the design departs from -----------------------------------------------


def _bfs(adj, s):
    seen, order, q = {s}, [s], collections.deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                order.append(v)
                q.append(v)
    return order


def _dfs(adj, s):
    seen, order = set(), []

    def visit(u):
        seen.add(u)
        order.append(u)
        for v in adj[u]:
            if v not in seen:
                visit(v)

    visit(s)
    return order


def _neighbours(g, undirected=False):
    adj = [set() for _ in range(g.n)]
    for i, j, _ in g.edges():
        if i != j:
            adj[i].add(j)
            if undirected:
                adj[j].add(i)
    return [sorted(a) for a in adj]


def test_scipy_directed_traversals_are_the_ascending_order():
    """docs/design/graph.md §2.2: directed orders are scipy's; the undirected ones differ."""
    for key, v in SCIPY.items():
        if key.endswith("/traversal"):
            g = G.get(key[len("graph/") : -len("/traversal")])
            adj = _neighbours(g)
            assert v["bfs"] == _bfs(adj, 0), g.id
            assert v["dfs"] == _dfs(adj, 0), g.id
    rec = SCIPY["rule/undirected_bfs"]
    assert rec["G"] == [0, 3, 4, 1, 2] and rec["GT"] == [0, 1, 2, 3, 4]


def test_scipy_component_labels():
    for key, v in SCIPY.items():
        if key.endswith("/components"):
            first = {}
            assert v["weak"] == [first.setdefault(x, len(first)) for x in v["weak"]], key
    assert not SCIPY["rule/strong_labels"]["components/40"]["smallest_node_order"]


def test_scipy_shortest_path_departures():
    assert SCIPY["rule/diamond_predecessors"] == {"D": 2, "BF": 1, "J": 2, "FW": 1}
    for gid in ("er/float100", "er/wide80", "heavytail/120"):
        r = SCIPY[f"rule/sums/{gid}"]
        assert r["D_equals_BF"] and r["D_equals_J"] and r["D_equals_left_to_right"]
        assert r["FW_differs"] > 0 and r["FW_max_rel"] < 1e-15
        assert not r["construct_equals_shortest_path"]
    auto = SCIPY["rule/auto"]
    assert all(v["equals_FW"] == (v["nnz_over_n2"] >= 0.25) for v in auto.values())
    assert SCIPY["rule/negative_cycle_class"][1] == "Exception"
    assert "UserWarning" in SCIPY["rule/negative_dijkstra"]["warnings"]
    assert "value" in SCIPY["rule/unreachable_cycle"]["bellman_ford"]
    assert SCIPY["rule/unreachable_cycle"]["johnson"]["error"] == "NegativeCycleError"
    assert arr(SCIPY["rule/limit"]["one"]["value"]).tolist()[:2] == [0.0, 1.0]
    assert np.isinf(arr(SCIPY["rule/limit"]["nan"]["value"])[1:]).all()
    assert SCIPY["rule/indices_2d"]["value"] == [2, 2, 10]
    assert SCIPY["rule/indices_negative"]["minus_one"] == SCIPY["rule/indices_negative"]["last"]
    assert SCIPY["rule/i_start_negative"]["error"] == "OverflowError"
    assert SCIPY["rule/i_start_out_of_range"]["value"][0]["array"] == []
    assert "ComplexWarning" in SCIPY["rule/complex"]["warnings"]
    assert arr(SCIPY["rule/nan_dijkstra"]["value"]).tolist() == [0.0, 1.0, math.inf, 5.0]
    assert all(SCIPY["rule/undirected_min"].values())


def test_scipy_flow_matching_tree_departures():
    assert SCIPY["rule/flow_past_int32"]["value"]["np_int"] == 5
    assert SCIPY["rule/flow_negative"]["value"]["flow_value"]["np_int"] == 4
    assert SCIPY["rule/flow_float"]["error"] == "ValueError"
    assert SCIPY["rule/flow_dense"]["error"] == "TypeError"
    full = SCIPY["rule/full_matching_zero"]
    assert full["value"][1]["array"] == [1, 2, 3, 4, 0]  # the stored zeros read as no edge
    assert SCIPY["rule/matching_zero"]["value"]["array"] == [0, 1, 2, 3, 4]  # here they are edges
    mst = SCIPY["rule/mst_dense_small"]["value"]
    assert (5, 1) not in set(zip(mst["row"], mst["col"])) and (1, 5) not in set(zip(mst["row"], mst["col"]))
    nan_mst = SCIPY["rule/nan_mst"]["value"]
    assert nan_mst["row"] == [0, 0, 2]  # the NaN edge 1 -> 2 is dropped, the tree takes 0 - 3 instead
    assert SCIPY["rule/dtypes"]["predecessors"] == "int32" and SCIPY["rule/dtypes"]["labels"] == "int32"
    assert SCIPY["rule/laplacian_sparse_type"] == "coo_array"


def test_networkx_structure_is_the_definition():
    """Articulation points, bridges and core numbers of the records against brute-force definitions."""

    def n_components(adj, removed_nodes=(), removed_edge=None):
        alive = [v for v in range(len(adj)) if v not in removed_nodes]
        seen, count = set(), 0
        for s in alive:
            if s in seen:
                continue
            count += 1
            stack = [s]
            seen.add(s)
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v in removed_nodes or {u, v} == removed_edge or v in seen:
                        continue
                    seen.add(v)
                    stack.append(v)
        return count

    for key, v in NX.items():
        if not key.endswith("/structure"):
            continue
        g = G.get(key[len("graph/") : -len("/structure")])
        adj = _neighbours(g, undirected=True)
        base = n_components(adj)
        art = [u for u in range(g.n) if n_components(adj, (u,)) > base - (1 if not adj[u] else 0)]
        assert v["articulation_points"] == art, g.id
        edges = sorted({(min(i, j), max(i, j)) for i in range(g.n) for j in adj[i]})
        assert [tuple(b) for b in v["bridges"]] == [e for e in edges if n_components(adj, (), set(e)) > base], g.id
        core, deg, alive = [0] * g.n, [len(a) for a in adj], set(range(g.n))
        k = 0
        while alive:
            u = min(alive, key=lambda x: (deg[x], x))
            k = max(k, deg[u])
            core[u] = k
            alive.remove(u)
            for w in adj[u]:
                if w in alive:
                    deg[w] -= 1
        assert v["core_number"] == core, g.id
        covered = sorted(tuple(e) for comp in v["biconnected_edges"] for e in comp)
        assert covered == edges, g.id


def test_networkx_topological_order_is_the_lexicographic_one():
    for key, v in NX.items():
        if key.endswith("/topological_sort"):
            g = G.get(key[len("graph/") : -len("/topological_sort")])
            indeg = [0] * g.n
            adj = _neighbours(g)
            for u in range(g.n):
                for w in adj[u]:
                    indeg[w] += 1
            heap = [u for u in range(g.n) if indeg[u] == 0]
            heapq.heapify(heap)
            order = []
            while heap:
                u = heapq.heappop(heap)
                order.append(u)
                for w in adj[u]:
                    indeg[w] -= 1
                    if indeg[w] == 0:
                        heapq.heappush(heap, w)
            assert v["lexicographical"] == order, g.id
    assert NX["rule/topological_selfloop"]["error"] == "NetworkXUnfeasible"
    assert NX["rule/core_number_selfloop"]["error"] == "NetworkXNotImplemented"
