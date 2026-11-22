"""sd.graph against scipy.sparse.csgraph 1.18.1 (and scipy.optimize.linear_sum_assignment) and, for the
algorithms scipy lacks, networkx 3.7 (docs/design/graph.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the
sd.graph names it needs, and cases() keeps a spec once its names are public (the GR1 milestone brings
the traversals and components, GR2 the shortest paths, GR3 the spanning tree, flow, matchings and
assignment, GR4 the ordering, structural rank and laplacian, GR5 the algorithms scipy lacks). The sizes
are those of docs/design/graph.md §5, where the GR0 milestone timed the comparators; a case's size is the
adjacency matrix's shape (bench/README.md, case ids) and the out-degree d goes into its name. scipy's
csgraph runs on one thread; networkx is Python. A sparse result is compared by its canonical CSR arrays.

Data: random directed graphs (each node d out-edges, weights in [0.01, 1.01); tests/graph/graphs.py
random_graph), "symmetric" adding the transpose, made in the setup from fixed seeds as scipy CSR and
sd.sparse.csr_array, outside the timing.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..registry import Case, Check, Ref


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]
    size: str
    setup: Callable[[], dict]
    sd: Callable
    refs: tuple = ()
    quick: bool = False
    note: str = ""
    check: Check | None = None


def _csr_parts(x):
    """A sparse result (sd's csr_array or scipy's) as its canonical CSR arrays, for the smoke check."""
    x = x.tocsr()
    if hasattr(x, "sort_indices"):  # scipy's trees come with unsorted rows
        x = x.copy()
        x.sort_indices()
    return [np.asarray(x.indptr, dtype=np.int64), np.asarray(x.indices, dtype=np.int64), np.asarray(x.data)]


CSR = Check(sd=_csr_parts, ref=lambda label, x: _csr_parts(x))


def _graph(n, d, symmetric=False, int_caps=False, seed=20261123):
    def setup():
        import scipy.sparse as S

        import stardust as sd
        from tests.graph.graphs import random_graph

        indptr, indices, data = random_graph(n, d, seed=seed, symmetric=symmetric)
        if int_caps:
            data = np.random.Generator(np.random.PCG64([seed, n, d, 1])).integers(1, 100, data.size)
        a = S.csr_array((data, indices, indptr), shape=(n, n))
        return {"a": a, "s": sd.sparse.csr_array((data, indices, indptr), shape=(n, n)), "n": n}

    return setup


def _dense(n, seed=20261124):
    def setup():
        import stardust as sd

        w = np.random.Generator(np.random.PCG64([seed, n])).uniform(0.01, 1.01, (n, n))
        np.fill_diagonal(w, 0.0)
        return {"w": w, "sw": sd.asarray(w), "n": n}

    return setup


def _g():
    import stardust as sd

    return sd.graph


def _lsa():
    from scipy.optimize import linear_sum_assignment

    return linear_sum_assignment


def _cs():
    import scipy.sparse.csgraph as C

    return C


def _nx_graph(d, directed=False):
    import networkx as nx

    return nx.from_scipy_sparse_array(d["a"], create_using=nx.DiGraph if directed else nx.Graph)


N5 = "(100000, 100000)"  # the adjacency matrix's shape; the out-degree d goes into the case name


def _dag(n, d, seed=20261125):
    """random_graph's graph keeping the edges that go forward in a random node order: a DAG of about d/2
    out-edges per node whose topological order is not the node order."""

    def setup():
        import scipy.sparse as S

        import stardust as sd
        from tests.graph.graphs import random_graph

        indptr, indices, data = random_graph(n, d, seed=seed)
        rank = np.random.Generator(np.random.PCG64([seed, n])).permutation(n)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        keep = rank[rows] < rank[indices]
        a = S.csr_array((data[keep], (rows[keep], indices[keep])), shape=(n, n))
        return {"a": a, "s": sd.sparse.csr_array(a), "n": n}

    return setup


def _sparse_dense(n, seed=20261126):
    """A dense (n, n) matrix with about half its entries 0 (no edge) and its csr_array."""

    def setup():
        import scipy.sparse.csgraph as C

        import stardust as sd

        w = np.random.Generator(np.random.PCG64([seed, n])).uniform(0.0, 1.0, (n, n))
        w[w < 0.5] = 0.0
        return {"w": w, "sw": sd.asarray(w), "a": C.csgraph_from_dense(w), "s": sd.graph.csgraph_from_dense(w)}

    return setup


def _gr1():
    out = []
    for conn in ("weak", "strong"):
        out.append(
            Spec(
                "GR1",
                "traversal",
                f"connected_components {conn} d=10",
                ("connected_components",),
                N5,
                _graph(10**5, 10),
                lambda d, c=conn: lambda: _g().connected_components(d["s"], connection=c),
                (Ref("scipy", lambda d, c=conn: lambda: _cs().connected_components(d["a"], connection=c)),),
                quick=conn == "weak",
            )
        )
    out.append(
        Spec(
            "GR1",
            "traversal",
            "connected_components undirected d=10",
            ("connected_components",),
            N5,
            _graph(10**5, 10),
            lambda d: lambda: _g().connected_components(d["s"], directed=False),
            (Ref("scipy", lambda d: lambda: _cs().connected_components(d["a"], directed=False)),),
        )
    )
    for name in ("breadth_first_order", "depth_first_order", "breadth_first_tree", "depth_first_tree"):
        out.append(
            Spec(
                "GR1",
                "traversal",
                f"{name} d=10",
                (name,),
                N5,
                _graph(10**5, 10),
                lambda d, f=name: lambda: getattr(_g(), f)(d["s"], 0),
                (Ref("scipy", lambda d, f=name: lambda: getattr(_cs(), f)(d["a"], 0)),),
                check=CSR if name.endswith("tree") else None,
            )
        )
    for name in ("breadth_first_order", "depth_first_order"):
        out.append(
            Spec(
                "GR1",
                "traversal",
                f"{name} undirected d=10",
                (name,),
                N5,
                _graph(10**5, 10),
                lambda d, f=name: lambda: getattr(_g(), f)(d["s"], 0, directed=False),
                (
                    Ref(
                        "scipy",
                        lambda d, f=name: lambda: getattr(_cs(), f)(d["a"], 0, directed=False),
                        differs="scipy looks at the out-neighbours first; sd at the merged ascending ones (GR13)",
                    ),
                ),
            )
        )
    out.append(
        Spec(
            "GR1",
            "traversal",
            "breadth_first_order undirected symmetric d=10",
            ("breadth_first_order",),
            N5,
            _graph(10**5, 10, symmetric=True),
            lambda d: lambda: _g().breadth_first_order(d["s"], 0, directed=False),
            (Ref("scipy", lambda d: lambda: _cs().breadth_first_order(d["a"], 0, directed=False)),),
        )
    )
    out.append(
        Spec(
            "GR1",
            "traversal",
            "topological_sort DAG d=10",
            ("topological_sort",),
            N5,
            _dag(10**5, 10),
            lambda d: lambda: _g().topological_sort(d["s"]),
            (Ref("networkx", _nx("lexicographical_topological_sort", True)),),
            note="a DAG of about 5 out-edges per node; networkx's lexicographical_topological_sort gives the same order",
        )
    )
    dense = _sparse_dense(2000)
    out += [
        Spec(
            "GR1",
            "traversal",
            "csgraph_from_dense",
            ("csgraph_from_dense",),
            "(2000, 2000)",
            dense,
            lambda d: lambda: _g().csgraph_from_dense(d["sw"]),
            (Ref("scipy", lambda d: lambda: _cs().csgraph_from_dense(d["w"])),),
            check=CSR,
        ),
        Spec(
            "GR1",
            "traversal",
            "csgraph_to_dense",
            ("csgraph_to_dense",),
            "(2000, 2000)",
            dense,
            lambda d: lambda: _g().csgraph_to_dense(d["s"]),
            (Ref("scipy", lambda d: lambda: _cs().csgraph_to_dense(d["a"])),),
        ),
        Spec(
            "GR1",
            "traversal",
            "connected_components dense",
            ("connected_components",),
            "(2000, 2000)",
            dense,
            lambda d: lambda: _g().connected_components(d["sw"]),
            (Ref("scipy", lambda d: lambda: _cs().connected_components(d["w"])),),
        ),
    ]
    return out


def _gr2():
    out = [
        Spec(
            "GR2",
            "paths",
            "dijkstra 1 source d=10",
            ("dijkstra",),
            "(100000, 100000)",
            _graph(10**5, 10),
            lambda d: lambda: _g().dijkstra(d["s"], indices=0),
            (Ref("scipy", lambda d: lambda: _cs().dijkstra(d["a"], indices=0)),),
            quick=True,
        ),
        Spec(
            "GR2",
            "paths",
            "dijkstra 1 source unweighted d=10",
            ("dijkstra",),
            "(100000, 100000)",
            _graph(10**5, 10),
            lambda d: lambda: _g().dijkstra(d["s"], indices=0, unweighted=True),
            (Ref("scipy", lambda d: lambda: _cs().dijkstra(d["a"], indices=0, unweighted=True)),),
        ),
        Spec(
            "GR2",
            "paths",
            "dijkstra 100 sources d=10",
            ("dijkstra",),
            "(100000, 100000)",
            _graph(10**5, 10),
            lambda d: lambda: _g().dijkstra(d["s"], indices=np.arange(100)),
            (Ref("scipy", lambda d: lambda: _cs().dijkstra(d["a"], indices=np.arange(100))),),
        ),
        Spec(
            "GR2",
            "paths",
            "dijkstra 100 sources min_only d=10",
            ("dijkstra",),
            "(100000, 100000)",
            _graph(10**5, 10),
            lambda d: lambda: _g().dijkstra(d["s"], indices=np.arange(100), min_only=True),
            (Ref("scipy", lambda d: lambda: _cs().dijkstra(d["a"], indices=np.arange(100), min_only=True)),),
        ),
        Spec(
            "GR2",
            "paths",
            "bellman_ford 1 source d=10",
            ("bellman_ford",),
            "(10000, 10000)",
            _graph(10**4, 10),
            lambda d: lambda: _g().bellman_ford(d["s"], indices=0),
            (Ref("scipy", lambda d: lambda: _cs().bellman_ford(d["a"], indices=0)),),
        ),
        Spec(
            "GR2",
            "paths",
            "johnson all pairs d=10",
            ("johnson",),
            "(2000, 2000)",
            _graph(2000, 10),
            lambda d: lambda: _g().johnson(d["s"]),
            (Ref("scipy", lambda d: lambda: _cs().johnson(d["a"])),),
        ),
        Spec(
            "GR2",
            "paths",
            "dijkstra all pairs d=10",
            ("dijkstra",),
            "(2000, 2000)",
            _graph(2000, 10),
            lambda d: lambda: _g().dijkstra(d["s"]),
            (Ref("scipy", lambda d: lambda: _cs().dijkstra(d["a"])),),
        ),
        Spec(
            "GR2",
            "paths",
            "floyd_warshall dense",
            ("floyd_warshall",),
            "(1000, 1000)",
            _dense(1000),
            lambda d: lambda: _g().floyd_warshall(d["sw"]),
            (Ref("scipy", lambda d: lambda: _cs().floyd_warshall(d["w"])),),
        ),
        Spec(
            "GR2",
            "paths",
            "shortest_path auto dense",
            ("shortest_path",),
            "(1000, 1000)",
            _dense(1000),
            lambda d: lambda: _g().shortest_path(d["sw"]),
            (
                Ref(
                    "scipy",
                    lambda d: lambda: _cs().shortest_path(d["w"]),
                    differs="scipy's auto takes FW here; sd's takes the dense Dijkstra (decision GR26)",
                ),
            ),
        ),
        Spec(
            "GR2",
            "paths",
            "yen K = 10 d=10",
            ("yen",),
            "(10000, 10000)",
            _graph(10**4, 10),
            lambda d: lambda: _g().yen(d["s"], 0, 9999, 10),
            (Ref("scipy", lambda d: lambda: _cs().yen(d["a"], 0, 9999, 10)),),
        ),
    ]
    return out


def _gr3():
    def lsa_setup():
        import stardust as sd

        c = np.random.Generator(np.random.PCG64(4)).uniform(0, 1, (1000, 1000))
        return {"c": c, "sc": sd.asarray(c)}

    return [
        Spec(
            "GR3",
            "tree_flow",
            "minimum_spanning_tree symmetric d=5",
            ("minimum_spanning_tree",),
            "(100000, 100000)",
            _graph(10**5, 5, symmetric=True),
            lambda d: lambda: _g().minimum_spanning_tree(d["s"]),
            (Ref("scipy", lambda d: lambda: _cs().minimum_spanning_tree(d["a"])),),
            quick=True,
        ),
        Spec(
            "GR3",
            "tree_flow",
            "maximum_flow dinic d=10",
            ("maximum_flow",),
            "(10000, 10000)",
            _graph(10**4, 10, int_caps=True),
            lambda d: lambda: _g().maximum_flow(d["s"], 0, 9999),
            (Ref("scipy", lambda d: lambda: _cs().maximum_flow(d["a"], 0, 9999)),),
        ),
        Spec(
            "GR3",
            "tree_flow",
            "maximum_flow edmonds_karp d=10",
            ("maximum_flow",),
            "(10000, 10000)",
            _graph(10**4, 10, int_caps=True),
            lambda d: lambda: _g().maximum_flow(d["s"], 0, 9999, method="edmonds_karp"),
            (Ref("scipy", lambda d: lambda: _cs().maximum_flow(d["a"], 0, 9999, method="edmonds_karp")),),
        ),
        Spec(
            "GR3",
            "tree_flow",
            "maximum_bipartite_matching d=10",
            ("maximum_bipartite_matching",),
            "(100000, 100000)",
            _graph(10**5, 10),
            lambda d: lambda: _g().maximum_bipartite_matching(d["s"]),
            (Ref("scipy", lambda d: lambda: _cs().maximum_bipartite_matching(d["a"])),),
        ),
        Spec(
            "GR3",
            "tree_flow",
            "linear_sum_assignment",
            ("linear_sum_assignment",),
            "(1000, 1000)",
            lsa_setup,
            lambda d: lambda: _g().linear_sum_assignment(d["sc"]),
            (Ref("scipy", lambda d: lambda: _lsa()(d["c"])),),
        ),
    ]


def _gr4():
    sym = _graph(10**5, 10, symmetric=True)
    return [
        Spec(
            "GR4",
            "ordering",
            "reverse_cuthill_mckee symmetric d=10",
            ("reverse_cuthill_mckee",),
            "(100000, 100000)",
            sym,
            lambda d: lambda: _g().reverse_cuthill_mckee(d["s"], symmetric_mode=True),
            (Ref("scipy", lambda d: lambda: _cs().reverse_cuthill_mckee(d["a"], symmetric_mode=True)),),
        ),
        Spec(
            "GR4",
            "ordering",
            "structural_rank d=10",
            ("structural_rank",),
            "(100000, 100000)",
            _graph(10**5, 10),
            lambda d: lambda: _g().structural_rank(d["s"]),
            (Ref("scipy", lambda d: lambda: _cs().structural_rank(d["a"])),),
        ),
        Spec(
            "GR4",
            "ordering",
            "laplacian symmetric d=10",
            ("laplacian",),
            "(100000, 100000)",
            sym,
            lambda d: lambda: _g().laplacian(d["s"]),
            (Ref("scipy", lambda d: lambda: _cs().laplacian(d["a"]), differs="scipy returns COO, sd CSR"),),
        ),
        Spec(
            "GR4",
            "ordering",
            "laplacian normed symmetric d=10",
            ("laplacian",),
            "(100000, 100000)",
            sym,
            lambda d: lambda: _g().laplacian(d["s"], normed=True),
            (
                Ref(
                    "scipy", lambda d: lambda: _cs().laplacian(d["a"], normed=True), differs="scipy returns COO, sd CSR"
                ),
            ),
        ),
    ]


def _nx(fname, directed=False, **kw):
    """A networkx Ref factory: the graph is converted in the setup's dict once, outside the timing."""

    def make(d):
        import networkx as nx

        G = _nx_graph(d, directed)
        f = getattr(nx, fname)
        return lambda: list(f(G, **kw)) if fname != "core_number" else f(G, **kw)

    return make


def _gr5():
    sym = _graph(10**4, 5, symmetric=True)
    out = []
    for name in ("articulation_points", "bridges", "biconnected_components", "core_number"):
        nxname = "biconnected_component_edges" if name == "biconnected_components" else name
        out.append(
            Spec(
                "GR5",
                "structure",
                f"{name} symmetric d=5",
                (name,),
                "(10000, 10000)",
                sym,
                lambda d, f=name: lambda: getattr(_g(), f)(d["s"]),
                (Ref("networkx", _nx(nxname)),),
                note="also against connected_components weak of the same graph (docs/design/graph.md §5)",
            )
        )
    out.append(
        Spec(
            "GR5",
            "centrality",
            "pagerank d=5",
            ("pagerank",),
            "(10000, 10000)",
            _graph(10**4, 5),
            lambda d: lambda: _g().pagerank(d["s"]),
            (
                Ref(
                    "networkx",
                    _nx("pagerank", True),
                    differs="networkx stops at tol=1e-6 by successive differences; sd at the proven 1e-12 bound",
                ),
            ),
        )
    )
    out.append(
        Spec(
            "GR5",
            "centrality",
            "betweenness_centrality unweighted d=5",
            ("betweenness_centrality",),
            "(1000, 1000)",
            _graph(1000, 5),
            lambda d: lambda: _g().betweenness_centrality(d["s"], unweighted=True),
            (Ref("networkx", _nx("betweenness_centrality", True)),),
        )
    )
    return out


SPECS = _gr1() + _gr2() + _gr3() + _gr4() + _gr5()


def _public(names) -> bool:
    import stardust as sd

    return all(n in sd.graph.__all__ for n in names)


def cases() -> list[Case]:
    return [
        Case(
            area="graph",
            group=s.group,
            name=s.name,
            size=s.size,
            setup=s.setup,
            sd=s.sd,
            refs=s.refs,
            quick=s.quick,
            note=s.note,
            **({"check": s.check} if s.check else {}),
        )
        for s in SPECS
        if _public(s.needs)
    ]


TITLES = {
    "traversal": "순회와 연결 요소",
    "paths": "최단 경로",
    "tree_flow": "신장 트리, 흐름, 매칭, 배정",
    "ordering": "순서, 구조적 랭크, 라플라시안",
    "structure": "단절점, 다리, 이중 연결 요소, 코어 수",
    "centrality": "PageRank와 매개 중심성",
}
