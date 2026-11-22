"""Probes of sd.graph's plan and of the libraries it is compared with (docs/design/graph.md §3.2, §5,
§7.4). The records the tests read go to tests/graph/data/; the timings and the FW measurement go to
the terminal and to build/logs/graph_probe/<stamp>.log, and docs/design/graph.md §3.2 and §5 quote
them.

    .venv/bin/python tools/graph_probe.py                   # every section
    .venv/bin/python tools/graph_probe.py scipy networkx    # only these sections

Sections (positional; default all):
  fw        the two candidates of the dense all-pairs path (tools/graph_probe_fw.cpp): per-source
            dense Dijkstra against blocked Floyd-Warshall, n = 256 .. 2048, 1 and 10 threads, densities
            1 and 1/4 (decision GR25), and scipy's floyd_warshall at the same sizes
  scipy     scipy 1.18.1's behaviours that docs/design/graph.md departs from or relies on (inputs, stored
            0/NaN/inf, negative weights and cycles, ties, auto, limit, indices, min_only, traversal
            orders, labels, the spanning tree's edges, the int32 capacities, matchings, laplacian, yen,
            the conversion tools) and its results on the test graphs -> tests/graph/data/scipy.json
  networkx  networkx 3.7's results for the seven algorithms scipy lacks (topological order,
            articulation points, bridges, biconnected components, core numbers, PageRank,
            betweenness) on the test graphs and its edge behaviours -> tests/graph/data/networkx.json
  time      the scipy timings of §5 (best of three, one call each over a second; scipy's csgraph is
            one thread) and a few of networkx's

The C++ program is compiled into build/graph_probe/ with sd's flags (-O3 -mcpu=native
-ffp-contract=off) under the shared build lock (build/.agent-build-lock). Every child process runs
under a time limit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import cli  # noqa: E402

BUILD = ROOT / "build" / "graph_probe"
DATA = ROOT / "tests" / "graph" / "data"
LOCK = ROOT / "build" / ".agent-build-lock"
FLAGS = ["-std=c++20", "-O3", "-mcpu=native", "-ffp-contract=off"]
SRC = ROOT / "tools" / "graph_probe_fw.cpp"
SECTIONS = ("fw", "scipy", "networkx", "time")
CHILD_SECONDS = 900
NONE = -9999


def G():
    from tests.graph import graphs

    return graphs


class Log:
    def __init__(self):
        self.path = cli.log_file("graph_probe")
        print(f"log: {self.path.relative_to(ROOT)}", flush=True)

    def __call__(self, text=""):
        print(text, flush=True)
        with self.path.open("a") as f:
            f.write(text + "\n")


def load(arg=""):
    return f"load average {os.getloadavg()[0]:.1f}{' ' + arg if arg else ''}"


# ---------------------------------------------------------------- fw


def build() -> pathlib.Path:
    exe = BUILD / "fw"
    if exe.exists() and exe.stat().st_mtime >= SRC.stat().st_mtime:
        return exe
    BUILD.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            LOCK.mkdir()
            break
        except FileExistsError:
            time.sleep(5)
    try:
        subprocess.run(["clang++", *FLAGS, "-o", str(exe), str(SRC)], check=True, timeout=600)
    finally:
        LOCK.rmdir()
    return exe


def fw(log):
    import scipy.sparse.csgraph as C

    exe = build()
    log(f"## fw ({load('at start')})")
    for n in (256, 512, 1024, 2048):
        for threads in (1, 10):
            for density in (1.0, 0.25):
                if density != 1.0 and n not in (1024,):
                    continue
                p = subprocess.run(
                    [str(exe), str(n), str(threads), str(density), "1", "3"],
                    capture_output=True,
                    text=True,
                    timeout=CHILD_SECONDS,
                )
                log("  " + (p.stdout.strip() or p.stderr.strip()))
    log("scipy floyd_warshall (one thread, best of three; the same graphs' weights in [0.01, 1.01), complete):")
    for n in (256, 512, 1024):
        g = np.random.Generator(np.random.PCG64([1, n]))
        W = g.uniform(0.01, 1.01, (n, n))
        np.fill_diagonal(W, 0.0)
        log(f"  n = {n:5d}: {best(lambda W=W: C.floyd_warshall(W)):9.2f} ms")


# ---------------------------------------------------------------- records


def plain(v):
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "f":
            return {
                "array": [float(x) if math.isfinite(x) else repr(float(x)) for x in v.ravel()],
                "dtype": str(v.dtype),
                "shape": list(v.shape),
            }
        return {"array": v.ravel().tolist(), "dtype": str(v.dtype), "shape": list(v.shape)}
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(v) else repr(float(v))
    if isinstance(v, np.integer):
        return {"np_int": int(v), "dtype": str(v.dtype)}
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if hasattr(v, "toarray") and hasattr(v, "nnz"):
        c = v.tocoo()
        order = np.lexsort((c.col, c.row))
        return {
            "sparse": type(v).__name__,
            "shape": list(v.shape),
            "row": c.row[order].tolist(),
            "col": c.col[order].tolist(),
            "data": plain(np.asarray(c.data[order], dtype=np.float64)),
        }
    if hasattr(v, "flow_value"):
        return {"flow_value": plain(v.flow_value), "flow": plain(v.flow)}
    return {"type": type(v).__name__, "repr": repr(v)[:200]}


def outcome(fn):
    """{"value": ..., "warnings": [...]} or {"error": type, "msg": message} of fn()."""
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = fn()
    except Exception as e:  # noqa: BLE001 -- recording what the library raises
        return {"error": type(e).__name__, "msg": str(e)[:300], "mro": [c.__name__ for c in type(e).__mro__]}
    out = {"value": plain(v)}
    if w:
        out["warnings"] = sorted({x.category.__name__ for x in w})
    return out


def write_records(name, meta, records):
    DATA.mkdir(parents=True, exist_ok=True)
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in records.items()]
    (DATA / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    return f"{len(records)} records -> {(DATA / name).relative_to(ROOT)}"


def left_to_right(W, pred_row, s):
    """The left-to-right sums along the predecessor tree of one source (W dense with inf = no edge)."""
    n = len(pred_row)
    out = np.full(n, np.inf)
    for v in range(n):
        chain, x = [], v
        while x != s and pred_row[x] != NONE:
            chain.append(x)
            x = pred_row[x]
        if x != s:
            continue
        d = 0.0
        for a, b in zip([s] + chain[::-1][:-1], chain[::-1]):
            d = d + W[a, b]
        out[v] = d
    return out


def scipy_records(log):
    import scipy
    import scipy.optimize as O
    import scipy.sparse as S
    import scipy.sparse.csgraph as C

    M = G()
    rec = {}

    def sp(gid):
        return M.get(gid).scipy()

    # inputs
    cplx = S.csr_array(np.array([[0, 1 + 1j, 0], [0, 0, 2 + 0j], [0, 0, 0]]))
    rec["rule/complex"] = outcome(lambda: C.dijkstra(cplx, indices=0))
    rec["rule/nan_dijkstra"] = outcome(lambda: C.dijkstra(sp("nonfinite/nan"), indices=0))
    rec["rule/nan_mst"] = outcome(lambda: C.minimum_spanning_tree(sp("nonfinite/nan")))
    rec["rule/inf_mst"] = outcome(lambda: C.minimum_spanning_tree(sp("nonfinite/inf")))
    rec["rule/neginf_bellman_ford"] = outcome(lambda: C.bellman_ford(sp("nonfinite/neginf"), indices=0))
    p10 = sp("path/10")
    rec["rule/indices_float"] = {
        "half": outcome(lambda: C.dijkstra(p10, indices=1.5)),
        "one": outcome(lambda: C.dijkstra(p10, indices=1)),
    }
    rec["rule/indices_negative"] = {
        "minus_one": outcome(lambda: C.dijkstra(p10, indices=-1)),
        "last": outcome(lambda: C.dijkstra(p10, indices=9)),
    }
    rec["rule/i_start_out_of_range"] = outcome(lambda: C.breadth_first_order(p10, 10))
    rec["rule/i_start_negative"] = outcome(lambda: C.breadth_first_order(p10, -1))
    rec["rule/indices_2d"] = outcome(lambda: C.dijkstra(p10, indices=[[0, 1], [2, 3]]).shape)
    rec["rule/positional_option"] = outcome(lambda: C.connected_components(p10, False))
    rec["rule/dtypes"] = {
        "predecessors": str(C.dijkstra(p10, indices=0, return_predecessors=True)[1].dtype),
        "labels": str(C.connected_components(p10)[1].dtype),
        "n_components": type(C.connected_components(p10)[0]).__name__,
        "order": str(C.breadth_first_order(p10, 0)[0].dtype),
        "rcm": str(C.reverse_cuthill_mckee(sp("star/out4")).dtype),
        "structural_rank": type(C.structural_rank(p10)).__name__,
        "matching": str(C.maximum_bipartite_matching(sp("bipartite/8x8")).dtype),
    }
    # the undirected rule: every method takes the smaller weight of the two directions
    tw = M.get("twoway/12")
    U = np.full((12, 12), np.inf)
    for (i, j), w in M.undirected(tw).items():
        U[i, j] = w
    ref = C.dijkstra(S.csr_array(np.where(np.isinf(U), 0, U)), directed=True)
    rec["rule/undirected_min"] = {
        m: bool(np.array_equal(C.shortest_path(tw.scipy(), method=m, directed=False), ref))
        for m in ("D", "BF", "J", "FW")
    }
    # labels: strong components numbered in the algorithm's order
    strong = {}
    for gid in ("components/40", "er/int60", "cycle/12", "dag/30", "selfloop/10", "negative/unreach12"):
        n_c, lab = C.connected_components(sp(gid), connection="strong")
        first = {}
        canon = [first.setdefault(int(x), len(first)) for x in lab]
        strong[gid] = {"labels": lab.tolist(), "smallest_node_order": canon == lab.tolist()}
    weak = {}
    for gid in ("components/40", "er/int60", "dag/30"):
        n_c, lab = C.connected_components(sp(gid), connection="weak")
        first = {}
        canon = [first.setdefault(int(x), len(first)) for x in lab]
        weak[gid] = {"labels": lab.tolist(), "smallest_node_order": canon == lab.tolist()}
    rec["rule/strong_labels"] = strong
    rec["rule/weak_labels"] = weak
    # traversal: undirected order, directed order on a grid
    oneway = S.csr_array((np.ones(4), ([0, 0, 1, 2], [3, 4, 0, 0])), shape=(5, 5))
    rec["rule/undirected_bfs"] = {
        "G": C.breadth_first_order(oneway, 0, directed=False)[0].tolist(),
        "GT": C.breadth_first_order(oneway.T.tocsr(), 0, directed=False)[0].tolist(),
    }
    g5 = sp("grid2d/5x5")
    rec["rule/grid_traversal"] = {
        "bfs": C.breadth_first_order(g5, 0)[0].tolist(),
        "dfs": C.depth_first_order(g5, 0)[0].tolist(),
        "bfs_pred": C.breadth_first_order(g5, 0)[1].tolist(),
        "dfs_pred": C.depth_first_order(g5, 0)[1].tolist(),
    }
    # predecessor ties per method
    dia = sp("diamond/4")
    rec["rule/diamond_predecessors"] = {
        m: int(C.shortest_path(dia, method=m, return_predecessors=True)[1][0, 3]) for m in ("D", "BF", "J", "FW")
    }
    # distances: D, BF, J agree with the left-to-right sums; FW's last bits differ
    for gid in ("er/float100", "er/wide80", "heavytail/120"):
        g = M.get(gid)
        A = g.scipy()
        W = g.dense(np.inf)
        dist = {m: C.shortest_path(A, method=m) for m in ("D", "BF", "J", "FW")}
        dD, pD = C.shortest_path(A, method="D", return_predecessors=True)
        ltr = np.array([left_to_right(W, pD[s], s) for s in range(g.n)])
        fin = np.isfinite(dist["D"])
        rel = np.abs(dist["FW"][fin] - dist["D"][fin]) / np.where(dist["D"][fin] > 0, dist["D"][fin], 1)
        rec[f"rule/sums/{gid}"] = {
            "D_equals_BF": bool(np.array_equal(dist["D"], dist["BF"])),
            "D_equals_J": bool(np.array_equal(dist["D"], dist["J"])),
            "D_equals_left_to_right": bool(np.array_equal(dist["D"], ltr)),
            "FW_differs": int(np.sum(dist["FW"] != dist["D"])),
            "FW_max_rel": float(rel.max()),
            "construct_equals_shortest_path": bool(np.array_equal(C.construct_dist_matrix(A, pD), dD)),
            "construct_differs": int(np.sum(C.construct_dist_matrix(A, pD) != dD)),
        }
    # auto: whose bits it gives, by density
    auto = {}
    for density in (0.09, 0.2, 0.24, 0.26, 0.3, 0.49):
        g = np.random.Generator(np.random.PCG64([7, int(density * 100)]))
        n = 200
        mask = g.random((n, n)) < density
        np.fill_diagonal(mask, False)
        A = S.csr_array(np.where(mask, g.uniform(0.01, 1.01, (n, n)), 0.0))
        a = C.shortest_path(A, method="auto")
        auto[str(density)] = {
            "nnz_over_n2": A.nnz / n**2,
            "equals_FW": bool(np.array_equal(a, C.shortest_path(A, method="FW"))),
            "equals_D": bool(np.array_equal(a, C.shortest_path(A, method="D"))),
        }
    rec["rule/auto"] = auto
    # limit
    unit = S.csr_array((np.ones(4), ([0, 1, 2, 3], [1, 2, 3, 4])), shape=(5, 5))
    rec["rule/limit"] = {
        "one": outcome(lambda: C.dijkstra(unit, indices=0, limit=1)),
        "nan": outcome(lambda: C.dijkstra(unit, indices=0, limit=np.nan)),
        "negative": outcome(lambda: C.dijkstra(unit, indices=0, limit=-1)),
    }
    rec["rule/min_only"] = outcome(lambda: C.dijkstra(g5, indices=[0, 24], min_only=True, return_predecessors=True))
    rec["rule/min_only_all"] = outcome(lambda: C.dijkstra(g5, min_only=True))
    rec["rule/negative_dijkstra"] = outcome(lambda: C.dijkstra(sp("negative/dag30"), indices=0))
    rec["rule/negative_cycle_class"] = [c.__name__ for c in C.NegativeCycleError.__mro__]
    un = sp("negative/unreach12")
    rec["rule/unreachable_cycle"] = {
        "bellman_ford": outcome(lambda: C.bellman_ford(un, indices=0)),
        "johnson": outcome(lambda: C.johnson(un, indices=0)),
        "bellman_ford_all": outcome(lambda: C.bellman_ford(un)),
    }
    rec["rule/undirected_negative_edge"] = outcome(lambda: C.bellman_ford(sp("negative/undirected6"), directed=False))
    rec["rule/zero_cycle"] = outcome(
        lambda: C.bellman_ford(sp("negative/zerocycle10"), indices=0, return_predecessors=True)
    )
    # minimum spanning tree
    ds = M.get("dense/special6").matrix
    rec["rule/mst_dense_small"] = outcome(lambda: C.minimum_spanning_tree(np.where(np.isfinite(ds), ds, 0)))
    rec["rule/mst_ties"] = outcome(lambda: C.minimum_spanning_tree(sp("grid2d/3x3")))
    # flow
    rec["rule/flow_float"] = outcome(lambda: C.maximum_flow(S.csr_array(np.array([[0, 1.0], [0, 0]])), 0, 1))
    rec["rule/flow_negative"] = outcome(
        lambda: C.maximum_flow(S.csr_array(np.array([[0, 5, -3], [0, 0, 4], [0, 0, 0]])), 0, 2)
    )
    rec["rule/flow_past_int32"] = outcome(lambda: C.maximum_flow(sp("flow/past_int32"), 0, 3).flow_value)
    rec["rule/flow_dense"] = outcome(lambda: C.maximum_flow(np.array([[0, 1], [0, 0]]), 0, 1))
    rec["rule/flow_same_node"] = outcome(lambda: C.maximum_flow(sp("flow/int20"), 0, 0))
    # matchings and assignment
    rec["rule/matching_zero"] = outcome(lambda: C.maximum_bipartite_matching(sp("bipartite/zero5x5")))
    rec["rule/full_matching_zero"] = outcome(lambda: C.min_weight_full_bipartite_matching(sp("bipartite/zero5x5")))
    bn = M.get("bipartite/8x8")
    bnan = S.csr_array((np.where(np.arange(bn.nnz) == 3, np.nan, bn.data), bn.indices, bn.indptr), shape=bn.shape)
    rec["rule/full_matching_nan"] = outcome(lambda: C.min_weight_full_bipartite_matching(bnan))
    rec["rule/lsa_nan"] = outcome(lambda: O.linear_sum_assignment(np.array([[1.0, np.nan], [2.0, 3.0]])))
    rec["rule/lsa_infeasible"] = outcome(lambda: O.linear_sum_assignment(M.get("cost/infeasible4x4").matrix))
    # ordering, rank, laplacian
    path4 = S.csr_array((np.ones(6), ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])), shape=(4, 4))
    rec["rule/rcm"] = {
        "star": C.reverse_cuthill_mckee(sp("star/out4")).tolist(),
        "path": C.reverse_cuthill_mckee(path4).tolist(),
        "dense": outcome(lambda: C.reverse_cuthill_mckee(np.eye(3))),
    }
    rec["rule/laplacian_sparse_type"] = type(C.laplacian(sp("er/symfloat50"))).__name__
    lneg = C.laplacian(sp("negative/dag30").toarray(), normed=True)
    rec["rule/laplacian_normed_negative_nan"] = int(np.isnan(lneg).sum())
    lz = C.laplacian(S.csr_array((np.array([0.0, 1.0]), ([0, 1], [1, 2])), shape=(3, 3)), normed=True).toarray()
    rec["rule/laplacian_negative_zero"] = int(np.sum((lz == 0) & np.signbit(lz)))
    # yen and conversion
    rec["rule/yen"] = {
        "K3": outcome(lambda: C.yen(dia, 0, 3, 3, return_predecessors=True)),
        "same": outcome(lambda: C.yen(dia, 1, 1, 2)),
        "K0": outcome(lambda: C.yen(dia, 0, 3, 0)),
    }
    rec["rule/from_dense"] = {
        "inf": outcome(lambda: C.csgraph_from_dense(np.array([[0, 1], [np.inf, 0]]), null_value=np.inf)),
        "none": outcome(lambda: C.csgraph_from_dense(np.array([[0, 1], [2, 0]]), null_value=None)),
    }
    # results on the test graphs
    for gid in M.ids():
        g = M.get(gid)
        if not g.square or g.n == 0 or {"cost", "nonfinite", "bipartite"} & g.tags:
            continue
        A = g.scipy()
        d = M.graph_digest(g)
        rec[f"graph/{gid}/components"] = {
            "digest": d,
            "weak": C.connected_components(A)[1].tolist(),
            "strong": C.connected_components(A, connection="strong")[1].tolist(),
        }
        rec[f"graph/{gid}/traversal"] = {
            "digest": d,
            "bfs": C.breadth_first_order(A, 0)[0].tolist(),
            "dfs": C.depth_first_order(A, 0)[0].tolist(),
            "bfs_undirected": C.breadth_first_order(A, 0, directed=False)[0].tolist(),
        }
        if "flow" in g.tags:
            rec[f"graph/{gid}/flow"] = {
                "digest": d,
                "dinic": outcome(lambda A=A, n=g.n: C.maximum_flow(A, 0, n - 1).flow_value),
                "edmonds_karp": outcome(
                    lambda A=A, n=g.n: C.maximum_flow(A, 0, n - 1, method="edmonds_karp").flow_value
                ),
            }
            continue
        if g.n <= 40 and "negcycle" not in g.tags:
            r = {"digest": d}
            for m in ("D", "FW") if "negative" not in g.tags else ("J", "FW"):
                r[m] = outcome(lambda A=A, m=m: C.shortest_path(A, method=m, return_predecessors=True))
            rec[f"graph/{gid}/shortest_path"] = r
        rec[f"graph/{gid}/mst"] = {"digest": d, "tree": outcome(lambda A=A: C.minimum_spanning_tree(A))}
        rec[f"graph/{gid}/rcm"] = {
            "digest": d,
            "perm": C.reverse_cuthill_mckee(A).tolist(),
            "structural_rank": int(C.structural_rank(A)),
        }
    for gid in M.ids("bipartite"):
        g = M.get(gid)
        A = g.scipy()
        rec[f"graph/{gid}/matching"] = {
            "digest": M.graph_digest(g),
            "row": outcome(lambda A=A: C.maximum_bipartite_matching(A)),
            "full": outcome(lambda A=A: C.min_weight_full_bipartite_matching(A)),
            "full_max": outcome(lambda A=A: C.min_weight_full_bipartite_matching(A, maximize=True)),
        }
    for gid in M.ids("cost"):
        g = M.get(gid)
        rec[f"graph/{gid}/assignment"] = {
            "digest": M.graph_digest(g),
            "min": outcome(lambda c=g.matrix: O.linear_sum_assignment(c)),
            "max": outcome(lambda c=g.matrix: O.linear_sum_assignment(c, maximize=True)),
        }
    meta = {
        "library": "scipy",
        "version": scipy.__version__,
        "numpy": np.__version__,
        "tool": "tools/graph_probe.py",
        "date": time.strftime("%Y-%m-%d"),
    }
    log("## scipy")
    log(write_records("scipy.json", meta, rec))


def networkx_records(log):
    import networkx as nx

    M = G()
    rec = {}

    def nxg(g, directed):
        D = nx.DiGraph() if directed else nx.Graph()
        D.add_nodes_from(range(g.n))
        edges = g.edges() if directed else [(i, j, w) for (i, j), w in sorted(M.undirected(g).items())]
        for i, j, w in edges:
            D.add_edge(i, j, weight=float(w))
        return D

    def noloops(D):
        D = D.copy()
        D.remove_edges_from(list(nx.selfloop_edges(D)))
        return D

    for gid in ("dag/30", "negative/dag30", "path/10", "diamond/4"):
        g = M.get(gid)
        D = nxg(g, True)
        rec[f"graph/{gid}/topological_sort"] = {
            "digest": M.graph_digest(g),
            "lexicographical": list(nx.lexicographical_topological_sort(D)),
        }
    rec["rule/topological_cycle"] = outcome(
        lambda: list(nx.lexicographical_topological_sort(nxg(M.get("cycle/12"), True)))
    )
    rec["rule/topological_selfloop"] = outcome(
        lambda: list(nx.lexicographical_topological_sort(nxg(M.get("selfloop/10"), True)))
    )
    for gid in (
        "star/9",
        "path/both10",
        "components/40",
        "er/sym60",
        "ladder/20",
        "grid2d/5x5",
        "er/int60",
        "heavytail/120",
        "dag/30",
        "selfloop/10",
        "isolated/5",
        "oneway/12",
    ):
        g = M.get(gid)
        D = noloops(nxg(g, False))
        bcc = sorted(sorted(tuple(sorted(e)) for e in comp) for comp in nx.biconnected_component_edges(D))
        rec[f"graph/{gid}/structure"] = {
            "digest": M.graph_digest(g),
            "articulation_points": sorted(nx.articulation_points(D)),
            "bridges": sorted(tuple(sorted(e)) for e in nx.bridges(D)),
            "biconnected_edges": bcc,
            "core_number": [nx.core_number(D)[i] for i in range(g.n)],
        }
    rec["rule/core_number_selfloop"] = outcome(lambda: nx.core_number(nxg(M.get("selfloop/10"), False)))
    for gid in ("dangling/50", "er/float100", "grid2d/6x6", "star/9", "selfloop/10", "isolated/5"):
        g = M.get(gid)
        D = nxg(g, True)
        rec[f"graph/{gid}/pagerank"] = {
            "digest": M.graph_digest(g),
            "default": outcome(lambda D=D, n=g.n: [nx.pagerank(D)[i] for i in range(n)]),
            "tight": outcome(lambda D=D, n=g.n: [nx.pagerank(D, tol=1e-15, max_iter=100000)[i] for i in range(n)]),
        }
    rec["rule/pagerank_no_convergence"] = outcome(
        lambda: nx.pagerank(nxg(M.get("dangling/50"), True), alpha=0.9999, max_iter=20)
    )
    for gid in ("grid2d/5x5", "diamond/4", "er/int60", "star/9", "empty/1"):
        g = M.get(gid)
        r = {"digest": M.graph_digest(g)}
        for directed in (True, False):
            for weight in (None, "weight"):
                for normalized, endpoints in ((True, False), (False, False), (True, True)):
                    key = f"{'d' if directed else 'u'}/{'w' if weight else 'u'}/{int(normalized)}{int(endpoints)}"
                    r[key] = outcome(
                        lambda g=g, d=directed, w=weight, nz=normalized, ep=endpoints: [
                            nx.betweenness_centrality(noloops(nxg(g, d)), weight=w, normalized=nz, endpoints=ep)[i]
                            for i in range(g.n)
                        ]
                    )
        rec[f"graph/{gid}/betweenness"] = r
    two = nx.Graph([(0, 1)])
    rec["rule/betweenness_small_n"] = {
        "n2_normalized": outcome(lambda: nx.betweenness_centrality(two)),
        "n2_endpoints": outcome(lambda: nx.betweenness_centrality(two, endpoints=True)),
    }
    meta = {
        "library": "networkx",
        "version": nx.__version__,
        "tool": "tools/graph_probe.py",
        "date": time.strftime("%Y-%m-%d"),
    }
    log("## networkx")
    log(write_records("networkx.json", meta, rec))


# ---------------------------------------------------------------- time


def best(fn, reps=3):
    """The best time in ms of up to `reps` calls (one call when the first takes over a second)."""
    out = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1e3
        out = min(out, dt)
        if dt > 1000:
            break
    return out


def timings(log):
    import networkx as nx
    import scipy.optimize as O
    import scipy.sparse as S
    import scipy.sparse.csgraph as C

    M = G()

    def graph(n, d, symmetric=False, seed=M.SEED):
        indptr, indices, data = M.random_graph(n, d, seed=seed, symmetric=symmetric)
        return S.csr_array((data, indices, indptr), shape=(n, n))

    log(f"## time ({load('at start')}; ms, scipy one thread, best of three)")
    A = graph(10**5, 10)
    rows = [
        ("connected_components weak, n = 1e5, d = 10", lambda: C.connected_components(A)),
        ("connected_components strong, n = 1e5, d = 10", lambda: C.connected_components(A, connection="strong")),
        ("breadth_first_order, n = 1e5, d = 10", lambda: C.breadth_first_order(A, 0)),
        ("depth_first_order, n = 1e5, d = 10", lambda: C.depth_first_order(A, 0)),
        ("dijkstra 1 source, n = 1e5, d = 10", lambda: C.dijkstra(A, indices=0)),
        ("dijkstra 1 source unweighted, n = 1e5, d = 10", lambda: C.dijkstra(A, indices=0, unweighted=True)),
        ("dijkstra 100 sources, n = 1e5, d = 10", lambda: C.dijkstra(A, indices=np.arange(100))),
        (
            "dijkstra 100 sources min_only, n = 1e5, d = 10",
            lambda: C.dijkstra(A, indices=np.arange(100), min_only=True),
        ),
        ("structural_rank, n = 1e5, d = 10", lambda: C.structural_rank(A)),
        ("maximum_bipartite_matching, n = 1e5, d = 10", lambda: C.maximum_bipartite_matching(A)),
    ]
    B = graph(10**4, 10)
    rows.append(("bellman_ford 1 source, n = 1e4, d = 10", lambda: C.bellman_ford(B, indices=0)))
    rows.append(("yen K = 10, n = 1e4, d = 10", lambda: C.yen(B, 0, 9999, 10)))
    Bi = S.csr_array(
        (np.random.Generator(np.random.PCG64(5)).integers(1, 100, B.nnz).astype(np.int32), B.indices, B.indptr),
        shape=B.shape,
    )
    rows.append(("maximum_flow dinic, n = 1e4, d = 10", lambda: C.maximum_flow(Bi, 0, 9999)))
    rows.append(
        ("maximum_flow edmonds_karp, n = 1e4, d = 10", lambda: C.maximum_flow(Bi, 0, 9999, method="edmonds_karp"))
    )
    Cm = graph(2000, 10)
    rows.append(("johnson all pairs, n = 2000, d = 10", lambda: C.johnson(Cm)))
    rows.append(("dijkstra all pairs, n = 2000, d = 10", lambda: C.dijkstra(Cm)))
    W = np.random.Generator(np.random.PCG64(3)).uniform(0.01, 1.01, (1000, 1000))
    np.fill_diagonal(W, 0)
    rows.append(("floyd_warshall dense, n = 1000", lambda: C.floyd_warshall(W)))
    Sy = graph(10**5, 5, symmetric=True)
    rows.append(("minimum_spanning_tree symmetric, n = 1e5, d = 5", lambda: C.minimum_spanning_tree(Sy)))
    Sy10 = graph(10**5, 10, symmetric=True)
    rows.append(
        ("reverse_cuthill_mckee symmetric, n = 1e5, d = 10", lambda: C.reverse_cuthill_mckee(Sy10, symmetric_mode=True))
    )
    rows.append(("laplacian symmetric, n = 1e5, d = 10", lambda: C.laplacian(Sy10)))
    rows.append(("laplacian normed symmetric, n = 1e5, d = 10", lambda: C.laplacian(Sy10, normed=True)))
    g = np.random.Generator(np.random.PCG64(4))
    dense_cost = g.uniform(0, 1, (1000, 1000))
    rows.append(
        (
            "min_weight_full_bipartite_matching dense 1000^2",
            lambda: C.min_weight_full_bipartite_matching(S.csr_array(dense_cost)),
        )
    )
    sp5 = graph(5000, 20) + S.eye_array(5000, format="csr")
    rows.append(
        (
            "min_weight_full_bipartite_matching sparse 5000, d = 20 + diagonal",
            lambda: C.min_weight_full_bipartite_matching(sp5),
        )
    )
    rows.append(("linear_sum_assignment 1000^2", lambda: O.linear_sum_assignment(dense_cost)))
    for name, fn in rows:
        log(f"  {name:62s} {best(fn):10.2f}")
    log(f"networkx {nx.__version__} ({load()}; ms, best of three):")
    N4 = graph(10**4, 5, symmetric=True)
    D4 = nx.from_scipy_sparse_array(N4)
    Dd = nx.from_scipy_sparse_array(graph(10**4, 5), create_using=nx.DiGraph)
    nrows = [
        ("articulation_points, n = 1e4, d = 5 symmetric", lambda: list(nx.articulation_points(D4))),
        ("bridges, n = 1e4, d = 5 symmetric", lambda: list(nx.bridges(D4))),
        ("biconnected_components, n = 1e4, d = 5 symmetric", lambda: list(nx.biconnected_component_edges(D4))),
        ("core_number, n = 1e4, d = 5 symmetric", lambda: nx.core_number(D4)),
        ("pagerank (tol 1e-6), n = 1e4, d = 5", lambda: nx.pagerank(Dd)),
        (
            "betweenness_centrality unweighted, n = 1000, d = 5",
            lambda: nx.betweenness_centrality(nx.from_scipy_sparse_array(graph(1000, 5), create_using=nx.DiGraph)),
        ),
    ]
    for name, fn in nrows:
        log(f"  {name:62s} {best(fn):10.2f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools/graph_probe.py", description=__doc__.split("\n\n")[0])
    p.add_argument("sections", nargs="*", metavar="SECTION", help=f"sections (default all of {', '.join(SECTIONS)})")
    args = p.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        p.error(f"unknown section {unknown[0]!r} (choose from {', '.join(SECTIONS)})")
    log = Log()
    table = {"fw": fw, "scipy": scipy_records, "networkx": networkx_records, "time": timings}
    for s in SECTIONS:
        if not args.sections or s in args.sections:
            table[s](log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
