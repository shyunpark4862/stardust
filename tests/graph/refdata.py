"""sd.graph's references and their contract (docs/design/graph.md §4, §7), as the tests and the
accuracy measurement (bench/accuracy) will both judge them.

tests/graph/ref/<family>.npz (tools/graph_ref.py; layout: tests/reffile.py) holds one point per graph
and options: the inputs "graph" (the id in tests/graph/graphs.py), "digest" (graphs.graph_digest of
the graph) and the family's options, and per method the reference hi + lo (a double-double; per-node
values are vectors):

    paths        source; methods directed, unweighted, undirected, undirected_unweighted (fields pred,
                 edges, abs_sum, exact_pred) and negcycle, negcycle_undirected (1 or 0)
    yen          source, sink; method lengths (fields edges, paths, total)
    assignment   method feasible (1 or 0), min and max (fields count, col_ind)
    pagerank     alpha, personalization, dangling (empty: the default), directed, unweighted; method x
    betweenness  directed, unweighted, normalized, endpoints; method bc
    laplacian    normed, use_out_degree, symmetrized; method L (row-major n x n)

tests/graph/data/scipy.json and networkx.json hold the comparison libraries' records
(tools/graph_probe.py), each with its own "_provenance".

The contract (draft until the milestones measure it; docs/design/graph.md §4), in units of its bound
(1 or less is within it): a shortest distance d of a path of at most H edges and absolute weight sum S
is within gamma_{H-1} S of the true distance; integer weights are exact.
"""

from __future__ import annotations

import json
import math
import pathlib

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("paths", "yen", "assignment", "pagerank", "betweenness", "laplacian")
U = 2.0**-53
NONE = -9999  # the predecessor marker
YEN_K = 20  # the paths per yen point (tools/graph_ref.py)


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


def ref(family: str) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def gamma(k: int) -> float:
    return k * U / (1 - k * U) if k > 0 else 0.0


def bound(edges: int, abs_sum: float) -> float:
    """The contract's absolute bound for a shortest distance: gamma_{H-1} S (docs/design/graph.md §4)."""
    return gamma(edges - 1) * abs_sum


def units(got: float, hi: float, lo: float, edges: int, abs_sum: float) -> float:
    """|got - ref| in units of the contract's bound (0 when both are exactly equal)."""
    if got == hi and lo == 0:
        return 0.0
    if math.isinf(hi) or math.isinf(got):
        return 0.0 if got == hi else math.inf
    err = abs((got - hi) - lo)
    b = bound(edges, abs_sum)
    return math.inf if b == 0 else err / b
