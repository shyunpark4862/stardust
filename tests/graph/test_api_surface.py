"""docs/api/graph.md and the module agree: every module-level name its signature blocks define is
planned or exported, nothing else is exported, an exported name's signature is the documented one,
and the documented signatures follow the area's rules (decisions GR1-GR28 in docs/design/graph.md
§1.1).

A block line is a module name (`name(...)`, or a bare exception class name) or a result field
(`Class.field`)."""

import inspect
import pathlib
import re

import scipy.sparse.csgraph

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "graph.md"

# Documented, not implemented yet (docs/design/graph.md §8). A milestone moves its names from here to
# sd.graph.__all__ (GR1: the traversals, connected_components, topological_sort, the conversion tools;
# GR2: the shortest paths; GR3: the spanning tree, flow, matchings and assignment).
GR1 = (
    "connected_components",
    "breadth_first_order",
    "depth_first_order",
    "breadth_first_tree",
    "depth_first_tree",
    "topological_sort",
    "csgraph_from_dense",
    "csgraph_to_dense",
)
GR2 = (
    "shortest_path",
    "dijkstra",
    "bellman_ford",
    "johnson",
    "floyd_warshall",
    "yen",
    "construct_dist_matrix",
    "reconstruct_path",
    "NegativeCycleError",
)
GR3 = (
    "minimum_spanning_tree",
    "maximum_flow",
    "maximum_bipartite_matching",
    "min_weight_full_bipartite_matching",
    "linear_sum_assignment",
)
PLANNED = {
    # GR4
    "reverse_cuthill_mckee",
    "structural_rank",
    "laplacian",
    # GR5
    "articulation_points",
    "bridges",
    "biconnected_components",
    "core_number",
    "pagerank",
    "betweenness_centrality",
}
RESULTS = {
    "ShortestPathResult",
    "MinOnlyResult",
    "YenResult",
    "TraversalResult",
    "ConnectedComponentsResult",
    "AssignmentResult",
    "LaplacianResult",
    "BiconnectedComponentsResult",
    "MaximumFlowResult",
}
MASKED = {"csgraph_masked_from_dense", "csgraph_from_masked", "csgraph_to_masked"}  # GR14: no masked arrays
POSITIONAL = {"csgraph", "graph", "biadjacency", "cost_matrix", "i_start", "source", "sink", "K", "predecessors"}
LINE = re.compile(r"^(?:(?P<cls>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)(?:\((?P<sig>.*)\))?$")


def entries():
    """[(class or None, name, signature text or None)] of every block line, in order."""
    out = []
    for block in signature_blocks(API.read_text()):
        for line in block:
            text = line.split("#")[0].strip()
            if not text:
                continue
            m = LINE.match(text)
            assert m, f"not a signature line: {line!r}"
            out.append((m.group("cls"), m.group("name"), m.group("sig")))
    return out


def module_names():
    return {name: sig for cls, name, sig in entries() if cls is None}


def params(sig):
    """{name: default text or None} of a documented signature, in order (`/` and `*` left out)."""
    out = {}
    for p in re.split(r",\s*(?![^()]*\))", sig):
        p = p.strip()
        if p in ("/", "*") or not p:
            continue
        name, _, default = p.partition("=")
        out[name.strip()] = default.strip() or None
    return out


def test_exports_match_api_md():
    doc = set(module_names())
    exported = set(sd.graph.__all__)
    assert exported.isdisjoint(PLANNED)
    assert exported | PLANNED == doc
    for name in exported:
        assert hasattr(sd.graph, name)


def test_exported_signatures_match_api_md():
    doc = module_names()
    for name in sd.graph.__all__:
        obj = getattr(sd.graph, name)
        if doc[name] is None:  # an exception class
            assert isinstance(obj, type) and issubclass(obj, ValueError), name
            continue
        assert str(inspect.signature(obj))[1:-1] == doc[name], name


def test_lines_are_unique_and_fields_belong_to_results():
    seen = [(c, n) for c, n, _ in entries()]
    assert len(seen) == len(set(seen))
    for cls, _, sig in entries():
        assert cls is None or (cls in RESULTS and sig is None), cls
    assert {c for c, _, _ in entries() if c} == RESULTS


def test_every_scipy_csgraph_name_is_planned_or_left_out():
    # GR1: scipy 1.18.1's csgraph in full, except the masked-array tools (GR14)
    doc = set(module_names())
    assert set(scipy.sparse.csgraph.__all__) == (doc & set(scipy.sparse.csgraph.__all__)) | MASKED
    assert not MASKED & doc
    assert "linear_sum_assignment" in doc  # GR4


def test_data_positional_options_keyword():
    # GR21: the graph and the main arguments before `/`, every option after `*` with a default
    for _, name, sig in entries():
        if not sig:
            continue
        parts = [p.strip() for p in re.split(r",\s*(?![^()]*\))", sig)]
        assert "/" in parts, name
        before = parts[: parts.index("/")]
        assert set(before) <= POSITIONAL, (name, before)
        after = parts[parts.index("/") + 1 :]
        assert not after or after[0] == "*", name
        for p in after[1:]:
            assert "=" in p, (name, p)


def test_no_scipy_only_options():
    # GR14: no overwrite, copy; GR20: no sampling options; GR19: no max_iter
    for _, name, sig in entries():
        p = params(sig) if sig else {}
        assert not {"overwrite", "copy", "k", "seed", "rng", "max_iter", "nstart", "weight"} & set(p), name


def test_shared_options_have_one_spelling():
    doc = module_names()
    for name in ("shortest_path", "dijkstra", "bellman_ford", "johnson", "floyd_warshall", "yen"):
        p = params(doc[name])
        assert p["directed"] == "True" and p["return_predecessors"] == "False" and p["unweighted"] == "False", name
    for name in ("dijkstra", "bellman_ford", "johnson", "shortest_path"):
        assert params(doc[name])["indices"] == "None", name
    for name in ("breadth_first_order", "depth_first_order"):
        assert params(doc[name])["return_predecessors"] == "True", name
    for name in ("min_weight_full_bipartite_matching", "linear_sum_assignment"):
        assert params(doc[name])["maximize"] == "False", name
    assert params(doc["laplacian"])["form"] == "'array'"
    assert params(doc["dijkstra"])["limit"] == "inf"


def test_module_is_wired():
    assert sd.graph.__all__ == list(GR1 + GR2 + GR3)
    for name in MASKED | {"Graph", "graph", "csgraph"}:
        assert not hasattr(sd.graph, name), name
