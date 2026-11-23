"""docs/api/spatial.md and the module agree: every module-level name its signature blocks define is
planned or exported, nothing else is exported, an exported name's signature is the documented one,
and the documented signatures follow the area's rules (decisions SPT1-SPT28 in
docs/design/spatial.md §1.1).

A block line is a module name (`name(...)`), a method (`Class.method(...)`) or an attribute
(`Class.attr`)."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "spatial.md"

# Documented, not implemented yet (docs/design/spatial.md §8): SPT1 made the distances public. A
# milestone moves its names from here to sd.spatial.__all__.
PLANNED: set[str] = set()  # SPT1-SPT3 made every module-level name public
# The computational geometry's classes are drafted in docs/api/spatial_geometry.md until milestone SPT14 merges
# that draft into spatial.md (decision SPT118); their signatures are checked against the draft.
DRAFT = API.parent / "spatial_geometry.md"
DRAFTED = {"Delaunay", "ConvexHull", "Voronoi"}
CLASSES = {"KDTree", "QueryResult", "RadiusResult"}
METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "minkowski", "seuclidean", "mahalanobis", "cosine",
           "correlation", "canberra", "braycurtis", "hamming", "jaccard", "haversine")  # fmt: skip
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
    exported = set(sd.spatial.__all__) - DRAFTED
    assert exported.isdisjoint(PLANNED)
    assert exported | PLANNED == doc
    for name in exported:
        assert hasattr(sd.spatial, name)


def test_exported_signatures_match_api_md():
    doc = module_names()
    for name in sd.spatial.__all__:
        if name not in DRAFTED:
            assert str(inspect.signature(getattr(sd.spatial, name)))[1:-1] == doc[name], name


def test_drafted_signatures_match_the_draft():
    draft = {}
    for block in signature_blocks(DRAFT.read_text()):
        for line in block:
            m = LINE.match(line.split("#")[0].strip())
            if m and m.group("cls") is None:
                draft[m.group("name")] = m.group("sig")
    for name in DRAFTED & set(sd.spatial.__all__):
        assert str(inspect.signature(getattr(sd.spatial, name)))[1:-1] == draft[name], name


def test_lines_are_unique_and_methods_belong_to_documented_classes():
    seen = [(c, n) for c, n, _ in entries()]
    assert len(seen) == len(set(seen))
    for cls, _, _ in entries():
        assert cls is None or cls in CLASSES, cls


def test_every_metric_has_its_function():
    # SPT1, SPT9: the fourteen metric names are the distance functions
    assert set(METRICS) <= set(module_names())


def test_data_positional_options_keyword():
    # the data before `/`, every option after `*` with a default (SPT15: metric, k and leafsize too)
    for cls, name, sig in entries():
        if not sig:
            continue
        parts = [p.strip() for p in re.split(r",\s*(?![^()]*\))", sig)]
        assert "/" in parts, (cls, name)
        after = parts[parts.index("/") + 1 :]
        assert not after or after[0] == "*", (cls, name)
        for p in after[1:]:
            assert "=" in p, (cls, name, p)
        for key in ("metric", "k", "leafsize", "p", "w"):
            assert key not in [q.split("=")[0] for q in parts[: parts.index("/")]], (cls, name, key)


def test_no_scipy_only_options():
    # SPT10, SPT11, SPT14, SPT17, SPT18, SPT20
    for _, name, sig in entries():
        p = params(sig) if sig else {}
        assert not {"workers", "n_jobs", "eps", "boxsize", "balanced_tree", "compact_nodes", "copy_data", "out",
                    "return_sorted", "output_type", "squared", "algorithm"} & set(p), name  # fmt: skip


def test_no_distance_submodule_or_aliases():
    # SPT5, SPT6: one flat namespace; no aliases (cKDTree, distance_matrix)
    doc = module_names()
    absent = ("distance", "cKDTree", "distance_matrix", "minkowski_distance", "minkowski_distance_p", "Rectangle",
              "BallTree", "NearestNeighbors", "pairwise_distances_argmin_min")  # fmt: skip
    for name in absent:
        assert name not in doc, name
        assert not hasattr(sd.spatial, name), name


def test_shared_options_have_one_spelling():
    doc = module_names()
    metric_opts = {"metric": "'euclidean'", "p": "None", "w": "None", "V": "None", "VI": "None"}
    for name in ("pdist", "cdist", "kneighbors", "radius_neighbors"):
        p = params(doc[name])
        for k, v in metric_opts.items():
            assert p[k] == v, (name, k)
    assert params(doc["kneighbors"])["distance_upper_bound"] == "inf"
    tree = {(c, n): s for c, n, s in entries() if c == "KDTree"}
    assert params(tree[("KDTree", "query")])["distance_upper_bound"] == "inf"
    for (_, n), s in tree.items():
        if s and n not in ("data", "n", "m", "leafsize", "maxes", "mins", "size"):
            assert params(s)["p"] == "2.0", n


def test_module_is_wired():
    # SPT1: pdist, cdist, squareform and the fourteen distance functions; SPT3: KDTree
    assert {"pdist", "cdist", "squareform", "KDTree"} <= set(sd.spatial.__all__)
    assert len([n for n in sd.spatial.__all__ if n not in ("kneighbors", "radius_neighbors") + tuple(DRAFTED)]) == 18
    assert not hasattr(sd.spatial, "distance")
    assert {"kneighbors", "radius_neighbors"} <= set(sd.spatial.__all__)  # SPT2


def test_radius_result_members_match_api_md():
    # SPT2: RadiusResult (not exported) has the documented fields and tocsr()
    from stardust.spatial._radius import RadiusResult

    for c, name, sig in entries():
        if c == "RadiusResult":
            member = getattr(RadiusResult, name)
            assert isinstance(member, property) if sig is None else callable(member), name
    assert not hasattr(sd.spatial, "RadiusResult")


def test_kdtree_members_match_api_md():
    # SPT3, SPT4: the constructor, the queries and the attributes are public with the documented signatures
    cls = sd.spatial.KDTree
    for c, name, sig in entries():
        if c != "KDTree":
            continue
        member = getattr(cls, name)
        if sig is None:
            assert isinstance(member, property), name
        else:
            assert str(inspect.signature(member))[1:-1] == "self, " + sig, name
