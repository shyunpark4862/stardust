"""docs/api/cluster.md and the module agree: every module-level name its signature blocks define belongs to the
milestone that makes it public (§8 of docs/design/cluster.md), the names public so far are exactly
sd.cluster.__all__ with the documented signatures, and the documented signatures follow the area's rules
(decisions CL1-CL50 in docs/design/cluster.md §1.1).

A block line is a module function (`name(...)`), a result class (`Name`) or a result's method
(`Name.method(...)`)."""

import inspect
import pathlib
import re

import pytest

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "cluster.md"

LINE = re.compile(r"^(?:(?P<cls>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)(?:\((?P<sig>.*)\))?$")
# the milestone that makes each documented module-level name public (docs/design/cluster.md §8)
MILESTONE = {
    **dict.fromkeys(
        "linkage agglomerative divisive HierarchyResult fcluster cut_tree cophenet cophenet_correlation CophenetResult "
        "leaves_list "
        "optimal_leaf_ordering dendrogram DendrogramResult inconsistent maxdists maxinconsts maxRstat leaders "
        "is_monotonic is_valid_linkage num_obs_linkage is_isomorphic".split(),
        "CL1",
    ),
    **dict.fromkeys(
        "kmeans minibatch_kmeans bisecting_kmeans KMeansResult BisectingKMeansResult kmedoids KMedoidsResult "
        "fuzzy_cmeans FuzzyCMeansResult".split(),
        "CL2",
    ),
    **dict.fromkeys("dbscan DBSCANResult hdbscan HDBSCANResult optics OPTICSResult".split(), "CL3"),
    **dict.fromkeys("gaussian_mixture GaussianMixtureResult".split(), "CL4"),
    **dict.fromkeys(
        "silhouette_samples silhouette_score calinski_harabasz_score davies_bouldin_score adjusted_rand_score "
        "rand_score adjusted_mutual_info_score normalized_mutual_info_score mutual_info_score homogeneity_score "
        "completeness_score v_measure_score homogeneity_completeness_v_measure HomogeneityResult "
        "fowlkes_mallows_score pair_confusion_matrix contingency_matrix gap_statistic GapResult "
        "gower_distance".split(),
        "CL5",
    ),
    **dict.fromkeys("affinity_propagation AffinityPropagationResult".split(), "CL6"),
}
DONE = ("CL1", "CL2", "CL3", "CL4", "CL5", "CL6")  # the milestones whose names are public
RESULTS = {
    "HierarchyResult",
    "KMeansResult",
    "BisectingKMeansResult",
    "KMedoidsResult",
    "FuzzyCMeansResult",
    "DBSCANResult",
    "HDBSCANResult",
    "OPTICSResult",
    "GaussianMixtureResult",
    "GapResult",
    "AffinityPropagationResult",
}
TUPLES = {"DendrogramResult", "CophenetResult", "HomogeneityResult"}  # namedtuples: no methods
RNG = (
    "kmeans",
    "minibatch_kmeans",
    "bisecting_kmeans",
    "fuzzy_cmeans",
    "gaussian_mixture",
    "gap_statistic",
)
DATA_FIRST = {"X", "y", "Z", "labels_true", "data", "T1"}


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


def parts(sig):
    return [p.strip() for p in re.split(r",\s*(?![^()]*\))", sig) if p.strip()]


def params(sig):
    """{name: default text or None} of a documented signature, in order (`/` and `*` left out)."""
    out = {}
    for p in parts(sig):
        if p in ("/", "*"):
            continue
        name, _, default = p.partition("=")
        out[name.strip()] = default.strip() or None
    return out


def test_every_documented_name_has_a_milestone():
    assert set(module_names()) == set(MILESTONE)


def test_exports_are_the_documented_names_of_the_done_milestones():
    exported = set(sd.cluster.__all__)
    assert exported == {n for n, m in MILESTONE.items() if m in DONE}
    for name in exported:
        assert hasattr(sd.cluster, name)


def test_exported_signatures_match_api_md():
    doc = module_names()
    for name in sd.cluster.__all__:
        if doc[name] is not None:
            assert str(inspect.signature(getattr(sd.cluster, name)))[1:-1] == doc[name], name


def test_documented_methods_match_the_exported_results():
    for cls, name, sig in entries():
        if cls is None or cls not in sd.cluster.__all__:
            continue
        method = getattr(getattr(sd.cluster, cls), name)
        assert str(inspect.signature(method)).removeprefix("(self, ").removeprefix("(self")[:-1] == sig, (cls, name)


def test_lines_are_unique_and_methods_belong_to_documented_results():
    seen = [(c, n) for c, n, _ in entries()]
    assert len(seen) == len(set(seen))
    bare = {n for c, n, s in entries() if c is None and s is None}
    assert bare == RESULTS | TUPLES
    for cls, _, sig in entries():
        assert cls is None or (cls in RESULTS and sig is not None), cls


def test_data_positional_options_keyword():
    # CL5 and the common rule (API.md §5): the data before `/`, every option after `*` with a default, except
    # rng (CL8: required, no default) and kmedoids' rng (only for method='clara': None)
    for cls, name, sig in entries():
        if not sig:
            continue
        ps = parts(sig)
        before, after = (ps[: ps.index("/")], ps[ps.index("/") + 1 :]) if "/" in ps else ([], ps)
        assert not after or after[0] == "*", (cls, name)
        for p in after[1:]:
            assert "=" in p or p == "rng", (cls, name, p)
        for p in before:
            assert "=" not in p or p.startswith("shape"), (cls, name, p)


def test_k_is_the_second_positional_argument():
    # CL5: (X, k) for every partitioning and mixture method; no n_clusters or n_components
    funcs = {n: s for c, n, s in entries() if c is None and s is not None}
    for name in ("kmeans", "minibatch_kmeans", "bisecting_kmeans", "kmedoids", "fuzzy_cmeans", "gaussian_mixture"):
        assert list(params(funcs[name]))[:2] == ["X", "k"], name
    for _, name, sig in entries():
        banned = {"n_clusters", "n_components", "random_state", "n_jobs", "copy"}
        if name == "cut_tree":  # a scipy-named tool keeps scipy's keyword (CL4)
            banned.discard("n_clusters")
        assert sig is None or not banned & set(params(sig)), name


def test_rng_is_required_where_a_function_draws():
    # CL8: keyword only and without a default where the method always draws; kmedoids takes it for 'clara' only
    tree = {(c, n): s for c, n, s in entries()}
    for name in RNG:
        p = params(tree[(None, name)])
        assert "rng" in p and p["rng"] is None, name
    assert params(tree[(None, "kmedoids")])["rng"] == "None"
    assert params(tree[("GaussianMixtureResult", "sample")])["rng"] is None
    for (cls, name), sig in tree.items():
        if sig and "rng" in params(sig):
            assert name in RNG + ("kmedoids", "sample"), (cls, name)


def test_shared_options_have_one_spelling():
    tree = {(c, n): s for c, n, s in entries()}
    for (cls, name), sig in tree.items():
        if not sig:
            continue
        p = params(sig)
        if "nan_policy" in p:
            assert p["nan_policy"] == "'raise'", (cls, name)  # CL11
        if "weights" in p:
            assert p["weights"] == "None", (cls, name)
        if "metric" in p:
            assert p["metric"] == "'euclidean'", (cls, name)
    for name in ("linkage", "agglomerative"):
        assert params(tree[(None, name)])["method"] == "'single'"  # CL16: scipy's default
    for cls in RESULTS:
        assert tree.get((cls, "summary")) == "", cls
    for cls in (
        "KMeansResult",
        "BisectingKMeansResult",
        "KMedoidsResult",
        "FuzzyCMeansResult",
        "GaussianMixtureResult",
    ):
        assert tree[(cls, "predict")] == "X_new, /", cls


def test_first_arguments_are_data():
    for cls, name, sig in entries():
        if cls is None and sig:
            assert list(params(sig))[0] in DATA_FIRST | {"y"}, name


def test_no_other_libraries_options():
    for _, name, sig in entries():
        p = params(sig) if sig else {}
        assert not {
            "random_state",
            "n_jobs",
            "copy",
            "algorithm",
            "leaf_size",
            "store_centers",
            "sample_weight",
            "verbose",
            "no_plot",
            "throw",
            "warning",
        } & set(p), name


def test_names_left_out_have_their_replacements():
    # scipy's object tree, old formats and colour state, and sklearn's estimators are not names here: §8 maps them
    doc = module_names()
    text = API.read_text()
    for name in ("to_tree", "ClusterNode", "fclusterdata", "whiten", "KMeans", "DBSCAN", "SpectralClustering"):
        assert name not in doc, name
        assert f"`{name}" in text or f"{name}(" in text, name


def test_module_is_wired():
    assert isinstance(sd.cluster.__all__, list) and len(set(sd.cluster.__all__)) == len(sd.cluster.__all__)
    assert "cluster" in dir(sd)
    assert hasattr(sd._core, "cluster")


def test_wrong_call_shapes_start_with_the_name_called():
    """API.md §2.1 for every exported function (none yet at CL0; test_results.py checks the results' base)."""
    for name in sd.cluster.__all__:
        f = getattr(sd.cluster, name)
        if inspect.isfunction(f):
            with pytest.raises(TypeError, match=rf"^{name}\(\): unexpected keyword argument 'foo'$"):
                f(foo=1)
