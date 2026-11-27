"""docs/api/embed.md and the module agree: the module-level names its signature blocks define are exactly
sd.embed.__all__ with the documented signatures, and the documented signatures follow the area's rules (decisions
in docs/design/embed.md §1.1). The same mechanism as tests/cluster/test_api_surface.py.

A block line is a module function (`name(...)`), a result class (`Name`) or a result's method
(`Name.method(...)`)."""

import inspect
import pathlib
import re

import pytest

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "embed.md"

LINE = re.compile(r"^(?:(?P<cls>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)(?:\((?P<sig>.*)\))?$")
# the milestone that makes each documented module-level name public (docs/design/embed.md §8)
MILESTONE = {
    **dict.fromkeys(
        "pca PCAResult truncated_svd TruncatedSVDResult random_projection RandomProjectionResult "
        "johnson_lindenstrauss_min_dim".split(),
        "EM1",
    ),
    **dict.fromkeys(
        "factor_analysis FactorAnalysisResult rotate_factors RotationResult fastica FastICAResult".split(), "EM2"
    ),
    **dict.fromkeys("nmf NMFResult kernel_pca KernelPCAResult".split(), "EM3"),
    **dict.fromkeys("classical_mds ClassicalMDSResult mds sammon MDSResult stress".split(), "EM4"),
    **dict.fromkeys(
        "isomap IsomapResult locally_linear_embedding LLEResult spectral_embedding SpectralEmbeddingResult "
        "trustworthiness continuity".split(),
        "EM5",
    ),
    **dict.fromkeys("tsne TSNEResult".split(), "EM6"),
    **dict.fromkeys("umap UMAPResult".split(), "EM7"),
}
DONE = ()  # the milestones whose names are public
RESULTS = {
    "PCAResult",
    "TruncatedSVDResult",
    "RandomProjectionResult",
    "FactorAnalysisResult",
    "FastICAResult",
    "NMFResult",
    "KernelPCAResult",
    "ClassicalMDSResult",
    "MDSResult",
    "IsomapResult",
    "LLEResult",
    "SpectralEmbeddingResult",
    "TSNEResult",
    "UMAPResult",
}
TUPLES = {"RotationResult"}  # namedtuples: no methods
RNG_REQUIRED = ("fastica", "random_projection", "umap")  # EM11: they always draw
RNG_OPTIONAL = ("pca", "truncated_svd", "nmf", "kernel_pca", "mds", "tsne")  # EM11: an option draws
VALUES = ("johnson_lindenstrauss_min_dim", "rotate_factors", "trustworthiness", "continuity", "stress")
TRANSFORM = (  # docs/design/embed.md §2.1: the results that move new points (X_new, /)
    "PCAResult",
    "TruncatedSVDResult",
    "RandomProjectionResult",
    "FastICAResult",
    "NMFResult",
    "KernelPCAResult",
    "IsomapResult",
    "LLEResult",
)
INVERSE = (
    "PCAResult",
    "TruncatedSVDResult",
    "RandomProjectionResult",
    "FastICAResult",
    "NMFResult",
    "KernelPCAResult",
)


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
    exported = set(sd.embed.__all__)
    assert exported == {n for n, m in MILESTONE.items() if m in DONE}
    for name in exported:
        assert hasattr(sd.embed, name)


def test_exported_signatures_match_api_md():
    doc = module_names()
    for name in sd.embed.__all__:
        if doc[name] is not None:
            assert str(inspect.signature(getattr(sd.embed, name)))[1:-1] == doc[name], name


def test_documented_methods_match_the_exported_results():
    for cls, name, sig in entries():
        if cls is None or cls not in sd.embed.__all__:
            continue
        method = getattr(getattr(sd.embed, cls), name)
        assert str(inspect.signature(method)).removeprefix("(self, ").removeprefix("(self")[:-1] == sig, (cls, name)


def test_lines_are_unique_and_methods_belong_to_documented_results():
    seen = [(c, n) for c, n, _ in entries()]
    assert len(seen) == len(set(seen))
    bare = {n for c, n, s in entries() if c is None and s is None}
    assert bare == RESULTS | TUPLES
    for cls, _, sig in entries():
        assert cls is None or (cls in RESULTS and sig is not None), cls


def test_data_positional_options_keyword():
    # EM6 and the common rule (API.md §5): the data and n_components before `/`, every option after `*` with a
    # default, except rng where the function always draws (EM11: required, no default)
    for cls, name, sig in entries():
        if not sig:
            continue
        ps = parts(sig)
        before, after = (ps[: ps.index("/")], ps[ps.index("/") + 1 :]) if "/" in ps else ([], ps)
        assert not after or after[0] == "*", (cls, name)
        for p in after[1:]:
            assert "=" in p or p == "rng", (cls, name, p)
        for p in before:
            assert "=" not in p or p.startswith("n_components="), (cls, name, p)


def test_n_components_is_the_second_positional_argument():
    # EM6: (X, n_components) for every fitting function; factor_analysis alone has no default (R's factanal)
    funcs = {n: s for c, n, s in entries() if c is None and s is not None}
    fitting = [n for n in funcs if n not in VALUES]
    assert len(fitting) == 15
    for name in fitting:
        p = params(funcs[name])
        assert list(p)[:2] == ["X", "n_components"], name
        assert (p["n_components"] is None) == (name == "factor_analysis"), name
    for _, name, sig in entries():
        banned = {"k", "n_clusters", "random_state", "n_jobs", "copy", "iterated_power", "fun_args", "verbose"}
        assert sig is None or not banned & set(params(sig)), name


def test_rng_is_required_where_a_function_draws():
    # EM11: keyword only without a default where the function always draws, None where an option draws
    tree = {(c, n): s for c, n, s in entries()}
    for name in RNG_REQUIRED:
        assert params(tree[(None, name)])["rng"] is None, name
    for name in RNG_OPTIONAL:
        assert params(tree[(None, name)])["rng"] == "None", name
    assert params(tree[("UMAPResult", "transform")])["rng"] is None
    for (cls, name), sig in tree.items():
        if sig and "rng" in params(sig):
            assert name in RNG_REQUIRED + RNG_OPTIONAL or (cls, name) == ("UMAPResult", "transform"), (cls, name)


def test_shared_options_have_one_spelling():
    tree = {(c, n): s for c, n, s in entries()}
    for (cls, name), sig in tree.items():
        if not sig:
            continue
        p = params(sig)
        if "nan_policy" in p:
            assert p["nan_policy"] == "'raise'", (cls, name)  # EM9
        if "metric" in p:
            assert p["metric"] == "'euclidean'", (cls, name)  # EM10
        if "neighbors" in p:
            assert p["neighbors"] == "'exact'", (cls, name)  # EM17, §9's second question: exact by default
        if "eigen_solver" in p:
            assert p["eigen_solver"] == "'auto'", (cls, name)
        if "n_oversamples" in p:
            assert p["n_oversamples"] == "10" and "n_power_iter" in p, (cls, name)  # EM21
    assert params(tree[(None, "pca")])["svd_solver"] == "'auto'"
    assert params(tree[(None, "truncated_svd")])["algorithm"] == "'lanczos'"  # EM22
    for cls in RESULTS:
        assert tree.get((cls, "summary")) == "", cls
    for cls in TRANSFORM:
        assert tree[(cls, "transform")] == "X_new, /", cls
    assert tree[("FactorAnalysisResult", "transform")] == "X_new, /, *, method='regression'"
    assert tree[("UMAPResult", "transform")] == "X_new, /, *, rng"
    for cls in INVERSE:
        assert tree[(cls, "inverse_transform")] == "Z, /", cls
    for name in ("trustworthiness", "continuity", "stress"):
        assert list(params(tree[(None, name)]))[:2] == ["X", "X_embedded"], name


def test_first_arguments_are_data():
    for cls, name, sig in entries():
        if cls is None and sig:
            assert list(params(sig))[0] in {"X", "n_samples", "loadings"}, name


def test_names_left_out_have_their_replacements():
    # sklearn's estimators and the later candidates are not names here: §7 maps them
    doc = module_names()
    text = API.read_text()
    for name in ("PCA", "TSNE", "UMAP", "IncrementalPCA", "SparsePCA", "LatentDirichletAllocation", "NNDescent"):
        assert name not in doc, name
        assert f"`{name}" in text or f"{name}(" in text, name


def test_module_is_wired():
    assert isinstance(sd.embed.__all__, list) and len(set(sd.embed.__all__)) == len(sd.embed.__all__)
    assert "embed" in dir(sd)
    assert hasattr(sd._core, "embed")


def test_wrong_call_shapes_start_with_the_name_called():
    """API.md §2.1 for every exported function (test_results.py checks the results' base)."""
    for name in sd.embed.__all__:
        f = getattr(sd.embed, name)
        if inspect.isfunction(f):
            with pytest.raises(TypeError, match=rf"^{name}\(\): unexpected keyword argument 'foo'$"):
                f(foo=1)
