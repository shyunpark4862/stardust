"""docs/api/smooth.md and the module agree: every module-level name its signature blocks define is planned or
exported, nothing else is exported, an exported name's signature is the documented one, and the documented
signatures follow the area's rules (decisions SM1-SM33 in docs/design/smooth.md §1.1).

A block line is a module function (`name(...)`), a result class (`Name`) or a result's method
(`Name.method(...)`)."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "smooth.md"

# Documented, not implemented yet (docs/design/smooth.md §8). A milestone moves its names from here to
# sd.smooth.__all__.
PLANNED: set[str] = set()
RESULTS = {
    "LowessResult",
    "LoessResult",
    "SmoothingSplineResult",
    "PSplineResult",
    "KernelRegressionResult",
    "KDEResult",
    "DensityResult",
}
REGRESSION_LIKE = RESULTS - {"KDEResult", "DensityResult"}
LINE = re.compile(r"^(?:(?P<cls>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)(?:\((?P<sig>.*)\))?$")
OPTIONS = (
    "frac",
    "it",
    "delta",
    "weights",
    "nan_policy",
    "span",
    "degree",
    "family",
    "iterations",
    "normalize",
    "k",
    "w",
    "axis",
    "lam",
    "s",
    "t",
    "nest",
    "bc_type",
    "df",
    "criterion",
    "n_segments",
    "diff_order",
    "bw",
    "kernel",
    "adjust",
    "bounds",
    "deriv",
    "pos",
    "use",
    "mode",
    "cval",
    "alpha",
    "rng",
    "method",
    "nb",
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


def test_exports_match_api_md():
    doc = set(module_names())
    exported = set(sd.smooth.__all__)
    assert exported.isdisjoint(PLANNED)
    assert exported | PLANNED == doc
    for name in exported:
        assert hasattr(sd.smooth, name)


def test_exported_signatures_match_api_md():
    doc = module_names()
    for name in sd.smooth.__all__:
        if doc[name] is not None:
            assert str(inspect.signature(getattr(sd.smooth, name)))[1:-1] == doc[name], name


def test_lines_are_unique_and_methods_belong_to_documented_results():
    seen = [(c, n) for c, n, _ in entries()]
    assert len(seen) == len(set(seen))
    bare = {n for c, n, s in entries() if c is None and s is None}
    assert bare == RESULTS
    for cls, _, sig in entries():
        assert cls is None or (cls in RESULTS and sig is not None), cls


def test_data_positional_options_keyword():
    # SM5 and the common rule (API.md §5): the data before `/`, every option after `*` with a default
    for cls, name, sig in entries():
        if not sig:
            continue
        ps = parts(sig)
        if "/" in ps:
            before, after = ps[: ps.index("/")], ps[ps.index("/") + 1 :]
        else:
            before, after = [], ps
        assert not after or after[0] == "*" or (cls == "KDEResult" and name == "sample"), (cls, name)
        for p in after[1:]:
            assert "=" in p or p == "*" or p == "rng", (cls, name, p)
        for key in OPTIONS:
            if key in ("t", "k") and name in ("make_lsq_spline",):  # the knots are data there; k is an option
                continue
            assert key not in [q.split("=")[0] for q in before], (cls, name, key)


def test_argument_order_is_x_then_y():
    # SM5: (x, y) and (X, y), never statsmodels' (endog, exog)
    funcs = {n: s for c, n, s in entries() if c is None and s is not None}
    for name in (
        "lowess",
        "make_lsq_spline",
        "make_smoothing_spline",
        "make_splrep",
        "generate_knots",
        "smoothing_spline",
        "pspline",
        "bw_dpill",
    ):
        assert list(params(funcs[name]))[:2] == ["x", "y"], name
    for name in ("loess", "kernel_regression"):
        assert list(params(funcs[name]))[:2] == ["X", "y"], name
    for _, _, sig in entries():
        assert sig is None or not {"endog", "exog", "dataset", "bw_method", "var_type", "reg_type"} & set(params(sig))


def test_weights_have_one_name_per_family():
    # SM6: the new functions say weights=, the scipy-named spline functions scipy's w=
    funcs = {n: s for c, n, s in entries() if c is None and s is not None}
    for name in ("lowess", "loess", "smoothing_spline", "pspline", "kernel_regression", "kde"):
        assert params(funcs[name])["weights"] == "None" and "w" not in params(funcs[name]), name
    for name in ("make_lsq_spline", "make_smoothing_spline", "make_splrep", "make_splprep", "generate_knots"):
        assert params(funcs[name])["w"] == "None" and "weights" not in params(funcs[name]), name


def test_shared_options_have_one_spelling():
    tree = {(c, n): s for c, n, s in entries()}
    funcs = {n: s for c, n, s in entries() if c is None and s is not None}
    for name in ("lowess", "loess", "smoothing_spline", "pspline", "kernel_regression", "kde"):
        assert params(funcs[name])["nan_policy"] == "'propagate'", name
    for name in ("smoothing_spline", "pspline"):
        p = params(funcs[name])
        assert p["lam"] == p["df"] == "None" and p["criterion"] == "'gcv'", name
    for name in ("make_splrep", "make_splprep", "generate_knots"):
        assert params(funcs[name])["s"] == "0.0" and params(funcs[name])["k"] == "3", name
    for cls in REGRESSION_LIKE:
        assert (cls, "predict") in tree and (cls, "summary") in tree, cls
        assert params(tree[(cls, "predict_interval")])["alpha"] == "0.05", cls
        assert tree[(cls, "summary")] == "", cls
    for m in ("pdf", "logpdf", "cdf", "sample", "density", "marginal"):
        assert ("KDEResult", m) in tree, m
    assert ("KDEResult", "summary") not in tree  # a density is not a regression box (§7)
    assert params(tree[("KDEResult", "density")])["binned"] == "True"  # SM26


def test_no_other_libraries_options():
    for _, name, sig in entries():
        p = params(sig) if sig else {}
        assert not {
            "check_finite",
            "is_sorted",
            "return_sorted",
            "missing",
            "xvals",
            "surface",
            "cell",
            "statistics",
            "spar",
            "all_knots",
            "fft",
            "gridsize",
            "clip",
            "rtol",
            "atol",
            "random_state",
        } & set(p), name


def test_scipy_names_left_out_have_their_replacements():
    # the legacy spline classes and functions, and gaussian_kde, are not names here: §8 maps them
    doc = module_names()
    text = API.read_text()
    for name in (
        "splrep",
        "splprep",
        "UnivariateSpline",
        "LSQUnivariateSpline",
        "gaussian_kde",
        "KernelReg",
        "KDEUnivariate",
        "KDEMultivariate",
        "KernelDensity",
        "SmoothBivariateSpline",
    ):
        assert name not in doc, name
        assert f"`{name}" in text, name


def test_module_is_wired():
    assert isinstance(sd.smooth.__all__, list) and len(set(sd.smooth.__all__)) == len(sd.smooth.__all__)
    assert {"lowess", "LowessResult"} <= set(sd.smooth.__all__)  # SM1
    assert "smooth" in dir(sd)
