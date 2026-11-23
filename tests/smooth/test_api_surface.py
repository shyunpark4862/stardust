"""docs/api/smooth.md and the module agree: the module-level names its signature blocks define are exactly
sd.smooth.__all__, every signature is the documented one, and the documented signatures follow the area's rules
(decisions SM1-SM35 and SM141 in docs/design/smooth.md §1.1).

A block line is a module function (`name(...)`), a result class (`Name`) or a result's method
(`Name.method(...)`)."""

import inspect
import pathlib
import re

import pytest

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "smooth.md"

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
    assert exported == doc
    for name in exported:
        assert hasattr(sd.smooth, name)


def test_exported_signatures_match_api_md():
    doc = module_names()
    for name in sd.smooth.__all__:
        if doc[name] is not None:
            assert str(inspect.signature(getattr(sd.smooth, name)))[1:-1] == doc[name], name


def test_documented_methods_match_the_results():
    for cls, name, sig in entries():
        if cls is None:
            continue
        method = getattr(getattr(sd.smooth, cls), name)
        assert str(inspect.signature(method)).removeprefix("(self, ").removeprefix("(self")[:-1] == sig, (cls, name)


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
    assert "smooth" in dir(sd)


def _results():
    x, y = [0.0, 1, 2, 3, 4, 5, 6, 7], [1.0, 2, 1, 3, 2, 4, 3, 5]
    return [
        sd.smooth.lowess(x, y),
        sd.smooth.loess(x, y),
        sd.smooth.smoothing_spline(x, y),
        sd.smooth.pspline(x, y),
        sd.smooth.kernel_regression(x, y, bw=1.0),
        sd.smooth.kde([0.0, 1.0, 3.0]),
    ]


def test_wrong_call_shapes_start_with_the_name_called():
    """API.md §2.1 for every public method of the results and every function (decision SM145): the method's own
    name, never a hidden base class's ("_PenalizedResult.predict() ...")."""
    for r in _results():
        for name in [m for m in dir(type(r)) if not m.startswith("_") and callable(getattr(type(r), m))]:
            method = getattr(r, name)
            for args, kwargs in (((), {"foo": 1}), ((1.0, 2.0, 3.0), {})):
                with pytest.raises(TypeError) as e:
                    method(*args, **kwargs)
                assert str(e.value).startswith(f"{name}(): "), (type(r).__name__, str(e.value))
    for name in sd.smooth.__all__:
        f = getattr(sd.smooth, name)
        if inspect.isfunction(f):
            with pytest.raises(TypeError, match=rf"^{name}\(\): unexpected keyword argument 'foo'$"):
                f(foo=1)


def test_results_show_no_constants():
    """The summary's and repr's constants are private (_TITLE, ...): dir() shows the fields and methods only."""
    for r in _results():
        assert not [m for m in dir(r) if m.isupper() and not m.startswith("_")], type(r).__name__
