"""docs/api/optimize.md and the module agree: every name its signature blocks define is either
exported or planned for a later milestone (O1-O5), nothing else is exported, and the exported
functions' signatures are the documented ones."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "optimize.md"

# Documented, not implemented yet (docs/design/optimize.md §8): O3-O4 minimize; O5 root lsq_linear
# nnls. (O1's approx_fprime check_grad least_squares and O2's root_scalar minimize_scalar find_root
# bracket_root find_minimum bracket_minimum are exported.)
PLANNED = {
    "lsq_linear",
    "nnls",
    "minimize",
    "root",
}
RESULTS = {"OptimizeResult"}


def documented():
    """{name: signature text} for every `name(...)` line in the signature blocks."""
    names = {}
    for block in signature_blocks(API.read_text()):
        joined = " ".join(line.split("#")[0].strip() for line in block)
        for m in re.finditer(r"([A-Za-z_0-9]+)\((.*?)\)(?=\s*([A-Za-z_0-9]+\(|$))", joined):
            names[m.group(1)] = m.group(2)
    return names


def test_exports_match_api_md():
    doc = set(documented())
    functions = set(sd.optimize.__all__) - RESULTS
    assert functions.isdisjoint(PLANNED)
    assert functions | PLANNED == doc
    for name in sd.optimize.__all__:
        assert hasattr(sd.optimize, name)
    assert RESULTS <= set(sd.optimize.__all__)


def test_signatures_match_api_md():
    doc = documented()
    for name in set(sd.optimize.__all__) - RESULTS:
        sig = str(inspect.signature(getattr(sd.optimize, name)))[1:-1]
        assert sig == doc[name], name


def test_no_scipy_individual_functions():
    # O10, O11: only the unified functions; no curve_fit
    for name in (
        "brentq",
        "toms748",
        "bisect",
        "newton",
        "fminbound",
        "brent",
        "golden",
        "fmin",
        "fmin_bfgs",
        "fmin_l_bfgs_b",
        "fmin_ncg",
        "fsolve",
        "leastsq",
        "curve_fit",
    ):
        assert not hasattr(sd.optimize, name)
        assert name not in documented()


BATCH = ("find_root", "bracket_root", "find_minimum", "bracket_minimum")


def test_no_args_or_kwargs():
    # O12: the user function's extra arguments are passed by closures, except the batch functions'
    # args (O39: the elements in progress get theirs)
    for name, sig in documented().items():
        params = [p.strip().split("=")[0] for p in sig.split(",")]
        assert "kwargs" not in params, name
        assert ("args" in params) == (name in BATCH), name


def test_iteration_caps_are_max_iter_and_max_nfev():
    # O14: regression's names, not scipy's maxiter / maxfev / maxfun
    for name, sig in documented().items():
        params = [p.strip().split("=")[0] for p in sig.split(",")]
        assert not {"maxiter", "maxfev", "maxfun"} & set(params), name
    caps = {n: [p.strip().split("=")[0] for p in s.split(",") if "max_" in p] for n, s in documented().items()}
    assert caps["least_squares"] == ["max_nfev"] and caps["root_scalar"] == ["max_iter"]
    assert caps["nnls"] == caps["lsq_linear"] == caps["find_root"] == ["max_iter"]


def test_batch_functions_take_tolerance_keywords_and_no_callback():
    # O36, O37: xatol xrtol fatol frtol instead of scipy's tolerances dict; no callback anywhere in O2
    doc = documented()
    for name in ("find_root", "find_minimum"):
        params = [p.strip().split("=")[0] for p in doc[name].split(",")]
        assert {"xatol", "xrtol", "fatol", "frtol"} <= set(params) and "tolerances" not in params, name
    for name in ("root_scalar", "minimize_scalar", "find_root", "bracket_root", "find_minimum", "bracket_minimum"):
        assert "callback" not in doc[name], name
