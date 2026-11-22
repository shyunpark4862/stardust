"""docs/api/interpolate.md and the module agree: every module-level name its signature blocks define is
planned or exported, nothing else is exported, an exported name's signature is the documented one,
and the documented signatures follow the area's rules (decisions IP1-IP35 in
docs/design/interpolate.md §1.1). sd.interp, an array-area function (IP2), is public since milestone IP1; its
signature is in docs/api/array.md §8.5 (tests/array/test_api_surface.py reads it).

A block line is a module name (`name(...)`), a method (`Class.method(...)`) or an attribute
(`Class.attr`)."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "interpolate.md"

# Documented, not implemented yet (docs/design/interpolate.md §8). A milestone moves its names from here
# to sd.interpolate.__all__.
PLANNED = {
    # IP3
    "BSpline", "make_interp_spline",
    # IP4
    "BarycentricInterpolator", "barycentric_interpolate", "KroghInterpolator", "krogh_interpolate",
    "FloaterHormannInterpolator", "AAA",
    # IP5
    "RegularGridInterpolator", "interpn",
    # IP6
    "RBFInterpolator", "NearestNDInterpolator",
}  # fmt: skip
IP2 = {"PPoly", "CubicHermiteSpline", "CubicSpline", "PchipInterpolator", "Akima1DInterpolator", "pchip_interpolate"}
CLASSES = {n for n in PLANNED | IP2 if n[0].isupper()}
PPOLY_FAMILY = ("PPoly", "CubicHermiteSpline", "CubicSpline", "PchipInterpolator", "Akima1DInterpolator")
INTERP = "interp(x, xp, fp, /, *, left=None, right=None, period=None)"
LINE = re.compile(r"^(?:(?P<cls>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)(?:\((?P<sig>.*)\))?$")
OPTIONS = ("axis", "extrapolate", "bc_type", "method", "kernel", "nu", "der", "epsilon", "degree", "smoothing",
           "neighbors", "bounds_error", "fill_value", "rtol", "max_terms", "left", "right", "period")  # fmt: skip


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
    exported = set(sd.interpolate.__all__)
    assert exported.isdisjoint(PLANNED)
    assert exported | PLANNED == doc
    for name in exported:
        assert hasattr(sd.interpolate, name)


def test_exported_signatures_match_api_md():
    doc = module_names()
    for name in sd.interpolate.__all__:
        assert str(inspect.signature(getattr(sd.interpolate, name)))[1:-1] == doc[name], name


def test_lines_are_unique_and_methods_belong_to_documented_classes():
    seen = [(c, n) for c, n, _ in entries()]
    assert len(seen) == len(set(seen))
    for cls, _, _ in entries():
        assert cls is None or cls in CLASSES, cls


def test_data_positional_options_keyword():
    # IP14: the data before `/`, every option after `*` with a default; a method without data starts at `*`
    for cls, name, sig in entries():
        if sig is None or sig == "":
            continue
        ps = parts(sig)
        if "/" in ps:
            before, after = ps[: ps.index("/")], ps[ps.index("/") + 1 :]
        else:
            before, after = [], ps
        assert not after or after[0] == "*", (cls, name)
        for p in after[1:]:
            assert "=" in p, (cls, name, p)
        for key in OPTIONS + ("k",):
            if key == "k" and name in ("BSpline", "design_matrix"):
                continue  # the degree that defines a B-spline is data there
            assert key not in [q.split("=")[0] for q in before], (cls, name, key)


def test_no_scipy_only_options():
    # IP12, IP19, IP21, IP23, IP31: no check_finite, rng, solver, construct_fast, tree_options
    for _, name, sig in entries():
        p = params(sig) if sig else {}
        assert not {"check_finite", "rng", "random_state", "solver", "solver_args", "tree_options", "workers",
                    "assume_sorted", "copy", "kind"} & set(p), name  # fmt: skip


def test_no_legacy_names_or_submodules():
    # IP7: the legacy names have replacements in §8, not aliases; IP3: no submodules
    doc = module_names()
    absent = ("interp1d", "interp2d", "Rbf", "InterpolatedUnivariateSpline", "UnivariateSpline", "RectBivariateSpline",
              "splrep", "splev", "splint", "sproot", "spalde", "splder", "splantider", "insert", "bisplrep", "bisplev",
              "lagrange", "approximate_taylor_polynomial", "pade", "griddata", "LinearNDInterpolator",
              "CloughTocher2DInterpolator", "BPoly", "NdBSpline", "NdPPoly", "make_smoothing_spline", "make_lsq_spline",
              "fitpack", "fitpack2", "interpnd", "polyint", "rbf", "ndgriddata", "interp")  # fmt: skip
    for name in absent:
        assert name not in doc, name
        assert not hasattr(sd.interpolate, name), name
    text = API.read_text()
    for name in ("interp1d", "interp2d", "Rbf", "InterpolatedUnivariateSpline", "RectBivariateSpline", "splrep",
                 "lagrange", "pade"):  # fmt: skip
        assert f"`{name}" in text, name  # each has its line in the replacement table


def test_shared_options_have_one_spelling():
    # IP13: one extrapolate=; axis=0 everywhere; the 1-D family's calls share nu and extrapolate
    tree = {(c, n): s for c, n, s in entries()}
    for cls in PPOLY_FAMILY:
        p = params(tree[(None, cls)])
        assert p["axis"] == "0" and p["extrapolate"] == "None", cls
    assert params(tree[(None, "BSpline")])["extrapolate"] == "True"  # scipy's default for BSpline
    for cls in ("PPoly", "BSpline"):
        assert params(tree[(cls, "__call__")]) == {"x": None, "nu": "0", "extrapolate": "None"}
        assert params(tree[(cls, "integrate")]) == {"a": None, "b": None, "extrapolate": "None"}
        assert params(tree[(cls, "derivative")]) == params(tree[(cls, "antiderivative")]) == {"nu": "1"}
    rgi, interpn = params(tree[(None, "RegularGridInterpolator")]), params(tree[(None, "interpn")])
    for key in ("method", "bounds_error", "fill_value"):
        assert rgi[key] == interpn[key], key
    for fn in ("pchip_interpolate", "krogh_interpolate", "barycentric_interpolate"):
        assert params(tree[(None, fn)])["der"] == "0", fn
    assert params(tree[(None, "AAA")])["clean_up_tol"] == params(tree[("AAA", "clean_up")])["clean_up_tol"]  # IP32


def test_interp_is_the_array_areas():
    # IP2: sd.interp is documented and exported by the array area (docs/api/array.md §8.5) with the signature the
    # draft planned; this document keeps no signature block of it
    array_md = (API.parent / "array.md").read_text()
    assert INTERP in [line.strip() for block in signature_blocks(array_md) for line in block]
    assert "```text" not in API.read_text()
    assert "interp" in sd.__all__
    assert sd.interp.__doc__.split("\n")[0] == INTERP


def test_module_is_wired():
    assert set(sd.interpolate.__all__) == IP2
    assert not hasattr(sd.interpolate, "interp")


def test_exported_methods_match_api_md():
    # the methods and attributes of the exported classes (PPoly.from_spline comes with BSpline, milestone IP3)
    later = {("PPoly", "from_spline")}
    for cls, name, sig in entries():
        if cls not in sd.interpolate.__all__ or (cls, name) in later:
            continue
        member = getattr(getattr(sd.interpolate, cls), name)
        if sig is None:
            assert isinstance(member, property), (cls, name)
        else:
            text = str(inspect.signature(member))[1:-1]
            if text.startswith("self"):
                text = text.removeprefix("self").removeprefix(", ")
            assert text == sig, (cls, name, text)
