"""docs/api/plot.md and the module agree: every module-level name its signature blocks define is
planned or exported, nothing else is exported, an exported name's signature is the documented
one (`**figure` standing for the figure options of §1.1), and the documented signatures follow
the area's rules (docs/design/plot.md §1.3).

A block line is a module name (`name(...)`, or `name` for a class used without a call), a
method (`Class.method(...)`) or an attribute (`Class.attr`)."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "plot.md"

# Documented, not implemented yet (docs/design/plot.md §9). A milestone moves its names from
# here to sd.plot.__all__.
PLANNED = set()
# Figure methods documented for a later milestone (docs/design/plot.md §9)
PLANNED_METHODS = set()
# functions that make several cells themselves: no method of the same name (docs/api/plot.md §1.1)
NO_METHOD = {"Figure", "grid", "diagnostics", "pairs", "facet"}
# methods that draw over an existing figure only: no module function (docs/api/plot.md §6)
METHOD_ONLY = {"compare", "fit"}
LINE = re.compile(r"^(?:(?P<cls>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)(?:\((?P<sig>.*)\))?$")


def entries():
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


def test_exports_match_api_md():
    doc = {name for cls, name, _ in entries() if cls is None}
    exported = set(sd.plot.__all__)
    assert exported.isdisjoint(PLANNED)
    assert exported | PLANNED == doc
    for name in exported:
        assert hasattr(sd.plot, name)


def test_exported_signatures_match_api_md():
    doc = {name: sig for cls, name, sig in entries() if cls is None}
    for name in sd.plot.__all__:
        assert str(inspect.signature(getattr(sd.plot, name)))[1:-1] == doc[name], name


def test_figure_has_the_documented_methods_and_attributes():
    documented = set()
    for cls, name, sig in entries():
        if cls != "Figure" or name in PLANNED_METHODS:
            continue
        documented.add(name)
        member = inspect.getattr_static(sd.plot.Figure, name)
        if sig is None:
            assert isinstance(member, property), name
        else:
            text = str(inspect.signature(member))[1:-1]
            assert text == ("self, " + sig if sig else "self"), name
    public = {n for n in vars(sd.plot.Figure) if not n.startswith("_")} | {"__getitem__"}
    # a plotting function is also a method that adds to the figure (docs/api/plot.md §1.1)
    functions = {n for n in sd.plot.__all__ if n not in NO_METHOD}
    assert public == documented | functions and METHOD_ONLY <= documented
    for name in functions:
        params = str(inspect.signature(getattr(sd.plot, name)))[1:-1]
        assert str(inspect.signature(getattr(sd.plot.Figure, name)))[1:-1] == "self, " + params.removesuffix(
            ", **figure"
        ), name


def test_lines_are_unique_and_methods_belong_to_figure():
    seen = set()
    for cls, name, _ in entries():
        assert (cls, name) not in seen, f"documented twice: {cls}.{name}"
        seen.add((cls, name))
        assert cls in (None, "Figure"), f"{cls}.{name}: the only class is Figure"


def test_documented_signatures_follow_the_rules():
    """Data and main arguments are positional-only, options keyword-only (DESIGN §8.1); a
    function that makes a figure takes the figure options last."""
    for cls, name, sig in entries():
        if sig is None or name == "__getitem__":
            continue
        parts = [p.strip() for p in re.split(r",\s*(?![^()]*\))", sig) if p.strip()]
        if not parts:
            continue
        positional = parts[: parts.index("/")] if "/" in parts else []
        rest = parts[parts.index("/") + 1 :] if "/" in parts else parts
        if rest:
            assert rest[0] == "*", f"{name}: options must be keyword-only: {sig}"
        for p in rest[1:]:
            assert "=" in p or p == "**figure", f"{name}: option {p!r} needs a default"
        if "**figure" in parts:
            assert parts[-1] == "**figure" and cls is None, name
        assert len(positional) == len(set(positional))
