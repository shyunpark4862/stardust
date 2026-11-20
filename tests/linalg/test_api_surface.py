"""docs/api/linalg.md and the module agree: every name its signature blocks define is
exported, nothing else is, the keyword-only options are the documented ones, and help()
shows the documented signature."""

import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "linalg.md"


def documented():
    """{name: signature text} for every `name(...)` line in the signature blocks."""
    names = {}
    for block in signature_blocks(API.read_text()):
        for line in block:
            m = re.match(r"([A-Za-z_]+)\((.*?)\)\s*(#.*)?$", line.strip())
            if m:
                names[m.group(1)] = m.group(2)
    return names


def test_exports_match_api_md():
    assert set(documented()) == set(sd.linalg.__all__)
    for name in sd.linalg.__all__:
        assert hasattr(sd.linalg, name)


def test_keyword_options_match_api_md():
    # the names after `*` in docs/api/linalg.md are exactly the keyword-only parameters of the function
    for name, sig in documented().items():
        if name == "LinAlgError":
            continue
        doc_kw = set()
        if "*" in sig:
            for part in sig.split("*", 1)[1].split(","):
                part = part.strip()
                if part:
                    doc_kw.add(part.split("=")[0].strip())
        assert keyword_only(getattr(sd.linalg, name)) == doc_kw, name


def keyword_only(f):
    """The keyword-only parameter names of a nanobind function, from its
    __nb_signature__ ('def f(arg0: object, /, *, axis: object = \\0) -> object')."""
    text = f.__nb_signature__[0][0]
    params = text[text.index("(") + 1 : text.rindex(")")]
    if "*," not in params and not params.rstrip().endswith("*"):
        return set()
    after = params.split("*,", 1)[1] if "*," in params else ""
    return {p.split(":")[0].split("=")[0].strip() for p in after.split(",") if p.strip()}


def test_signatures_match_api_md():
    # the bindings' nb::sig (the arguments stay unnamed, so positional-only): help() shows
    # the documented names, not arg0, arg1
    for name, sig in documented().items():
        if name == "LinAlgError":
            continue
        f = getattr(sd.linalg, name)
        assert f.__nb_signature__[0][0] == f"def {name}({sig})", name
        assert f.__doc__.splitlines()[0] == f"{name}({sig})", name
