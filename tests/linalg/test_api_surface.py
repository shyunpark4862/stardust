"""API.md and the module agree (M11): every name API.md's code blocks define is
exported, nothing else is, and the keyword-only options are the documented ones."""

import pathlib
import re

import stardust as sd

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "linalg.md"


def documented():
    """{name: signature text} for every `name(...)` line in API.md's code blocks."""
    names = {}
    inside = False
    for line in API.read_text().splitlines():
        if line.startswith("```"):
            inside = not inside
            continue
        m = re.match(r"([A-Za-z_]+)\((.*?)\)\s*(#.*)?$", line.strip()) if inside else None
        if m:
            names[m.group(1)] = m.group(2)
    return names


def test_exports_match_api_md():
    assert set(documented()) == set(sd.linalg.__all__)
    for name in sd.linalg.__all__:
        assert hasattr(sd.linalg, name)


def test_keyword_options_match_api_md():
    # the names after `*` in API.md are exactly the keyword-only parameters of the function
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
    return {p.split(":")[0].strip() for p in after.split(",") if p.strip()}
