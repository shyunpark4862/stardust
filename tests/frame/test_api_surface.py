"""docs/api/frame.md and sd.frame agree: every public name its signature blocks
list exists, nothing public is missing from them, and each signature matches:
parameter names, kinds and defaults (positional-only names are not API, so
only their count and defaults count)."""

import inspect
import pathlib
import re

import stardust as sd
from stardust.frame._groupby import GroupBy

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "frame.md"
MODULE = "frame"  # the owner of module-level names (written bare in the document)
OWNERS = {
    "Series": sd.frame.Series,
    "DataFrame": sd.frame.DataFrame,
    "Index": sd.frame.Index,
    "RangeIndex": sd.frame.RangeIndex,
    "GroupBy": GroupBy,
}


def listed():
    """{(owner, name): signature text or None for an attribute} from the signature blocks.
    `name(...)` and `name` are sd.frame's own; `Owner.name(...)` and `Owner.name` a member."""
    out = {}
    for block in signature_blocks(API.read_text()):
        for line in block:
            line = line.split("#")[0].strip()
            if not line:
                continue
            m = re.fullmatch(r"(?:(\w+)\.)?(\w+)(?:\((.*)\))?", line)
            assert m, f"unreadable line in docs/api/frame.md: {line!r}"
            key = (m.group(1) or MODULE, m.group(2))
            assert key not in out, f"listed twice: {line}"
            out[key] = m.group(3)
    return out


def public(owner):
    if owner == MODULE:
        return set(sd.frame.__all__)
    return {n for n in dir(OWNERS[owner]) if not n.startswith("_")}


def test_every_public_name_is_listed_and_exists():
    entries = listed()
    for owner in [MODULE, *OWNERS]:
        names = {n for o, n in entries if o == owner}
        assert names == public(owner), owner
    for n in sd.frame.__all__:
        assert hasattr(sd.frame, n)


def test_attributes_and_methods_match():
    for (owner, name), sig in listed().items():
        obj = getattr(sd.frame if owner == MODULE else OWNERS[owner], name)
        if sig is None:  # an attribute: a property on a class, or a module constant (NA)
            if owner != MODULE:
                assert isinstance(inspect.getattr_static(OWNERS[owner], name), property), (owner, name)
        else:
            assert callable(obj), (owner, name)


def documented_params(sig):
    """[(name, kind, default text)] from a documented signature such as 'by, *, ascending=True'."""
    params, kind = [], inspect.Parameter.POSITIONAL_OR_KEYWORD
    for part in split_top(sig):
        if part == "/":
            params = [(n, inspect.Parameter.POSITIONAL_ONLY, d) for n, _, d in params]
        elif part == "*":
            kind = inspect.Parameter.KEYWORD_ONLY
        elif part.startswith("**"):
            params.append((part[2:], inspect.Parameter.VAR_KEYWORD, None))
        else:
            name, _, default = part.partition("=")
            params.append((name.strip(), kind, default.strip() or None))
    return params


def split_top(sig):
    """Split on commas outside brackets and quotes."""
    parts, depth, quoted, cur = [], 0, False, ""
    for ch in sig:
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch in "([":
            depth += 1
        elif not quoted and ch in ")]":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def actual_params(f, method):
    """The same triples from the function itself (nanobind functions through __nb_signature__)."""
    if hasattr(f, "__nb_signature__"):
        text = f.__nb_signature__[0][0]
        inner = text[text.index("(") + 1 : text.rindex(")")]
        params = documented_params(re.sub(r":\s*[\w\[\], ]+?(?=\s*(=|,|$))", "", inner)) if inner else []
        return params
    ps = list(inspect.signature(f).parameters.values())
    if method:
        ps = ps[1:]  # self
    out = []
    for p in ps:
        if p.default is inspect.Parameter.empty:
            d = None
        elif type(p.default) is object:
            d = "<없음>"  # a private "not given" sentinel
        else:
            d = repr(p.default).replace("'", '"')
        out.append((p.name, p.kind, d))
    return out


def test_signatures_match():
    for (owner, name), sig in listed().items():
        if sig is None:
            continue
        if owner == MODULE:
            f = getattr(sd.frame, name)
            if isinstance(f, type):
                f, method = f.__init__, True
            else:
                method = False
        else:
            f, method = getattr(OWNERS[owner], name), True
        doc, act = documented_params(sig), actual_params(f, method)
        strip = lambda ps: [(None if k == inspect.Parameter.POSITIONAL_ONLY else n, k, d) for n, k, d in ps]  # noqa: E731
        assert strip(doc) == strip(act), f"{owner}.{name}: docs/api/frame.md ({sig}) vs code {act}"
