"""docs/api/special.md and the module agree: every name its signature blocks define is
either exported or still planned (a later milestone), nothing else is exported, and the
keyword-only options of the exported functions are the documented ones."""

import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "special.md"

# Documented but not implemented yet: each milestone (ROADMAP.md, S1-S8) moves its
# functions from here to sd.special.__all__. Empty after S8.
PLANNED = {
    # S1
    "expit", "logit", "log_expit", "xlogy", "xlog1py", "entr", "rel_entr",
    # S2
    "erf", "erfc", "erfcx", "erfinv", "erfcinv", "ndtr", "log_ndtr", "ndtri",
    # S3
    "gamma", "gammaln", "rgamma", "digamma", "beta", "betaln", "binom", "zeta", "polygamma",
    # S4
    "gammainc", "gammaincc", "betainc", "betaincc",
    "gammaincinv", "gammainccinv", "betaincinv", "betainccinv",
    # S5
    "logsumexp", "softmax", "log_softmax", "expi", "expn", "ellipk", "ellipe",
    # S6
    "i0", "i0e", "i1", "i1e", "j0", "j1", "y0", "y1", "k0", "k0e", "k1", "k1e",
    "jv", "yv", "iv", "kv", "jve", "yve", "ive", "kve", "airy",
    # S7
    "hyp1f1", "hyp2f1", "lambertw",
    # S8
    "wofz", "loggamma",
}


def documented():
    """{name: signature text} for every `name(...)` line in the signature blocks."""
    names = {}
    for block in signature_blocks(API.read_text()):
        for line in block:
            m = re.match(r"([A-Za-z_0-9]+)\((.*?)\)\s*(#.*)?$", line.strip())
            if m:
                names[m.group(1)] = m.group(2)
    return names


def test_exports_match_api_md():
    doc = set(documented())
    exported = set(sd.special.__all__)
    assert exported.isdisjoint(PLANNED)
    assert exported | PLANNED == doc
    for name in exported:
        assert hasattr(sd.special, name)


def test_no_undocumented_public_names():
    # the native submodule holds the public functions and private (underscore) test helpers only
    public = {n for n in dir(sd.special._core) if not n.startswith("_")}
    assert public == set(sd.special.__all__)


def test_keyword_options_match_api_md():
    # the names after `*` in docs/api/special.md are exactly the keyword-only parameters
    for name, sig in documented().items():
        if name not in sd.special.__all__:
            continue
        doc_kw = set()
        if "*" in sig:
            for part in sig.split("*", 1)[1].split(","):
                part = part.strip()
                if part:
                    doc_kw.add(part.split("=")[0].strip())
        assert keyword_only(getattr(sd.special, name)) == doc_kw, name


def keyword_only(f):
    """The keyword-only parameter names of a nanobind function, from its
    __nb_signature__ ('def f(arg0: object, /, *, out: object = \\0) -> object')."""
    text = f.__nb_signature__[0][0]
    params = text[text.index("(") + 1 : text.rindex(")")]
    if "*," not in params and not params.rstrip().endswith("*"):
        return set()
    after = params.split("*,", 1)[1] if "*," in params else ""
    return {p.split(":")[0].strip() for p in after.split(",") if p.strip()}
