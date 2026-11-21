"""API.md and docs/api/array.md are the contract -- every name it lists must exist, and
the package must export nothing it does not list."""

import re
from pathlib import Path

import stardust as sd

from .._api_doc import signature_blocks

# The array names are in docs/api/array.md; the library-wide ones (thread count) in API.md.
ROOT = Path(__file__).parents[2]
API = (ROOT / "API.md").read_text() + "\n" + (ROOT / "docs" / "api" / "array.md").read_text()
METHODS = {
    "reshape",
    "astype",
    "copy",
    "tolist",
    "sum",
    "prod",
    "min",
    "max",
    "mean",
    "var",
    "std",
    "argmin",
    "argmax",
    "any",
    "all",
}
ATTRIBUTES = {"dtype", "shape", "ndim", "size", "strides", "itemsize", "nbytes", "base", "T", "mT"}
CONSTANTS = {"e", "pi", "inf", "nan", "newaxis", "bool", "int64", "float64", "complex128", "string", "ndarray"}


def api_names():
    names = set()
    for block in signature_blocks(API):
        for line in block:
            code = line.split("#")[0]
            names |= set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", code))
            if "(" not in code:
                names |= {w for w in code.split() if re.fullmatch(r"[a-z_][a-z0-9_]*", w)}
    # Most methods are also module functions (docs/api/array.md §10.2); copy and tolist are methods only.
    return names - {"copy", "tolist"} - ATTRIBUTES


def test_every_documented_function_exists():
    missing = sorted(n for n in api_names() if not hasattr(sd, n))
    assert missing == []


def test_every_documented_ndarray_member_exists():
    x = sd.zeros((2, 2))
    assert sorted(m for m in METHODS | ATTRIBUTES if not hasattr(x, m)) == []


def test_nothing_undocumented_is_exported():
    assert sorted(set(sd.__all__) - api_names() - CONSTANTS) == []
    assert set(sd.__all__) <= set(dir(sd))


def test_method_docstrings_have_the_module_functions_signature():
    # Each public method's __doc__ starts with the module function's signature, self for x
    # (no nanobind type annotations): sum(self, /, *, axis=None, dtype=None, keepdims=False).
    x = sd.zeros((2, 2))
    public = sorted(m for m in dir(x) if not m.startswith("_") and callable(getattr(x, m)))
    assert public == sorted(METHODS)
    for m in public:
        first = getattr(x, m).__doc__.split("\n")[0]
        assert ":" not in first and "->" not in first, first
        mod = getattr(sd, m, None)
        if mod is not None:  # reshape, astype and the reductions: the function's, self for x
            want = mod.__doc__.split("\n")[0].replace(f"{m}(x, ", f"{m}(self, ", 1)
        else:  # copy, tolist: methods only
            want = f"{m}(self, /)"
        assert first == want, (m, first)
