"""docs/api/stats.md and the module agree: the method signatures of the distribution blocks
(§2, §2.5, §3, §3.1), the descriptive statistics' signatures (§4), the classes of the §2.1
table with their keyword names, and the exports."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "stats.md"
S = sd.stats


def _blocks():
    return ["\n".join(b) for b in signature_blocks(API.read_text())]


def _norm(sig):
    return re.sub(r"\s+", "", sig).replace("e-0", "e-")  # (repr's 1e-06 is the document's 1e-6)


def _params(sig_text):
    """The parameter list of a documented call without `self`."""
    return _norm(sig_text)


def _actual(f):
    sig = inspect.signature(f)
    ps = [p for p in sig.parameters.values() if p.name != "self"]
    return _norm(str(sig.replace(parameters=ps))[1:-1])


def _methods(block):
    """{name: parameters} for every `X.name(...)` in a block (comments dropped)."""
    out = {}
    for line in block.splitlines():
        line = line.split("#", 1)[0]
        for m in re.finditer(r"X\.(\w+)\(([^)]*)\)", line):
            out[m.group(1)] = m.group(2)
    return out


def test_univariate_method_signatures():
    blocks = _blocks()
    doc = _methods(blocks[0]) | _methods(blocks[1])  # §2 and §2.5
    cont, disc = S.Normal, S.Poisson
    for name, params in doc.items():
        assert _actual(getattr(cont, name)) == _params(params), name
    # the discrete distributions' density methods: the same shape as pdf's
    assert _actual(disc.pmf) == _params(doc["pdf"]).replace("x", "k", 1)
    assert _actual(disc.logpmf) == _params(doc["logpdf"]).replace("x", "k", 1)
    # every public method of a univariate distribution is documented
    public = {n for n in dir(cont) if not n.startswith("_") and callable(getattr(cont, n))}
    assert public == set(doc), public ^ set(doc)


def test_multivariate_and_matrix_method_signatures():
    blocks = _blocks()
    mv, mat = _methods(blocks[2]), _methods(blocks[3])
    for cls in (S.MultivariateNormal, S.MultivariateT):
        for name, params in mv.items():
            assert _actual(getattr(cls, name)) == _params(params), (cls.__name__, name)
    for name in ("pmf", "logpmf", "mean", "covariance", "entropy", "sample"):
        assert hasattr(S.Multinomial, name)
    for name in ("pdf", "logpdf", "mean", "covariance", "entropy", "sample"):
        assert hasattr(S.Dirichlet, name)
    for cls in (S.Wishart, S.InverseWishart):
        for name, params in mat.items():
            assert _actual(getattr(cls, name)) == _params(params), (cls.__name__, name)
    for name in ("pdf", "logpdf", "mean", "entropy", "sample"):
        assert hasattr(S.MatrixNormal, name)
    assert not hasattr(S.MatrixNormal, "variance") and not hasattr(S.Wishart, "var")


def test_descriptive_signatures():
    block = _blocks()[4]
    text = re.sub(r"#[^\n]*", "", block)
    text = re.sub(r",\s*\n\s*", ", ", text)  # a signature continued on the next line
    found = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\w+)\((.*)\)\s*$", line)
        if m:
            found[m.group(1)] = m.group(2)
    assert set(found) == {
        "describe",
        "skew",
        "kurtosis",
        "moment",
        "zscore",
        "rankdata",
        "sem",
        "iqr",
        "mode",
        "entropy",
        "differential_entropy",
    }
    for name, params in found.items():
        assert _actual(getattr(S, name)) == _params(params), name


def test_distribution_table():
    rows = re.findall(r"^\| [^|]+ \| `(\w+)\(([^)]*)\)`", API.read_text(), re.M)
    assert len(rows) == 31
    for name, params in rows:
        cls = getattr(S, name)
        doc = [p.split("=")[0].strip() for p in params.split(",") if p.strip() not in ("*", "")]
        sig = inspect.signature(cls)
        assert all(p.kind is p.KEYWORD_ONLY for p in sig.parameters.values()), name
        assert list(sig.parameters) == doc, (name, doc)


def test_exports():
    names = set(S.__all__)
    for n in names:
        assert hasattr(S, n), n
    assert {"DescribeResult", "ModeResult"} <= names and "CdfResult" not in names
