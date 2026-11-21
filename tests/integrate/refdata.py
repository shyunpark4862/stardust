"""The Arb references of sd.integrate (tools/integrate_ref.py), shared by the tests and the
accuracy measurement (bench/accuracy/integrate.py):

    ref/integrals.json   per one-dimensional integral of problems.INTEGRALS: {"value": [hi, lo],
                         "method": the method stored, "checks": the methods that agreed with it,
                         "digest": the problem's definition (tools/integrate_ref.integral_digest)}
    ref/series.json      per series of problems.SERIES, the same fields (closed forms)
    ref/multiple.json    per multiple integral of problems.MULTI
    ref/vector.json      per vector-valued integrand of problems.VECTOR: "value" a list of pairs

Each file's "_provenance" says what computed it (the npz files' fields of tests/reffile.py).
"""

import json
import pathlib

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"


def _json(path, keep_provenance=False):
    data = json.loads(path.read_text())
    if not keep_provenance:
        data.pop("_provenance", None)
    return data


def integrals(keep_provenance=False):
    return _json(REF / "integrals.json", keep_provenance)


def series(keep_provenance=False):
    return _json(REF / "series.json", keep_provenance)


def multiple(keep_provenance=False):
    return _json(REF / "multiple.json", keep_provenance)


def vector(keep_provenance=False):
    return _json(REF / "vector.json", keep_provenance)


def value(entry):
    """The reference as a float (hi; lo is below its rounding)."""
    return float(entry["value"][0])


def data(name):
    """A record file of tests/integrate/data (scipy_*.json, fortran_*.json) as a dict."""
    return json.loads((DATA / name).read_text())
