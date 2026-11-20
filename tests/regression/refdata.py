"""Arb references for the linear models' tests (tools/regression_ref.py from tools/regref.py;
docs/design/regression.md §6).

tests/regression/ref/linear.json holds, per problem, the inputs as doubles (y, X rows without
the constant column, weights, intercept, cluster labels) and regref.ols's values as [hi, lo]
pairs: random problems (OLS with and without the intercept, WLS with zero weights, an
ill-conditioned one with a large residual) and the 11 NIST StRD problems as a user gives them
(decimal data and powers rounded to double once). The random problems have cluster labels
(in wls_zero one cluster has only zero-weight rows, so it is not counted); HAC uses the
default lags. Regenerate with

    .venv/bin/python tools/regression_ref.py linear
"""

import json
import pathlib

REF = pathlib.Path(__file__).parent / "ref" / "linear.json"


def load():
    """{problem: {...}} (the file's "_provenance" left out)."""
    with open(REF) as fh:
        data = json.load(fh)
    data.pop("_provenance", None)
    return data


def value(hl):
    """A reference [hi, lo] as one float."""
    return hl[0] + hl[1]


def rel_err(got, hl):
    """|got - ref| / |ref| with the reference in double-double."""
    hi, lo = hl
    ref = hi + lo
    d = (got - hi) - lo
    return abs(d) / abs(ref) if ref != 0 else abs(d)
