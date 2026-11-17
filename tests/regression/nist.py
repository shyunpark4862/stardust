"""The NIST StRD linear least squares problems (tests/regression/data/*.dat, from
https://www.itl.nist.gov/div898/strd/lls/): data as decimal strings and the certified
values (docs/design/regression.md §6)."""

import os
import re

DATA = os.path.join(os.path.dirname(__file__), "data")

# name -> (degree of the polynomial in x, or None for the columns as given; intercept)
MODELS = {
    "Norris": (1, True), "Pontius": (2, True), "NoInt1": (1, False), "NoInt2": (1, False),
    "Filip": (10, True), "Longley": (None, True),
    "Wampler1": (5, True), "Wampler2": (5, True), "Wampler3": (5, True), "Wampler4": (5, True), "Wampler5": (5, True),
}
NAMES = list(MODELS)


def load(name):
    """dict: y and x as lists of decimal strings (x: rows), degree, intercept, and the
    certified params, bse, residual standard deviation (rsd) and R^2 as strings."""
    with open(os.path.join(DATA, name + ".dat"), encoding="ascii") as fh:
        lines = fh.read().splitlines()
    head = "\n".join(lines[:10])
    a, b = map(int, re.search(r"Data\s+\(lines (\d+) to (\d+)\)", head).groups())
    rows = [ln.split() for ln in lines[a - 1:b] if ln.strip()]
    y = [r[0] for r in rows]
    x = [r[1:] for r in rows]
    params, bse = [], []
    rsd = rsq = None
    for ln in lines:
        m = re.match(r"\s+B(\d+)\s+(\S+)\s+(\S+)\s*$", ln)
        if m:
            params.append(m.group(2))
            bse.append(m.group(3))
        m = re.match(r"\s+Standard Deviation\s+(\S+)", ln)
        if m:
            rsd = m.group(1)
        m = re.match(r"\s+R-Squared\s+(\S+)", ln)
        if m:
            rsq = m.group(1)
    degree, intercept = MODELS[name]
    return {"y": y, "x": x, "degree": degree, "intercept": intercept,
            "params": params, "bse": bse, "rsd": rsd, "rsquared": rsq}


def design_strings(d):
    """The design as the user would give it: the columns (powers of x for a polynomial
    model) without the constant one, as decimal strings for x and floats rounded once per
    entry for the powers: (y floats, X float rows)."""
    y = [float(v) for v in d["y"]]
    if d["degree"] is None:
        X = [[float(v) for v in r] for r in d["x"]]
    else:
        X = [[float(r[0]) ** j for j in range(1, d["degree"] + 1)] for r in d["x"]]
    return y, X


def exact_design(d):
    """Callables for tools/regref.ols giving the data as arbs from the decimal strings at the
    working precision, with polynomial columns as powers in Arb (what NIST certifies)."""
    from flint import arb

    y = lambda: [arb(v) for v in d["y"]]
    if d["degree"] is None:
        X = lambda: [[arb(v) for v in r] for r in d["x"]]
    else:
        X = lambda: [[arb(r[0]) ** j for j in range(1, d["degree"] + 1)] for r in d["x"]]
    return y, X
