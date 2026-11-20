"""Reference values of the array functions (tools/array_ref.py; layout: tests/reffile.py) and
their contract measure, shared by the accuracy measurement (bench/accuracy/array.py) and
tests/array/test_reference.py.

The element-wise contract (docs/api/array.md): at every point the result is within ULP_BOUND
of the exact value rounded to a double (Arb's hi), in ulp_distance (bench/accuracy/common.py:
the number of doubles between the two). The reductions have no documented bound.

tests/array/ref/<function>.npz: the float64 element-wise functions on vForce, pow and the
modulus of a complex (cabs), one method each at the inputs x (atan2: y, x; pow: x, y; cabs:
re, im), the exact function at the double inputs as hi + lo.

tests/array/ref/reductions.npz: one point per reduction case, the methods sum, mean, var, std,
prod, vecdot and matmul. Inputs up to 200,000 elements are stored (x, y); larger ones are
regenerated from x_kind, x_shape and x_seed and checked against x_sha256 (an empty x_kind:
stored). axis and correction are -1 where the case has none.
"""

import pathlib

import numpy as np

from bench.accuracy.common import ulps

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
ELEMENTWISE = (
    "exp", "log", "log2", "log10", "sin", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "pow", "cabs",
)  # fmt: skip
REDUCTIONS = ("sum", "mean", "var", "std", "prod", "vecdot", "matmul")

# The largest ulp_distance to the correctly rounded value at the reference points, per function:
# the maximum measured over the reference points (vForce for the fifteen, libm's pow, sd's modulus
# of a complex) plus 2.
ULP_BOUND = {
    "exp": 3, "log": 4, "log2": 4, "log10": 4, "sin": 5, "asin": 4, "acos": 3, "atan": 3, "atan2": 3,
    "sinh": 4, "cosh": 4, "tanh": 4, "asinh": 4, "acosh": 4, "atanh": 5, "pow": 3, "cabs": 3,
}  # fmt: skip


def elementwise(name):
    """({input name: values}, hi, lo) of an element-wise function's file."""
    r = reffile.read(REF / f"{name}.npz")
    m = r.methods[name]
    return r.inputs, m.hi, m.lo


def reduction_groups():
    """The reduction cases as dicts: func, label, the stored inputs x, y (or their x_kind,
    x_shape, x_seed, x_sha256 to regenerate them), axis and correction where the case has
    them, and hi, lo, cond (the reference and its condition number per output element)."""
    r = reffile.read(REF / "reductions.npz")
    I = r.inputs
    out = []
    for i in range(r.points):
        g = {"func": I["func"][i], "label": I["label"][i]}
        for v in ("x", "y"):
            if str(I[f"{v}_kind"][i]):
                for k in ("kind", "shape", "seed", "sha256"):
                    g[f"{v}_{k}"] = I[f"{v}_{k}"][i]
            elif I[v][i].size:
                g[v] = I[v][i]
        for k in ("axis", "correction"):
            if I[k][i] >= 0:
                g[k] = I[k][i]
        m = r.methods[str(I["func"][i])]
        j = int(np.flatnonzero(m.index == i)[0])
        g["hi"], g["lo"], g["cond"] = m.hi[j], m.lo[j], m.fields["cond"][j]
        out.append(g)
    return out


def elementwise_contract(name, got, hi):
    """(error, bound): the ulp_distance of each value to the reference rounded to a double (inf
    where one of them is NaN or an unequal infinity), and the function's ULP_BOUND."""
    return ulps(got, hi), float(ULP_BOUND[name])
