"""The Arb references of sd.optimize (tools/optimize_ref.py), shared by the tests and the
accuracy measurement (bench/accuracy/optimize.py):

    ref/scalar_roots.json    per root case of functions.py, the root of the function as written
                             located to one ULP: {"kind": "bracket", "a", "b"} (adjacent doubles
                             with f of opposite signs) or {"kind": "exact", "x"} (f is exactly 0)
    ref/scalar_minima.json   per minimize_scalar case, the local minimizers as [hi, lo] pairs:
                             "bracket" (inside the bracket's outer points) and "bounds"
    ref/derivatives.npz      per MGH case with factor 1, at x0, the gradient and the Jacobian
                             (layout: tests/reffile.py)

Each JSON file's "_provenance" says what computed it, like the npz files' fields.
"""

import json
import pathlib

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"


def _json(name):
    data = json.loads((REF / name).read_text())
    data.pop("_provenance", None)
    return data


def roots():
    """{case id: {"kind": "bracket", "a", "b"} or {"kind": "exact", "x"}}."""
    return _json("scalar_roots.json")


def minima():
    """{case id: {"bracket": [[hi, lo], ...], "bounds": [...]}}."""
    return _json("scalar_minima.json")


def derivatives():
    """[(case id, x0, (gradient hi, lo), (Jacobian hi, lo))] per MGH case."""
    r = reffile.read(REF / "derivatives.npz")
    g, J = r.methods["gradient"], r.methods["jacobian"]
    return [
        (str(r.inputs["id"][i]), r.inputs["x0"][i], (g.hi[i], g.lo[i]), (J.hi[i], J.lo[i])) for i in range(r.points)
    ]
