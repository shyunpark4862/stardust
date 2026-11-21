"""The Arb references of sd.differentiate (tools/differentiate_ref.py), shared by the tests and the
accuracy measurement (bench/accuracy/differentiate.py):

    ref/derivatives.json   per "<function>/<x!r>" (problems.ELEMENTARY at its points and the
                           EDGES' points): {"df": [hi, lo], "ddf": [hi, lo]}
    ref/mgh.npz            per MGH case with factor 1 at x0: the Jacobian of F, the gradient of
                           sum F_i^2 (jacobian's scalar f) and, for n <= 10, its Hessian (layout:
                           tests/reffile.py)

The JSON file's "_provenance" says what computed it, like the npz file's fields.
"""

import json
import pathlib

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"


def derivatives(keep_provenance=False):
    d = json.loads((REF / "derivatives.json").read_text())
    if not keep_provenance:
        d.pop("_provenance", None)
    return d


def mgh():
    """{case id: {"x0", "jacobian": (hi, lo), "gradient": (hi, lo), "hessian": (hi, lo) or None}}."""
    r = reffile.read(REF / "mgh.npz")
    J, H, G = r.methods["jacobian"], r.methods["hessian"], r.methods["gradient"]
    out = {}
    for i in range(r.points):
        out[str(r.inputs["id"][i])] = {"x0": r.inputs["x0"][i], "jacobian": (J.hi[i], J.lo[i]),
                                       "gradient": (G.hi[i], G.lo[i]), "hessian": None}  # fmt: skip
    for k, i in enumerate(H.index):
        out[str(r.inputs["id"][int(i)])]["hessian"] = (H.hi[k], H.lo[k])
    return out


def data(name):
    return json.loads((DATA / name).read_text())
