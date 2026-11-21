"""docs/api/differentiate.md and the module agree: the signature blocks name scipy.differentiate's
three functions (decision C2) and the result class, the module exports them with the documented
signatures (C2), and the signatures follow the area's naming decisions (C4, C11, C12, C14)."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "differentiate.md"
FUNCTIONS = {"derivative", "jacobian", "hessian"}  # milestone C2
RESULTS = {"DifferentiateResult"}


def documented():
    names, bare = {}, set()
    for block in signature_blocks(API.read_text()):
        for line in block:
            line = line.split("#")[0].strip()
            m = re.fullmatch(r"([A-Za-z_0-9]+)\((.*)\)", line)
            if m:
                names[m.group(1)] = m.group(2)
            elif re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", line):
                bare.add(line)
    return names, bare


def test_exports_match_the_document():
    funcs, bare = documented()
    assert set(funcs) == FUNCTIONS and bare == RESULTS
    assert set(sd.differentiate.__all__) == FUNCTIONS | RESULTS


def test_signatures_match_the_document():
    funcs, _ = documented()
    for name, sig in funcs.items():
        got = str(inspect.signature(getattr(sd.differentiate, name)))[1:-1]
        assert got == sig, (name, got, sig)


def test_signatures_follow_the_naming_decisions():
    funcs, _ = documented()
    for name, sig in funcs.items():
        params = [p.strip().split("=")[0].strip() for p in sig.split(",")]
        assert params[:2] == ["f", "x"] and params[2] == "/", name
        assert ("args" in params) == (name == "derivative"), name  # C4: the elementwise family only
        for gone in ("kwargs", "callback", "preserve_shape", "tolerances", "maxiter"):
            assert gone not in params, (name, gone)
        assert "atol" in params and "rtol" in params and "max_iter" in params, name
