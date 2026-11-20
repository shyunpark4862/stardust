"""The residual contract (docs/design/linalg.md §13) on the matrices of tests/linalg/ref, judged
by tests/linalg/refdata.py's residual(), as the accuracy measurement (bench/accuracy/linalg.py)
judges it: square, symmetric, positive definite and rectangular matrices at kappa_2 = 1 to 1e12."""

import pytest

from . import refdata


@pytest.mark.parametrize("fn", refdata.RESIDUAL)
def test_residual_contract(fn):
    sd = refdata.SD()
    for g in refdata.residual_matrices(fn):
        backward, orth = refdata.residual(fn, g["a"], g.get("b"), refdata.call(sd, fn, g["a"], g.get("b")))
        worst = max(backward, orth or 0.0)
        assert worst <= refdata.RESIDUAL_C, (fn, g["a"].shape, float(g.get("kappa", 0.0)), worst)
