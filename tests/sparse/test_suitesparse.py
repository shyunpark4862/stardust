"""The SuiteSparse matrices of decision SP20 (tools/sparse_suitesparse.py, docs/design/sparse.md
§7.2): their sizes are the collection's, and the Accelerate LU wrapper solves the square ones
(LU/AMD, the same bits on every call). External tier: the files live outside the repository in
build/sparse/suitesparse/ and each test skips, naming the fetch command, when its file is
missing."""

import importlib.util
import pathlib

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import matrices as M

pytestmark = pytest.mark.external

ROOT = pathlib.Path(__file__).parents[2]


def _tool():
    spec = importlib.util.spec_from_file_location("sparse_suitesparse", ROOT / "tools" / "sparse_suitesparse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _tool()


def get(name):
    if not T.available(name):
        pytest.skip(f"{name} is not fetched: .venv/bin/python tools/sparse_suitesparse.py {name}")
    shape, r, c, v = T.load(name)
    return M.Matrix(name, shape, r, c, v, frozenset())


@pytest.mark.parametrize(("name", "kind", "m", "n", "nnz"), T.MATRICES)
def test_sizes_are_the_collections(name, kind, m, n, nnz):
    a = get(name)
    assert a.shape == (m, n)
    assert a.canonical()[0].size in (nnz, nnz + (a.data == 0).sum())  # cvxqp3 lists explicit zeros apart
    assert (a.dtype == np.complex128) == ("complex" in kind)


@pytest.mark.parametrize("name", [m[0] for m in T.MATRICES if m[2] == m[3]])
def test_lu_solves_and_repeats(name):
    a = get(name)
    indptr, indices, data = a.csc()
    b = M.rhs(a)

    def solve():
        return np.asarray(sd.sparse._core._lu_solve(a.shape, sd.asarray(indptr), sd.asarray(indices), sd.asarray(data),
                                                     sd.asarray(b)))  # fmt: skip

    x = solve()
    assert bits_equal(solve(), x)
    if np.isfinite(x).all():
        s = a.scipy("csr")
        r = b - s @ x
        scale = abs(s).sum(axis=1).max() * np.abs(x).max() + np.abs(b).max()
        assert np.abs(r).max() <= 1e-10 * scale


# SP4 (docs/design/sparse.md §3.6, §7.2): the public solves on the real matrices. sherman3 and
# cvxqp3 have kappa_1 of about 6.9e16 and 2.2e16 (scipy's SuperLU and the estimate agree), past
# 1/eps: by decision SP62 they are LinAlgError, where scipy returns a solution without a word.
TOO_ILL = {"HB/sherman3", "GHS_indef/cvxqp3"}
SQUARE = [m[0] for m in T.MATRICES if m[2] == m[3]]


@pytest.mark.parametrize("name", [n for n in SQUARE if n not in TOO_ILL])
def test_spsolve_within_the_contract(name):
    from . import refdata as R

    a = get(name)
    indptr, indices, data = a.csc()
    A = sd.sparse.csc_array((sd.asarray(data), sd.asarray(indices), sd.asarray(indptr)), shape=a.shape)
    b = M.rhs(a, 2)
    lu = sd.sparse.splu(A)
    s = a.scipy("csr")
    for trans, op in (("N", s), ("T", s.T), ("H", s.conj().T)):
        x = np.asarray(lu.solve(sd.asarray(b), trans=trans))
        assert R.backward_error(op, x, b) <= R.C_BACKWARD, trans
    assert bits_equal(sd.sparse.spsolve(A, sd.asarray(b)), lu.solve(sd.asarray(b)))
    assert lu._factor.core.cond * R.EPS < 1


@pytest.mark.parametrize("name", sorted(TOO_ILL))
def test_condition_past_one_over_eps_is_singular(name):
    a = get(name)
    indptr, indices, data = a.csc()
    A = sd.sparse.csc_array((sd.asarray(data), sd.asarray(indices), sd.asarray(indptr)), shape=a.shape)
    with pytest.raises(sd.linalg.LinAlgError, match=r"^spsolve\(\): singular matrix$"):
        sd.sparse.spsolve(A, sd.asarray(M.rhs(a)))
