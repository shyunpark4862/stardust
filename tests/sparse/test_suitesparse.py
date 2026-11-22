"""The SuiteSparse matrices of decision SP20 (tools/sparse_suitesparse.py, docs/design/sparse.md
§7.2): their sizes are the collection's, the Accelerate LU wrapper solves the square ones
(LU/AMD, the same bits on every call), and the iterative solvers follow scipy on ten problems.
External tier: the files live outside the repository in build/sparse/suitesparse/ and each test
skips, naming the fetch command, when its file is missing."""

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


# SP5 (docs/design/sparse.md §3.7, §7.2): the iterative solvers on the real matrices, against scipy
# 1.18.1 live, b = ones (n,). Exact agreement (status and iteration count) where the problem is
# not chaotic: memplus' GMRES(20) takes 5,425 inner steps in both, cvxqp3's MINRES stops at the
# same step (its cond(A) estimate); bcsstk14's Jacobi-preconditioned CG within one step of scipy's
# (370), msc10848's within 8 of its 5,276 (5,272: over five thousand steps the count follows the
# last bits of the dot products, which are Accelerate's chunked ones since 2026-11-22, SP8).
# GMRES on young1c and LSQR / LSMR on well1850 within 20% of scipy's
# count (2,809 and 2,822 inner steps; 323 and 324, 338 and 338). BiCGSTAB is chaotic on these
# matrices, as on tests/sparse/test_iterative.py's random ones: the same status (memplus 901
# steps, scipy 1,166; young1c 342 and 394; sherman3 breaks down, rho = 0, at step 266 and 728).
# A converged result meets its test when the residual is recomputed.
ITERATIVE = [
    ("Hamm/memplus", "gmres", {}, 0),
    ("Hamm/memplus", "bicgstab", {}, -1),
    ("HB/young1c", "gmres", {}, None),
    ("HB/young1c", "bicgstab", {}, -1),
    ("HB/sherman3", "bicgstab", {}, -1),
    ("GHS_indef/cvxqp3", "minres", {}, 0),
    ("HB/bcsstk14", "cg", {"jacobi": True}, 1),
    ("Boeing/msc10848", "cg", {"jacobi": True}, 8),
    ("HB/well1850", "lsqr", {}, None),
    ("HB/well1850", "lsmr", {}, None),
]


@pytest.mark.usefixtures("reference_scipy")
@pytest.mark.parametrize(("name", "fn", "opts", "slack"), ITERATIVE, ids=[f"{n}-{f}" for n, f, _, _ in ITERATIVE])
def test_iterative_solvers_follow_scipy(name, fn, opts, slack):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    a = get(name)
    s = a.scipy("csr")
    indptr, indices, data = a.csr()
    A = sd.sparse.csr_array((sd.asarray(data), sd.asarray(indices), sd.asarray(indptr)), shape=a.shape)
    b = np.ones(a.shape[0], dtype=a.dtype)
    kw, skw = {}, {}
    if opts.get("jacobi"):
        Mi = sp.diags_array(1 / s.diagonal()).tocsr()
        kw["M"], skw["M"] = sd.sparse.csr_array(Mi), Mi
    r = getattr(sd.sparse, fn)(A, sd.asarray(b), **kw)
    if fn in ("lsqr", "lsmr"):
        ref = getattr(spla, fn)(s, b)
        status, count, mine = ref[1], ref[2], r.n_iter
    else:
        c = [0]
        extra = {"callback_type": "pr_norm"} if fn == "gmres" else {}
        _, status = getattr(spla, fn)(s, b, callback=lambda *_: c.__setitem__(0, c[0] + 1), **skw, **extra)
        count, mine = c[0], (r.n_inner if fn == "gmres" else r.n_iter)
    assert r.status == status
    if slack is None:
        assert abs(mine - count) <= 0.2 * count, (mine, count)
    elif slack >= 0:  # (-1: chaotic, the status only)
        assert abs(mine - count) <= slack, (mine, count)
    if r.converged and fn not in ("lsqr", "lsmr", "minres"):
        assert np.linalg.norm(b - s @ np.asarray(r.x)) <= 1.01e-5 * np.linalg.norm(b)


# SP6 (docs/design/sparse.md §3.8, §7.2): the eigensolvers on the real matrices, against scipy
# 1.18.1 live with the same v0 (sin(1..n)): the same status, the eigenvalues to 1e-9 relative,
# and the same number of OP x products (bcsstk14 106, msc10848 37, memplus 125, young1c 372; the
# shift-invert ones are compared by their values: scipy's inverse is its own SuperLU, and goodwin's
# eigenvalues nearest 0 (1e-5, far from normal) agree to 1e-6 relative, 1e-12 absolute, now that
# the shift-invert solve is a plain LU solve as scipy's is, SP63); svds against
# the dense SVD (well1850, 1850 x 712).
EIGEN = [
    ("HB/bcsstk14", "eigsh", {"k": 6}, 0),
    ("HB/bcsstk14", "eigsh", {"k": 6, "sigma": 0.0}, 0),
    ("Boeing/msc10848", "eigsh", {"k": 4}, 0),
    ("Hamm/memplus", "eigs", {"k": 6}, 0),
    ("HB/young1c", "eigs", {"k": 6}, 0),
    ("Goodwin/goodwin", "eigs", {"k": 4, "sigma": 0.0}, 0),
]


@pytest.mark.usefixtures("reference_scipy")
@pytest.mark.parametrize(("name", "fn", "kw", "slack"), EIGEN, ids=[f"{n}-{f}-{sorted(k)}" for n, f, k, _ in EIGEN])
def test_eigensolvers_follow_scipy(name, fn, kw, slack):
    import scipy.sparse.linalg as spla

    a = get(name)
    s = a.scipy("csr")
    indptr, indices, data = a.csr()
    A = sd.sparse.csr_array((sd.asarray(data), sd.asarray(indices), sd.asarray(indptr)), shape=a.shape)
    n = a.shape[0]
    v0 = np.sin(np.arange(1.0, n + 1)).astype(a.dtype)
    c = [0]

    def mv(x):
        c[0] += 1
        return s @ x

    op = s if "sigma" in kw else spla.LinearOperator(s.shape, matvec=mv, dtype=s.dtype)
    w = getattr(spla, fn)(op, v0=v0, return_eigenvectors=False, **kw)
    r = getattr(sd.sparse, fn)(A, v0=sd.asarray(v0), return_eigenvectors=False, **kw)
    assert r.converged
    got = np.asarray(r.eigenvalues)
    rtol = 1e-6 if name == "Goodwin/goodwin" else 1e-9
    np.testing.assert_allclose(np.sort_complex(got), np.sort_complex(w), rtol=rtol)
    if "sigma" not in kw:
        if slack is None:
            assert abs(r.n_op - c[0]) <= 0.2 * c[0], (r.n_op, c[0])
        else:
            assert abs(r.n_op - c[0]) <= slack, (r.n_op, c[0])


def test_svds_on_well1850():
    a = get("HB/well1850")
    indptr, indices, data = a.csr()
    A = sd.sparse.csr_array((sd.asarray(data), sd.asarray(indices), sd.asarray(indptr)), shape=a.shape)
    s = np.linalg.svd(a.dense(), compute_uv=False)
    for which, ref in (("LM", np.sort(s[:6])), ("SM", np.sort(s[-6:]))):
        r = sd.sparse.svds(A, k=6, which=which)
        assert r.converged
        np.testing.assert_allclose(np.asarray(r.S), ref, rtol=1e-10)
