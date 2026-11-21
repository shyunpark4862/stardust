"""The iterative solvers of sd.sparse (milestone SP5; docs/api/sparse.md §6, docs/design/sparse.md
§3.7, §4): cg, bicgstab, gmres, minres, lsqr, lsmr.

What the tests hold them to:
  - scipy 1.18.1's algorithms and stopping tests: the status and the iteration count of every
    recorded problem (tests/sparse/data/scipy.json) and, live, the residual norms step by step
    (gmres' preconditioned residual estimates, cg's, bicgstab's and minres' against the true
    residual of scipy's iterates, lsqr's and lsmr's r1norm / normr by stopping scipy after each
    step), preconditioned, restarted, shifted and damped problems, x0, breakdowns, the limits;
  - the contract (docs/design/sparse.md §4): a converged result is a point that meets the test
    again when its residual is recomputed, and against the verified references
    (tests/sparse/ref/solve.npz, lstsq.npz) its forward error is within what that residual
    allows: ||x - ref|| <= ||A^-1|| ||b - A x|| (least squares: ||(A^T A)^-1|| ||A^T r||);
  - the bits: the same for 1, 3 and 10 threads (long vectors: the pool splits the products, the
    inner products and the updates), for CSR, CSC, COO and DOK, for the matrix and its
    aslinearoperator, and for a Python LinearOperator computing A @ v;
  - the result objects (SP72): scipy's tuple when unpacked, read-only fields, repr, summary();
  - the inputs and errors (SP73, SP74 and docs/api/sparse.md §6).
"""

import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import matrices as M
from . import refdata as R
from .conftest import LIVE, coo

S = sd.sparse
EPS = R.EPS
SQUARE = ("cg", "bicgstab", "gmres", "minres")
LSQ = ("lsqr", "lsmr")


def sparse(m, fmt="csr"):
    return coo(m).asformat(fmt)


def solve(fn, A, b, **kw):
    return getattr(S, fn)(A, b, **kw)


def run(fn, A, b, **kw):
    """(result, history): the residual norms the solver tracked, step by step."""
    from stardust.sparse import _iterative

    return _iterative._run(fn, A, b, **kw)


# ---------------------------------------------------------------- scipy's record

RECORDED = sorted(k for k in R.data("scipy") if k.startswith("iterative/"))

# The problems where scipy itself is chaotic: BiCGSTAB on the random nonsymmetric matrices and
# LSQR / LSMR on lap2d/30 (456-506 steps, the Lanczos vectors far from orthogonal). scipy with b
# changed by 1e-16 relative moves its iteration counts by up to 20% (bicgstab: 54-71 for the
# record's 57) and lsqr's final normr by 4% (test_chaotic_problems_stay_within_scipys_spread);
# there the record holds the status, and the counts are held to scipy's own spread.
CHAOTIC = {"bicgstab/random_unsym/1000x8", "bicgstab/random_complex/800", "lsqr/lap2d/30", "lsmr/lap2d/30"}


@pytest.mark.parametrize("key", RECORDED)
def test_status_and_iterations_are_scipys(key):
    """Every iterative record of tools/gen_sparse_scipy.py: the same status and iteration count
    (cg, bicgstab, minres: scipy's callbacks; gmres: its inner iterations, callback_type='pr_norm';
    lsqr, lsmr: istop and itn), and a residual as small (the chaotic ones: the status, and a
    converged residual)."""
    rec = R.data("scipy")[key]
    _, fn, fam, size, tol = key.split("/")
    chaotic = f"{fn}/{fam}/{size}" in CHAOTIC
    m = M.get(f"{fam}/{size}")
    assert rec["digest"] == M.digest(m)
    A, b = sparse(m), sd.asarray(M.rhs(m))
    if fn in LSQ:
        kw = {} if tol == "default" else {"atol": 1e-12, "btol": 1e-12}
        r = solve(fn, A, b, **kw)
        assert r.status == rec["istop"]
        if not chaotic:
            assert r.n_iter == rec["n_iter"]
            assert r.normr == pytest.approx(rec["normr"], rel=1e-6)
    else:
        kw = {} if tol == "default" else {"rtol": 1e-10}
        r = solve(fn, A, b, **kw)
        assert r.status == rec["info"]
        count = r.n_inner if fn == "gmres" else r.n_iter
        # bicgstab may stop half way through an iteration (its ||s|| test), which n_iter counts
        # and scipy's callback does not
        if not chaotic:
            assert count == rec["n_callbacks"] or (fn == "bicgstab" and count == rec["n_callbacks"] + 1)
    Ad = m.scipy("csr")
    bn = np.asarray(b)
    relres = np.linalg.norm(bn - Ad @ np.asarray(r.x)) / np.linalg.norm(bn)
    if chaotic:
        assert fn in LSQ or relres <= 1.01 * kw.get("rtol", 1e-5)
    else:
        assert relres <= 1.05 * rec["relres"] + 1e-14


def _perturbed(b, k):
    """b changed by about 1e-16 relative (k = 1, 2, ...): the size of a rounding difference."""
    return b * (1 + 1e-16 * np.cos((k + 1) * np.arange(b.size)))


@LIVE
@pytest.mark.parametrize("key", sorted(CHAOTIC))
@pytest.mark.parametrize("tight", [False, True])
def test_chaotic_problems_stay_within_scipys_spread(key, tight):
    """Where scipy is chaotic, this library's count and final residual estimate lie within the
    spread of scipy's own over five right-hand sides changed by 1e-16 (widened by 10%)."""
    import scipy.sparse.linalg as spla

    fn, mid = key.split("/", 1)
    m = M.get(mid)
    A, b = m.scipy("csr"), M.rhs(m)
    counts, norms = [], []
    for k in range(6):
        bk = b if k == 0 else _perturbed(b, k)
        if fn in LSQ:
            r = getattr(spla, fn)(A, bk, **({"atol": 1e-12, "btol": 1e-12} if tight else {}))
            counts.append(r[2])
            norms.append(float(r[3]))
        else:
            c = [0]
            getattr(spla, fn)(A, bk, rtol=1e-10 if tight else 1e-5, callback=lambda _: c.__setitem__(0, c[0] + 1))
            counts.append(c[0])
    kw = ({"atol": 1e-12, "btol": 1e-12} if tight else {}) if fn in LSQ else {"rtol": 1e-10 if tight else 1e-5}
    r = solve(fn, sparse(m), sd.asarray(b), **kw)
    lo, hi = min(counts), max(counts)
    assert 0.9 * lo - 1 <= r.n_iter <= 1.1 * hi + 1, (r.n_iter, counts)
    if norms:
        assert 0.9 * min(norms) <= r.normr <= 1.1 * max(norms), (r.normr, norms)


# ---------------------------------------------------------------- residual histories (live)

PROBLEMS = {
    "cg": ("lap2d/30", "random_spd/1500", "random_herm/800"),
    "bicgstab": ("convdiff2d/40", "random_unsym/1000x8", "random_complex/800"),
    "gmres": ("convdiff2d/40", "random_unsym/1000x8", "random_complex/800"),
    "minres": ("lap2d/30", "random_indef/1500", "saddle/600x150"),
    "lsqr": ("rect/60x20", "rect/1200x300", "rect/80x30d"),
    "lsmr": ("rect/60x20", "rect/1200x300", "rect/80x30d"),
}


def _scipy_history(fn, A, b, **kw):
    """scipy's residual norms step by step: gmres' callback_type='pr_norm' (times ||b||), the
    true residual of cg's, bicgstab's and minres' iterates (the callback's x), lsqr's r1norm and
    lsmr's normr from runs stopped after each step (iter_lim / maxiter = k)."""
    import scipy.sparse.linalg as spla

    bn = np.linalg.norm(b)
    if fn in LSQ:
        out = []
        for k in range(1, 400):
            r = getattr(spla, fn)(A, b, **{"iter_lim" if fn == "lsqr" else "maxiter": k}, **kw)
            out.append(float(r[3]))
            if r[1] != 7:
                break
        return np.array(out)
    got = []
    if fn == "gmres":
        spla.gmres(A, b, callback=lambda pr: got.append(pr * bn), callback_type="pr_norm", **kw)
    else:
        getattr(spla, fn)(A, b, callback=lambda x: got.append(np.linalg.norm(b - A @ x)), **kw)
    return np.array(got)


@LIVE
@pytest.mark.parametrize(("fn", "mid"), [(f, m) for f, ms in PROBLEMS.items() for m in ms])
def test_residual_histories_follow_scipy(fn, mid):
    """The residual norms step by step agree with scipy's to 1e-6 relative while they are above
    1e-9 ||b|| (below, the recurrences and the true residual part at the level of rounding) and
    while scipy's own history stays within 1e-9 of itself when b changes by 1e-16 (bicgstab on
    the random nonsymmetric matrices and minres on the indefinite ones are chaotic: scipy's
    residuals move by 1e-1 within ten steps of such a change; at least the first two steps)."""
    m = M.get(mid)
    A, b = m.scipy("csr"), M.rhs(m)
    kw = {"atol": 1e-10, "btol": 1e-10} if fn in LSQ else {"rtol": 1e-8}
    r, hist = run(fn, sparse(m), sd.asarray(b), **kw)
    mine = np.asarray(hist)
    theirs = _scipy_history(fn, A, b, **kw)
    other = _scipy_history(fn, A, _perturbed(b, 1), **kw)
    k = min(len(other), len(theirs))
    drift = np.abs(other[:k] - theirs[:k]) / theirs[:k]
    stable = int(np.argmax(drift > 1e-9)) if (drift > 1e-9).any() else len(theirs)
    if fn in ("cg", "bicgstab", "minres"):
        mine = mine[1:]  # after each step (the callback's iterates)
    elif fn in LSQ:
        mine = mine[1:]
    if stable == len(theirs):  # (bicgstab's last value may be its half step's ||s||)
        assert len(mine) - len(theirs) in ((0, 1) if fn == "bicgstab" else (0,))
    else:
        assert stable >= 2
    n = min(stable, len(mine), len(theirs))
    keep = theirs[:n] > 1e-9 * np.linalg.norm(b)
    assert keep.sum() >= min(3, n)
    rel = np.abs(mine[:n][keep] - theirs[:n][keep]) / theirs[:n][keep]
    assert rel.max() <= 1e-6, (rel.max(), int(np.argmax(rel)))


@LIVE
@pytest.mark.parametrize("fn", SQUARE)
def test_preconditioned_and_x0_follow_scipy(fn):
    """A Jacobi preconditioner (a diagonal sparse M, its inverse for M of cg and minres) and a
    start x0: the same status and iteration count as scipy."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    mid = {"cg": "random_spd/1500", "minres": "random_indef/1500"}.get(fn, "convdiff2d/40")
    m = M.get(mid)
    A, b = m.scipy("csr"), M.rhs(m)
    d = A.diagonal()
    Minv = sp.diags_array(1 / (np.abs(d) if fn == "minres" else d)).tocsr()
    x0 = 0.1 * np.cos(np.arange(A.shape[0]))
    count = [0]
    kw = {"callback_type": "pr_norm"} if fn == "gmres" else {}
    xs, info = getattr(spla, fn)(A, b, M=Minv, x0=x0, rtol=1e-9, callback=lambda *_: count.__setitem__(0, count[0] + 1),
                                 **kw)  # fmt: skip
    r = solve(fn, sparse(m), sd.asarray(b), M=S.csr_array(Minv), x0=sd.asarray(x0), rtol=1e-9)
    assert r.status == info
    assert (r.n_inner if fn == "gmres" else r.n_iter) == count[0]
    assert np.linalg.norm(np.asarray(r.x) - xs) <= 1e-6 * np.linalg.norm(xs)


@LIVE
@pytest.mark.parametrize(("restart", "max_iter"), [(5, 3), (7, 40), (30, None), (1, 50)])
def test_gmres_restart_and_max_iter_are_scipys(restart, max_iter):
    """SP75: restart is the inner iterations of a cycle, max_iter the cycles; info is max_iter
    when it stops there."""
    import scipy.sparse.linalg as spla

    m = M.get("convdiff2d/40")
    A, b = m.scipy("csr"), M.rhs(m)
    count = [0]
    xs, info = spla.gmres(A, b, restart=restart, maxiter=max_iter, rtol=1e-9,
                          callback=lambda _: count.__setitem__(0, count[0] + 1), callback_type="pr_norm")  # fmt: skip
    r = S.gmres(sparse(m), sd.asarray(b), restart=restart, max_iter=max_iter, rtol=1e-9)
    assert (r.status, r.n_inner) == (info, count[0])
    assert r.n_inner <= restart * r.n_iter
    if max_iter is not None:
        assert r.n_iter <= max_iter


@LIVE
@pytest.mark.parametrize("shift", [0.5, -1.25])
def test_minres_shift_follows_scipy(shift):
    import scipy.sparse.linalg as spla

    m = M.get("random_indef/1500")
    A, b = m.scipy("csr"), M.rhs(m)
    count = [0]
    xs, info = spla.minres(A, b, shift=shift, rtol=1e-9, callback=lambda _: count.__setitem__(0, count[0] + 1))
    r = S.minres(sparse(m), sd.asarray(b), shift=shift, rtol=1e-9)
    assert (r.status, r.n_iter) == (info, count[0])
    assert np.linalg.norm(np.asarray(r.x) - xs) <= 1e-6 * np.linalg.norm(xs)


@LIVE
@pytest.mark.parametrize("fn", LSQ)
@pytest.mark.parametrize("mid", ["rect/1200x300", "rect/20x60", "rect/80x30d", "random_complex/40"])
def test_lsq_damp_x0_and_estimates_follow_scipy(fn, mid):
    """damp > 0 and x0: istop, itn and the estimates (norms, cond) as scipy's; lsqr's var is
    scipy's calc_var=True (the squares' moduli for complex, docs/api/sparse.md §6)."""
    import scipy.sparse.linalg as spla

    m = M.get(mid)
    A, b = m.scipy("csr"), M.rhs(m)
    x0 = 0.05 * np.sin(np.arange(A.shape[1])) * (1 + 0j if np.iscomplexobj(b) else 1)
    for kw in ({"damp": 0.3}, {"x0": x0}, {"damp": 0.05, "x0": x0, "atol": 1e-10, "btol": 1e-10}):
        sk = dict(kw, calc_var=True) if fn == "lsqr" else dict(kw)
        ref = getattr(spla, fn)(A, b, **sk)
        mk = {k: (sd.asarray(v) if k == "x0" else v) for k, v in kw.items()}
        r = solve(fn, sparse(m), sd.asarray(b), **mk)
        got = tuple(r)
        assert (got[1], got[2]) == (ref[1], ref[2])
        mine = np.array([float(v) for v in got[3 : 9 if fn == "lsqr" else 8]])
        theirs = np.array([float(v) for v in ref[3 : 9 if fn == "lsqr" else 8]])
        floor = 1e-12 * r.norma * max(r.normr, 1e-300)  # ||A^H r|| near its rounding level
        assert np.all(np.abs(mine - theirs) <= 1e-6 * np.abs(theirs) + floor), (mine, theirs)
        np.testing.assert_allclose(np.asarray(r.x), ref[0], rtol=1e-8, atol=1e-12)
        if fn == "lsqr" and not np.iscomplexobj(b):  # (rank deficient: var is ill-determined)
            np.testing.assert_allclose(np.asarray(r.var), ref[9], rtol=1e-6)


@LIVE
def test_breakdowns_and_limits_are_scipys():
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    # bicgstab: rtilde^T A p = 0 at the first step (-11)
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = np.array([1.0, 0.0])
    assert spla.bicgstab(A, b)[1] == -11
    r = S.bicgstab(S.csr_array(sd.asarray(A)), sd.asarray(b))
    assert (r.status, r.converged) == (-11, False) and r.message.startswith("breakdown")
    # gmres: the Krylov space holds the solution after one step (breakdown, converged)
    r = S.gmres(S.eye_array(5) * 3.0, sd.asarray(np.arange(1.0, 6.0)))
    assert (r.status, r.n_iter, r.n_inner) == (0, 1, 1)
    np.testing.assert_allclose(np.asarray(r.x), np.arange(1.0, 6.0) / 3, rtol=1e-15)
    # the cap: status max_iter, converged False, the last iterate
    m = M.get("lap2d/30")
    for fn in SQUARE:
        x, info = getattr(spla, fn)(m.scipy("csr"), M.rhs(m), maxiter=3)
        r = solve(fn, sparse(m), sd.asarray(M.rhs(m)), max_iter=3)
        assert r.status == info
        if fn != "minres":
            assert (r.status, r.converged, r.message) == (3, False, "the iteration limit was reached")
    for fn in LSQ:
        ref = getattr(spla, fn)(m.scipy("csr"), M.rhs(m), **{"iter_lim" if fn == "lsqr" else "maxiter": 4})
        r = solve(fn, sparse(m), sd.asarray(M.rhs(m)), max_iter=4)
        assert (r.status, r.n_iter, r.converged) == (ref[1], ref[2], False) == (7, 4, False)
    # b = 0: x = 0 at once
    z = np.zeros(m.shape[0])
    for fn in SQUARE + LSQ:
        r = solve(fn, sparse(m), sd.asarray(z))
        ref = getattr(spla, fn)(m.scipy("csr"), z)
        assert r.status == ref[1] and r.n_iter == 0 and not np.asarray(r.x).any()
    # minres: an indefinite preconditioner (scipy: ValueError; here LinAlgError, a ValueError)
    with pytest.raises(ValueError, match="indefinite"):
        spla.minres(m.scipy("csr"), M.rhs(m), M=-sp.eye_array(m.shape[0]))
    with pytest.raises(sd.linalg.LinAlgError, match=r"minres\(\): the preconditioner is not positive definite"):
        S.minres(sparse(m), sd.asarray(M.rhs(m)), M=-S.eye_array(m.shape[0]))


# ---------------------------------------------------------------- the contract


def _true_residual(m, x, b):
    A = m.scipy("csr")
    return np.linalg.norm(b - A @ x)


@pytest.mark.parametrize("fn", SQUARE)
@pytest.mark.parametrize("rtol", [1e-5, 1e-10])
def test_converged_means_the_true_residual_meets_the_test(fn, rtol):
    """A converged result's recomputed residual meets the stopping test (cg, bicgstab, gmres:
    ||b - A x|| <= rtol ||b||; minres: its relative residual test, ||r|| <= rtol ||A|| ||x||), to
    the gap between the recurrence and the true residual (1% here; docs/design/sparse.md §4)."""
    mids = {"cg": PROBLEMS["cg"], "minres": PROBLEMS["minres"]}.get(fn, PROBLEMS["gmres"])
    for mid in mids:
        m = M.get(mid)
        b = M.rhs(m)
        r = solve(fn, sparse(m), sd.asarray(b), rtol=rtol)
        assert r.converged and r.status == 0
        true = _true_residual(m, np.asarray(r.x), b)
        if fn == "minres":
            if r.message.startswith("a solution"):
                assert true <= 1.01 * rtol * r.norma * r.normx
        else:
            assert true <= 1.01 * rtol * np.linalg.norm(b), (mid, true / np.linalg.norm(b))


def _residual_bound(m, x, b):
    """||b - A x||_inf from the float64 residual and its rounding bound: an upper bound of the
    exact residual's norm."""
    A = m.scipy("csr")
    r = b - A @ x
    absAx = abs(A) @ np.abs(x)
    k = np.diff(A.indptr).max() + 1
    return float(np.max(np.abs(r) + R.gamma(int(k)) * (np.abs(b) + absAx)))


SOLVE_POINTS = [p for p in R.points("solve") if "graded" not in p["matrix"] and "near_singular" not in p["matrix"]]


def _method_for(m):
    if "spd" in m.tags:
        return "cg"
    if "symmetric" in m.tags or "hermitian" in m.tags:
        return "minres"
    return "gmres"


@pytest.mark.parametrize("p", SOLVE_POINTS, ids=[p["matrix"] for p in SOLVE_POINTS])
def test_forward_error_against_the_verified_solutions(p):
    """Against Arb's solutions (tools/sparse_ref.py): ||x - ref||_inf <= ||A^-1||_inf ||b - A x||_inf,
    with ||A^-1||_inf = kappa_inf / ||A||_inf from the reference and the residual bounded from
    above (its rounding added), for the method that suits the matrix and for bicgstab (gmres with
    restart n for the ill-conditioned) at rtol 1e-12, and the same bound for lsqr and lsmr."""
    m = M.get(p["matrix"])
    b = p["b"]
    ref = p["solve_hi"] + p["solve_lo"]
    anorm = float(abs(m.scipy("csr")).sum(axis=1).max())
    inv_norm = p["solve_cond"] / anorm
    n = m.shape[0]
    tried = 0
    for fn in (_method_for(m), "bicgstab", "lsmr"):
        kw = {"atol": 1e-14, "btol": 1e-14, "max_iter": 20 * n} if fn in LSQ else {"rtol": 1e-12, "max_iter": 20 * n}
        if fn == "gmres":
            kw["restart"] = n
        r = solve(fn, sparse(m), sd.asarray(b), **kw)
        if not r.converged:
            continue
        tried += 1
        x = np.asarray(r.x)
        err = float(np.max(np.abs(x - ref)))
        bound = inv_norm * _residual_bound(m, x, b)
        assert err <= bound * (1 + 1e-6) + 4 * EPS * float(np.max(np.abs(ref))), (fn, err, bound)
    assert tried >= 1


@pytest.mark.parametrize("p", R.points("lstsq"), ids=[p["matrix"] for p in R.points("lstsq")])
@pytest.mark.parametrize("fn", LSQ)
def test_least_squares_against_the_verified_solutions(p, fn):
    """A^T A (x - ref) = A^T r: ||x - ref||_inf <= ||(A^T A)^-1||_inf ||A^T (b - A x)||_inf, the
    reference's kappa_inf(A^T A) giving the inverse's norm; and the solution to 1e-9 relative at
    atol = btol = 1e-14."""
    m = M.get(p["matrix"])
    b = p["b"]
    ref = p["lstsq_hi"] + p["lstsq_lo"]
    A = m.scipy("csr")
    G = (A.T @ A).toarray()
    inv_norm = p["lstsq_cond"] / float(np.abs(G).sum(axis=1).max())
    r = solve(fn, sparse(m), sd.asarray(b), atol=1e-14, btol=1e-14)
    assert r.converged
    x = np.asarray(r.x)
    res = b - A @ x
    atr = abs(A.T) @ (np.abs(res) + R.gamma(8) * (np.abs(b) + abs(A) @ np.abs(x)))
    bound = inv_norm * float(np.max(np.abs(A.T @ res) + R.gamma(A.shape[0]) * atr))
    err = float(np.max(np.abs(x - ref)))
    assert err <= bound * (1 + 1e-6) + 4 * EPS * float(np.max(np.abs(ref)))
    assert err <= 1e-9 * float(np.max(np.abs(ref)))


# ---------------------------------------------------------------- the bits


def _wide(m_rows, n_cols, lower=-1.0):
    """A well-conditioned banded matrix long enough that the pool splits every vector operation
    (n >= 2^15): 4 on the diagonal, -1 above it and `lower` below it (symmetric for -1)."""
    k = min(m_rows, n_cols)
    return S.diags_array([sd.full(k, 4.0), sd.full(k, -1.0), sd.full(k, lower)], offsets=[0, 1, -1],
                         shape=(m_rows, n_cols), format="csr")  # fmt: skip


def _rhs(n, complex_=False):
    t = np.arange(n, dtype=np.float64)
    b = np.sin(0.37 * t) + 0.1
    return sd.asarray(b + 1j * np.cos(0.11 * t) if complex_ else b)


@pytest.mark.parametrize("complex_", [False, True])
def test_same_bits_on_every_thread_count(restore_threads, complex_):
    A, U = _wide(40000, 40000), _wide(40000, 40000, 0.5)
    R_ = _wide(45000, 40000, 0.5)
    if complex_:  # Hermitian A: real diagonal, conjugate off-diagonals
        A = A + S.diags_array([sd.full(39999, 0.5j), sd.full(39999, -0.5j)], offsets=[1, -1], format="csr")
        U, R_ = U * (1 + 0.5j), R_ * (1 - 0.25j)
    b, br = _rhs(40000, complex_), _rhs(45000, complex_)
    calls = [lambda fn=fn: solve(fn, A if fn in ("cg", "minres") else U, b, rtol=1e-12) for fn in SQUARE] + [
        lambda fn=fn: solve(fn, R_, br, atol=1e-12, btol=1e-12) for fn in LSQ
    ]
    first = None
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        got = [run_ for run_ in (c() for c in calls)]
        xs = [np.asarray(g.x) for g in got]
        if first is None:
            first = xs
            assert all(g.converged for g in got)
        else:
            assert all(bits_equal(a, f) for a, f in zip(xs, first))


@pytest.mark.parametrize("fn", SQUARE + LSQ)
def test_same_bits_for_every_format_and_operator(fn):
    """CSR, CSC, COO, DOK, aslinearoperator(A) and a Python LinearOperator v -> A @ v (whose
    products are the same SpMV) give the same bits; so does M given as each of them."""
    mid = {"cg": "random_spd/60", "minres": "random_indef/60", "lsqr": "rect/60x20", "lsmr": "rect/60x20"}.get(
        fn, "random_unsym/50x5"
    )
    m = M.get(mid)
    b = sd.asarray(M.rhs(m))
    A = sparse(m)
    first = np.asarray(solve(fn, A, b).x)
    ops = [sparse(m, f) for f in ("csc", "coo", "dok")] + [S.aslinearoperator(A)]
    AH = A.T.tocsr()
    ops.append(S.LinearOperator(A.shape, lambda v: A @ v, rmatvec=lambda v: AH @ v))
    for op in ops:
        assert bits_equal(np.asarray(solve(fn, op, b).x), first)
    if fn in SQUARE:
        D = S.diags_array(sd.asarray(1.0 / np.abs(np.asarray(A.diagonal()))), format="csr")
        ref = np.asarray(solve(fn, A, b, M=D).x)
        for Mop in (D.tocsc(), S.aslinearoperator(D), S.LinearOperator(D.shape, lambda v: D @ v)):
            assert bits_equal(np.asarray(solve(fn, A, b, M=Mop).x), ref)


def test_dense_matrices_solve_as_their_stored_entries():
    """A dense sd.ndarray A is the CSR of all its entries: the same solution as the sparse array
    to rounding (zeros stored or not change only the sign of a zero sum)."""
    m = M.get("random_spd/60")
    b = sd.asarray(M.rhs(m))
    for fn in SQUARE + LSQ:
        xs = np.asarray(solve(fn, sparse(m), b).x)
        xd = np.asarray(solve(fn, sd.asarray(m.dense()), b).x)
        np.testing.assert_allclose(xd, xs, rtol=0, atol=1e-14 * np.abs(xs).max())


# ---------------------------------------------------------------- the results


def test_results_unpack_as_scipys_tuples():
    m = M.get("lap2d/10")
    A, b = sparse(m), sd.asarray(M.rhs(m))
    for fn in SQUARE:
        r = solve(fn, A, b)
        x, info = r
        assert len(r) == 2 and info == r.status and x is r.x and r[0] is r.x and r[-1] == r.status
    r = S.lsqr(A, b)
    assert len(r) == 10
    x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = r
    assert (istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm) == (
        r.status,
        r.n_iter,
        r.normr,
        r.normr_damped,
        r.norma,
        r.conda,
        r.normar,
        r.normx,
    )
    assert var is r.var and x is r.x
    r = S.lsmr(A, b)
    assert len(r) == 8
    x, istop, itn, normr, normar, norma, conda, normx = r
    assert (istop, itn, normr, normar, norma, conda, normx) == (
        r.status,
        r.n_iter,
        r.normr,
        r.normar,
        r.norma,
        r.conda,
        r.normx,
    )


def test_result_fields_are_read_only():
    m = M.get("lap2d/10")
    r = S.cg(sparse(m), sd.asarray(M.rhs(m)))
    with pytest.raises(ValueError, match="read-only"):
        r.x[0] = 1.0
    with pytest.raises(AttributeError):
        r.status = 3
    with pytest.raises(AttributeError):
        _ = r.norma
    assert r.x is r.x
    v = S.lsqr(sparse(m), sd.asarray(M.rhs(m))).var
    with pytest.raises(ValueError, match="read-only"):
        v[0] = 1.0


def test_repr_and_summary():
    m = M.get("lap2d/10")
    r = S.cg(sparse(m), sd.asarray(M.rhs(m)))
    want = f"CgResult(converged=True, status=0, n_iter={r.n_iter}, normr={r.normr:.6g}, x=<array of shape (100,)>)"
    assert repr(r) == want
    s = r.summary()
    lines = s.splitlines()
    assert lines[0].strip() == "Conjugate Gradient Results"
    assert "Converged:" in s and "No. Iterations:" in s and "Message:" in s and "x[99]" in s and "..." in s
    assert all(len(line) <= 78 for line in lines)
    z = S.gmres(sparse(M.get("random_complex/40")), sd.asarray(M.rhs(M.get("random_complex/40"))))
    assert "<array of shape (40,)>" in repr(z) and "real" in z.summary() and "imag" in z.summary()
    with pytest.raises(TypeError, match=r"summary\(\): takes 0 positional arguments but 1 was given"):
        r.summary(1)
    for fn, title in (("bicgstab", "BiCGSTAB Results"), ("gmres", "GMRES Results"), ("minres", "MINRES Results"),
                      ("lsqr", "LSQR Results"), ("lsmr", "LSMR Results")):  # fmt: skip
        assert solve(fn, sparse(m), sd.asarray(M.rhs(m))).summary().splitlines()[0].strip() == title


# ---------------------------------------------------------------- inputs and errors


def test_shapes_and_dtypes():
    m = M.get("lap2d/10")
    A, b = sparse(m), M.rhs(m)
    x1 = np.asarray(S.cg(A, sd.asarray(b)).x)
    r = S.cg(A, sd.asarray(b.reshape(-1, 1)))  # (n, 1): scipy's ravel, x is (n,)
    assert r.x.shape == (100,) and bits_equal(np.asarray(r.x), x1)
    assert S.cg(A.astype(sd.int64), sd.asarray(b)).x.dtype == sd.float64
    assert S.cg(A, sd.asarray(b) * 1j).x.dtype == sd.complex128
    assert S.cg(A, sd.asarray(b), x0=sd.zeros(100, dtype=sd.complex128)).x.dtype == sd.complex128
    assert S.gmres(A, sd.asarray(b), M=S.eye_array(100) * (1 + 0j)).x.dtype == sd.complex128
    assert S.lsqr(A, sd.asarray(np.ones(100, dtype=bool))).x.dtype == sd.float64
    E = S.csr_array((0, 0))
    for fn in SQUARE + LSQ:
        r = solve(fn, E, sd.zeros(0))
        assert r.x.shape == (0,) and r.status == 0
    W = S.csr_array((3, 0))
    assert S.lsqr(W, sd.ones(3)).x.shape == (0,)


def test_wrong_calls():
    m = M.get("lap2d/10")
    A, b = sparse(m), sd.asarray(M.rhs(m))
    cases = [
        (lambda: S.cg([[1.0]], b), TypeError,
         r"cg\(\): 'A' must be a sparse array, an sd.ndarray or a LinearOperator, not list"),
        (lambda: S.cg(A, list(M.rhs(m))), TypeError, r"cg\(\): 'b' must be an sd.ndarray, not list"),
        (lambda: S.cg(A, np.ones(100)), TypeError, r"'b' must be an sd.ndarray, not numpy.ndarray"),
        (lambda: S.cg(S.csr_array((3, 4)), sd.ones(3)), ValueError, r"cg\(\): 'A' must be square, got 3x4"),
        (lambda: S.gmres(A, sd.ones(3)), ValueError, r"gmres\(\): 'b' must have shape \(100,\) or \(100, 1\)"),
        (lambda: S.cg(A, b, x0=sd.ones(3)), ValueError, r"'x0' must have shape \(100,\) or \(100, 1\)"),
        (lambda: S.cg(A, b, M=S.eye_array(3)), ValueError, r"'M' must have the shape of 'A'"),
        (lambda: S.cg(A, b, rtol=-1.0), ValueError, r"cg\(\): 'rtol' must be a finite number >= 0"),
        (lambda: S.cg(A, b, atol=math.nan), ValueError, r"'atol' must be a finite number >= 0"),
        (lambda: S.cg(A, b, rtol="1e-5"), TypeError, r"'rtol' must be an int or a float, not str"),
        (lambda: S.cg(A, b, max_iter=0), ValueError, r"cg\(\): 'max_iter' must be >= 1"),
        (lambda: S.cg(A, b, max_iter=2.0), TypeError, r"'max_iter' must be an int or None, not float"),
        (lambda: S.gmres(A, b, restart=0), ValueError, r"gmres\(\): 'restart' must be >= 1"),
        (lambda: S.minres(A, b, shift=math.inf), ValueError, r"minres\(\): 'shift' must be finite"),
        (lambda: S.lsqr(A, b, damp=-1.0), ValueError, r"lsqr\(\): 'damp' must be a finite number >= 0"),
        (lambda: S.lsmr(A, b, conlim=-1.0), ValueError, r"lsmr\(\): 'conlim' must be a number >= 0"),
        (lambda: S.lsqr(A, sd.ones(3)), ValueError, r"lsqr\(\): 'b' must have shape \(100,\) or \(100, 1\)"),
        (lambda: S.lsmr(S.csr_array((5, 3)), sd.ones(5), x0=sd.ones(5)), ValueError, r"'x0' must have shape \(3,\)"),
        (lambda: S.cg(A, b, maxiter=3), TypeError, r"cg\(\): unexpected keyword argument 'maxiter'"),
        (lambda: S.cg(A, b, callback=print), TypeError, r"cg\(\): unexpected keyword argument 'callback'"),
        (lambda: S.lsqr(A, b, iter_lim=3), TypeError, r"lsqr\(\): unexpected keyword argument 'iter_lim'"),
        (lambda: S.lsqr(A, b, calc_var=True), TypeError, r"unexpected keyword argument 'calc_var'"),
        (lambda: S.minres(A, b, check=True), TypeError, r"unexpected keyword argument 'check'"),
        (lambda: S.cg(A, b, 1e-5), TypeError, r"cg\(\): takes 2 positional arguments but 3 were given"),
        (lambda: S.cg(S.csr_array(sd.asarray(np.array([[1.0, np.nan], [0, 1]]))), sd.ones(2)), ValueError,
         r"cg\(\): 'A' contains NaN or infinity"),
        (lambda: S.cg(A, b * math.inf), ValueError, r"cg\(\): 'b' contains NaN or infinity"),
        (lambda: S.cg(S.csr_array(sd.ones(3)), sd.ones(3)), ValueError, r"'A' must have 2 dimensions, got 1"),
    ]  # fmt: skip
    for call, exc, pattern in cases:
        with pytest.raises(exc, match=pattern):
            call()


def test_python_operators():
    """A LinearOperator's products run through Python: its errors pass through, a complex
    product in a real system is a TypeError, lsqr needs rmatvec."""
    m = M.get("lap2d/10")
    A, b = sparse(m), sd.asarray(M.rhs(m))
    calls = []

    def mv(v):
        calls.append(1)
        return A @ v

    r = S.cg(S.LinearOperator(A.shape, mv), b)
    assert r.converged and len(calls) >= r.n_iter

    def bad(v):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        S.gmres(S.LinearOperator(A.shape, bad, dtype=sd.float64), b)
    with pytest.raises(TypeError, match="complex128 values for a float64 system"):
        S.cg(S.LinearOperator(A.shape, lambda v: (A @ v) * 1j, dtype=sd.float64), b)
    with pytest.raises(ValueError, match="^lsqr\\(\\): the operator was made without 'rmatvec'$"):
        S.lsqr(S.LinearOperator(A.shape, lambda v: A @ v), b)


def test_operator_errors_name_the_solver():
    """An error of a Python operator inside a solve starts with the solver's name (R228); lsqr and
    lsmr check for rmatvec before the first product."""
    op = S.LinearOperator((2, 2), lambda v: v * 1j, dtype=sd.float64)
    for f in (S.cg, S.gmres, S.bicgstab, S.minres):
        with pytest.raises(TypeError, match=rf"^{f.__name__}\(\): the operator returned complex128 values for a "
                                            r"float64 system; give it dtype=complex128$"):  # fmt: skip
            f(op, sd.ones(2))
    m = M.get("lap2d/10")
    A, b = sparse(m), sd.asarray(M.rhs(m))
    calls = []

    def mv(v):
        calls.append(1)
        return A @ v

    for f in (S.lsqr, S.lsmr):
        with pytest.raises(ValueError, match=rf"^{f.__name__}\(\): the operator was made without 'rmatvec'$"):
            f(S.LinearOperator(A.shape, mv, dtype=sd.float64), b)
        assert not calls  # before the first product
        # an operator made from one without rmatvec (its .H, a sum) names the solver too
        L = S.LinearOperator(A.shape, mv, dtype=sd.float64)
        with pytest.raises(ValueError, match=rf"^{f.__name__}\(\): the operator was made without 'rmatvec'$"):
            f(L + L, b)
        calls.clear()


def test_minres_names_the_preconditioner():
    """r2^H M r2 < 0 inside the iteration comes from an indefinite M (R227, SP126): the message
    names it; a nonsymmetric A is not checked (scipy's)."""
    T = S.diags_array([-1.0, 2.0, -1.0], offsets=[-1, 0, 1], shape=(30, 30))
    A = S.kronsum(T, T)
    d = np.ones(900)
    d[::7] = -1.0
    with pytest.raises(sd.linalg.LinAlgError, match=r"^minres\(\): the preconditioner is not positive definite$"):
        S.minres(A, sd.ones(900), M=S.diags_array(sd.asarray(d)))
    D1 = S.diags_array([-1.0, 1.0], offsets=[-1, 0], shape=(30, 30))
    C = A + 1.5 * S.kron(S.eye_array(30), D1)
    r = S.minres(C, sd.ones(900))
    assert r.status == 0  # no symmetry check: the result is not a solution of C x = b
    assert float(sd.linalg.vector_norm(sd.ones(900) - C @ r.x)) > 1.0
