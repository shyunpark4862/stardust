"""The eigensolvers of sd.sparse (milestone SP6; docs/api/sparse.md §7, docs/design/sparse.md §3.8,
§4): eigsh, eigs, svds.

What the tests hold them to:
  - ARPACK's computation, as scipy 1.18.1 runs it (SP83): with the same v0, the same status and
    the same number of OP x products as scipy (counted through a LinearOperator) on the problems
    where scipy is not itself sensitive to rounding; on the sensitive ones (repeated eigenvalues,
    slow convergence, nonnormal matrices) a count within scipy's own spread under one-ulp changes
    of v0 (widened by 10%). The eigenvalues agree with scipy's to 1e-9 relative, and the
    recorded ones of tools/gen_sparse_scipy.py (v0 = ones) too; every mode and spectral
    transformation (M, sigma, buckling, Cayley, OPpart, complex Hermitian through eigs);
  - the contract (docs/design/sparse.md §4) against Arb (tests/sparse/ref/eig.npz): eigsh's and
    svds's values within C_EIG eps ||A||_2, eigs's within C_EIG eps ||A||_2 kappa(lambda); the
    residuals ||A v - w M v|| and the orthogonality of the vectors;
  - the R comparators (tests/sparse/data/r_eigen.json: RSpectra, irlba): where they find the
    wanted eigenvalues, they and sd agree;
  - svds (SP85): the k largest / smallest singular values of tall, wide and complex matrices, its
    vectors, return_singular_vectors;
  - the start vector (SP84): v0 = None is the documented fixed vector, rng changes it;
  - the bits: the same for 1, 3 and 10 threads, for every format and the operator forms;
  - the result objects (SP87): (w, v) and (u, s, vh) when unpacked, status and partial results
    at the iteration limit, repr, summary();
  - the inputs and errors.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import stardust as sd

from .._bits import bits_equal
from . import matrices as M
from . import refdata as R
from .conftest import LIVE, coo

S = sd.sparse
EPS = R.EPS


def sparse(m, fmt="csr"):
    return coo(m).asformat(fmt)


def npy(x):
    return np.asarray(x.tolist()) if x is not None else None


def _lap1(n):
    return sp.diags_array([2 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)], offsets=[0, 1, -1]).tocsr()


def _sd(A):
    return S.csr_array(sp.csr_array(A))


def _v0(n, complex_=False):
    t = np.arange(n) + 1.0
    return np.sin(t) + 1j * np.cos(t) if complex_ else np.sin(t)


def _counted(A):
    c = [0]

    def mv(x):
        c[0] += 1
        return A @ x

    return spla.LinearOperator(A.shape, matvec=mv, dtype=A.dtype), c


def _scipy_run(fn, A, v0, **kw):
    """(status, OP count, eigenvalues) of scipy's eigsh / eigs with v0 (0, or 1 at the limit)."""
    op, c = _counted(A)
    try:
        w, _ = getattr(spla, fn)(op, v0=v0, **kw)
        return 0, c[0], w
    except spla.ArpackNoConvergence as e:
        return 1, c[0], e.eigenvalues


def _spread_runs(fn, A, v0, **kw):
    """scipy's runs (status, OP count, eigenvalues) with v0 and with one entry of v0 moved by one
    ulp, five ways."""
    out = []
    for p in range(6):
        v = v0.copy()
        if p:
            v[p * 7] = np.nextafter(v[p * 7].real, 2) + (1j * v[p * 7].imag if np.iscomplexobj(v) else 0)
        out.append(_scipy_run(fn, A, v, **kw))
    return out


def _spread(fn, A, v0, **kw):
    """scipy's OP counts with v0 and with one entry of v0 moved by one ulp, five ways."""
    counts = [c for _, c, _ in _spread_runs(fn, A, v0, **kw)]
    return min(counts), max(counts)


# ---------------------------------------------------------------- ARPACK, as scipy runs it

_N = 400
_RSYM = (lambda R_: (R_ + R_.T).tocsr())(sp.random_array((_N, _N), density=0.02, rng=np.random.default_rng(3)))
_L2 = sp.kronsum(_lap1(20), _lap1(20)).tocsr()
_UNSYM = (sp.random_array((300, 300), density=0.03, rng=np.random.default_rng(5)) + sp.diags_array(
    np.linspace(-1, 3, 300))).tocsr()  # fmt: skip
_CPLX = (sp.random_array((300, 300), density=0.03, rng=np.random.default_rng(7))
         + 1j * sp.random_array((300, 300), density=0.03, rng=np.random.default_rng(8))
         + sp.diags_array(np.linspace(0, 2, 300))).tocsr()  # fmt: skip

# (name, function, matrix, keywords, exact): exact problems give the same eigenvalues in scipy's
# order, the others (scipy sensitive to the last bits: lap2d/20's repeated eigenvalues, the slow
# 'SM' of the random ones) the eigenvalues of one of scipy's runs under one-ulp changes of v0, in
# any order. The OP count is within the spread of those runs widened by 10%: the basis products
# are Accelerate's gemv, whose sums are not scipy's BLAS's in the last bits, so a count can move
# as scipy's own does under one-ulp changes of v0 (lap1d-LM-3: 4483 against scipy's 4466-4483).
ARPACK = [
    ("lap1d-LM-3", "eigsh", _lap1(400), {"k": 3, "which": "LM"}, True),
    ("lap1d-SA-6", "eigsh", _lap1(400), {"k": 6, "which": "SA"}, True),
    ("lap1d-BE-6", "eigsh", _lap1(400), {"k": 6, "which": "BE"}, True),
    ("lap2d-LM-1", "eigsh", _L2, {"k": 1}, True),
    ("lap2d-SA-3", "eigsh", _L2, {"k": 3, "which": "SA"}, True),
    ("lap2d-LM-3", "eigsh", _L2, {"k": 3}, False),
    ("lap2d-BE-6", "eigsh", _L2, {"k": 6, "which": "BE"}, False),
    ("rsym-LM-6", "eigsh", _RSYM, {"k": 6}, True),
    ("rsym-SA-3", "eigsh", _RSYM, {"k": 3, "which": "SA"}, True),
    ("rsym-BE-3", "eigsh", _RSYM, {"k": 3, "which": "BE"}, True),
    ("unsym-LM-1", "eigs", _UNSYM, {"k": 1}, True),
    ("unsym-LM-4", "eigs", _UNSYM, {"k": 4}, True),
    ("unsym-LR-1", "eigs", _UNSYM, {"k": 1, "which": "LR"}, True),
    ("unsym-LI-4", "eigs", _UNSYM, {"k": 4, "which": "LI"}, False),
    ("unsym-SI-1", "eigs", _UNSYM, {"k": 1, "which": "SI"}, True),
    ("cplx-LM-4", "eigs", _CPLX, {"k": 4}, False),
    ("cplx-SR-4", "eigs", _CPLX, {"k": 4, "which": "SR"}, False),
    ("cplx-LI-4", "eigs", _CPLX, {"k": 4, "which": "LI"}, True),
    ("cplx-SI-4", "eigs", _CPLX, {"k": 4, "which": "SI"}, True),
    ("cplx-LR-4", "eigs", _CPLX, {"k": 4, "which": "LR"}, True),
]


@LIVE
@pytest.mark.parametrize(("name", "fn", "A", "kw", "exact"), ARPACK, ids=[a[0] for a in ARPACK])
def test_arpack_follows_scipy(name, fn, A, kw, exact):
    """The same status and number of OP x as scipy with the same v0; the same eigenvalues."""
    n = A.shape[0]
    v0 = _v0(n, A.dtype == np.complex128)
    runs = _spread_runs(fn, A, v0, **kw)
    status, count, w = runs[0]
    r = getattr(S, fn)(_sd(A), v0=sd.asarray(v0), **kw)
    assert r.status == status
    lo, hi = min(c for _, c, _ in runs), max(c for _, c, _ in runs)
    assert 0.9 * lo <= r.n_op <= 1.1 * hi, (r.n_op, lo, hi)
    got = npy(r.eigenvalues)
    if exact:
        np.testing.assert_allclose(got, w, rtol=1e-9, atol=1e-9 * np.abs(w).max())
        return
    # equally converged values may come in another order; a sensitive problem may find another
    # set, as scipy's own runs do
    got = np.sort_complex(got)
    close = [
        np.allclose(got, np.sort_complex(wr), rtol=1e-9, atol=1e-9 * np.abs(wr).max())
        for _, _, wr in runs if len(wr) == len(got)
    ]  # fmt: skip
    assert any(close), (got, w)


@LIVE
@pytest.mark.parametrize(
    "kw",
    [
        {"M": "M"},
        {"M": "M", "which": "SA"},
        {"sigma": 0.5},
        {"sigma": 0.5, "M": "M"},
        {"sigma": 0.0},
        {"sigma": 0.5, "mode": "buckling"},
        {"sigma": 0.5, "mode": "cayley"},
        {"sigma": 0.5, "mode": "cayley", "M": "M"},
        {"sigma": 0.5, "return_eigenvectors": False},
        {"return_eigenvectors": False},
        {"which": "SA", "return_eigenvectors": False},
    ],  # fmt: skip
    ids=str,
)
def test_eigsh_modes_follow_scipy(kw):
    """M (mode 2), shift-invert (3), buckling (4), Cayley (5): the eigenvalues, their order and the
    OP count of scipy (the inverses are LUs, as scipy's splu)."""
    n = 200
    A = _lap1(n)
    Mm = sp.diags_array(np.linspace(1, 2, n)).tocsr()
    v0 = _v0(n)
    skw = {**kw, **({"M": Mm} if "M" in kw else {})}
    if "sigma" in kw or not kw.get("return_eigenvectors", True):  # (a LinearOperator would make scipy use gmres)
        w = spla.eigsh(A, k=4, v0=v0, **skw)
        status, count, w = 0, None, w[0] if isinstance(w, tuple) else w
    else:
        status, count, w = _scipy_run("eigsh", A, v0, k=4, **skw)
    r = S.eigsh(_sd(A), k=4, v0=sd.asarray(v0), **{**kw, **({"M": _sd(Mm)} if "M" in kw else {})})
    assert r.status == status == 0
    if count is not None:
        assert r.n_op == count
    np.testing.assert_allclose(npy(r.eigenvalues), w, rtol=1e-10)
    if kw.get("return_eigenvectors", True):
        V = npy(r.eigenvectors)
        lam = npy(r.eigenvalues)
        # buckling's B is A itself (M = None): K x = lambda x, vectors A-orthonormal
        B = Mm if "M" in kw else sp.eye_array(n)
        inner = A if kw.get("mode") == "buckling" else B
        assert np.linalg.norm(A @ V - (B @ V) * lam) <= 1e-10 * np.abs(lam).max()
        assert np.abs(V.T @ (inner @ V) - np.eye(4)).max() <= 1e-12


@LIVE
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"which": "LR"},
        {"M": "M"},
        {"sigma": 0.5},
        {"sigma": 0.5 + 0.3j},
        {"sigma": 0.5 + 0.3j, "OPpart": "i"},
        {"sigma": 1.0, "M": "M", "k": 3},
        {"return_eigenvectors": False},
    ],
    ids=str,
)
def test_eigs_modes_follow_scipy(kw):
    """eigs of a real nonsymmetric matrix: M, shift-invert with a real and a complex sigma (OPpart
    'r' and 'i', the Rayleigh quotients of scipy's extract): scipy's eigenvalues in scipy's order,
    and eigenvectors with small residuals."""
    A = _UNSYM[:200, :200].tocsr()
    n = 200
    Mm = sp.diags_array(np.linspace(1, 2, n)).tocsr()
    v0 = _v0(n)
    kw = {"k": 4, **kw}
    skw = {**kw, **({"M": Mm} if "M" in kw else {})}
    w = spla.eigs(A, v0=v0, **skw)
    w = w[0] if isinstance(w, tuple) else w
    r = S.eigs(_sd(A), v0=sd.asarray(v0), **{**kw, **({"M": _sd(Mm)} if "M" in kw else {})})
    assert r.status == 0 and r.eigenvalues.dtype == sd.complex128
    got = npy(r.eigenvalues)
    if not kw.get("return_eigenvectors", True):
        # without vectors, a conjugate pair split by k may give the other member (a tie of |w|)
        got, w = got.real + 1j * np.abs(got.imag), w.real + 1j * np.abs(w.imag)
    np.testing.assert_allclose(got, w, rtol=1e-9)
    if kw.get("return_eigenvectors", True):
        V, lam = npy(r.eigenvectors), npy(r.eigenvalues)
        B = Mm if "M" in kw else sp.eye_array(n)
        assert np.linalg.norm(A @ V - (B @ V) * lam) <= 1e-10 * np.abs(lam).max()
    else:
        assert r.eigenvectors is None


@LIVE
@pytest.mark.parametrize("kw", [{}, {"which": "SA"}, {"which": "LA"}, {"sigma": 0.3}], ids=str)
def test_complex_hermitian_runs_on_complex_arnoldi(kw):
    """A complex Hermitian eigsh is scipy's: eigs (znaupd) with 'LA' as 'LR', real eigenvalues;
    the values ascending as the real path's (SP127; scipy keeps ARPACK's order), each column of
    the vectors moved with its value."""
    H = (_CPLX + _CPLX.conj().T).tocsr()
    v0 = _v0(300, True)
    w, V = spla.eigsh(H, k=4, v0=v0, **kw)
    r = S.eigsh(_sd(H), k=4, v0=sd.asarray(v0), **kw)
    assert r.eigenvalues.dtype == sd.float64 and r.eigenvectors.dtype == sd.complex128
    got = npy(r.eigenvalues)
    assert np.all(np.diff(got) >= 0)
    np.testing.assert_allclose(got, np.sort(w), rtol=1e-10)
    Z = npy(r.eigenvectors)
    np.testing.assert_allclose(np.linalg.norm(H @ Z - Z * got, axis=0), 0, atol=1e-10 * abs(w).max())


RECORDED = sorted(k for k in R.data("scipy") if k.startswith("eigen/eigs"))


@pytest.mark.parametrize("key", RECORDED)
def test_recorded_eigenvalues_are_scipys(key):
    """tools/gen_sparse_scipy.py's eigsh / eigs records (k = 6, v0 = ones, the values only)."""
    rec = R.data("scipy")[key]
    _, fn, fam, size, which = key.split("/")
    m = M.get(f"{fam}/{size}")
    assert rec["digest"] == M.digest(m)
    r = getattr(S, fn)(sparse(m), k=6, which=which, v0=sd.ones(m.shape[0]), return_eigenvectors=False)
    assert r.converged
    w = npy(r.eigenvalues)
    if fn == "eigsh":
        np.testing.assert_allclose(np.sort(w), rec["eigenvalues"], rtol=1e-10)
    else:
        ref = np.asarray([complex(*z) if isinstance(z, list) else z for z in rec["eigenvalues"]])
        w = w[np.lexsort((w.imag, w.real))]
        np.testing.assert_allclose(w, ref, rtol=1e-9)


# ---------------------------------------------------------------- the contract (Arb)


def _norm2(m):
    return float(np.linalg.norm(m.dense(), 2))


EIG_POINTS = [p for p in R.points("eig") if p["kind"] in ("closed", "arb")]


@pytest.mark.parametrize("p", EIG_POINTS, ids=[p["matrix"] for p in EIG_POINTS])
def test_eigsh_within_the_contract(p):
    """The 6 smallest and largest eigenvalues within C_EIG eps ||A||_2 of Arb's; the residuals
    ||A v - w v|| <= C_EIG eps ||A||_2 and V^H V = I to C_EIG k eps."""
    m = M.get(p["matrix"])
    assert p["digest"] == M.digest(m)
    A = sparse(m)
    n = m.shape[0]
    norm = 4.0 * int(p["matrix"][3]) if p["kind"] == "closed" else _norm2(m)
    k = min(6, n - 1)
    for which, key in (("SA", "smallest"), ("LA", "largest")):
        ref = (p[f"{key}_hi"] + p[f"{key}_lo"])[:k]
        kw = {"which": which}
        if which == "SA" and (n > 2000 or p["matrix"].startswith("graded")):
            # shift-invert for the smallest of the big Laplacians (speed) and of graded/100x8
            # (kappa 1e10: plain Lanczos does not converge in 10 n restarts, nor does scipy's)
            kw = {"sigma": -1.0 if n > 2000 else 0.0}
        r = S.eigsh(A, k=k, **kw)
        assert r.converged
        w = npy(r.eigenvalues)
        got = np.sort(w) if key == "smallest" else np.sort(w)[::-1]
        assert np.abs(got - ref).max() <= R.C_EIG * EPS * norm, (which, np.abs(got - ref).max() / (EPS * norm))
        V = npy(r.eigenvectors)
        As = m.scipy("csr")
        assert np.abs(As @ V - V * w).max() <= R.C_EIG * EPS * norm
        assert np.abs(V.conj().T @ V - np.eye(k)).max() <= R.C_EIG * k * EPS


SVD_POINTS = [p for p in R.points("eig") if p["kind"] == "svd"]


@pytest.mark.parametrize("p", SVD_POINTS, ids=[p["matrix"] for p in SVD_POINTS])
@pytest.mark.parametrize("which", ["LM", "SM"])
def test_svds_within_the_contract(p, which):
    """The k largest / smallest singular values within C_EIG eps ||A||_2 of Arb's; A v = s u and
    A^H u = s v to that scale; U, V orthonormal; S ascending."""
    m = M.get(p["matrix"])
    ref = p["svdvals_hi"] + p["svdvals_lo"]  # descending
    k = 6
    r = S.svds(sparse(m), k=k, which=which)
    assert r.converged
    s = npy(r.S)
    assert np.all(np.diff(s) >= 0)
    want = np.sort(ref[:k] if which == "LM" else ref[-k:])
    norm = float(ref[0])
    assert np.abs(s - want).max() <= R.C_EIG * EPS * norm
    U, Vh = npy(r.U), npy(r.Vh)
    As = m.scipy("csr")
    assert np.abs(As @ Vh.conj().T - U * s).max() <= R.C_EIG * EPS * norm
    assert np.abs(As.conj().T @ U - Vh.conj().T * s).max() <= R.C_EIG * EPS * norm
    assert np.abs(U.conj().T @ U - np.eye(k)).max() <= R.C_EIG * k * EPS
    assert np.abs(Vh @ Vh.conj().T - np.eye(k)).max() <= R.C_EIG * k * EPS


NONSYM_POINTS = [p for p in R.points("eig") if p["kind"] == "nonsym"]


@pytest.mark.parametrize("p", NONSYM_POINTS, ids=[p["matrix"] for p in NONSYM_POINTS])
@pytest.mark.parametrize("which", ["LM", "LR", "SR", "LI"])
def test_eigs_within_the_contract(p, which):
    """Each eigenvalue within C_EIG eps ||A||_2 kappa(lambda) of the nearest of Arb's (kappa the
    eigenvalue's condition number 1 / |y^H x|), and its residual within C_EIG eps ||A||_2."""
    import scipy.linalg as sl

    m = M.get(p["matrix"])
    ref = p["eigvals_hi"] + p["eigvals_lo"]
    D = m.dense()
    lam, Y, X = sl.eig(D, left=True, right=True)
    norm = _norm2(m)
    r = S.eigs(sparse(m), k=4, which=which)
    assert r.converged
    w, V = npy(r.eigenvalues), npy(r.eigenvectors)
    for i, z in enumerate(w):
        j = int(np.argmin(np.abs(lam - z)))
        kappa = 1.0 / abs(np.vdot(Y[:, j], X[:, j]) / (np.linalg.norm(Y[:, j]) * np.linalg.norm(X[:, j])))
        assert np.abs(ref - z).min() <= R.C_EIG * EPS * norm * kappa
        assert np.linalg.norm(D @ V[:, i] - z * V[:, i]) <= R.C_EIG * EPS * norm


# ---------------------------------------------------------------- R (RSpectra, irlba)

_R_EIGEN = R.DATA / "r_eigen.json"
RRECORDS = sorted(k for k in R.data("r_eigen")) if _R_EIGEN.exists() else []


@pytest.mark.skipif(not _R_EIGEN.exists(), reason="tests/sparse/data/r_eigen.json: tools/gen_sparse_r.py --eigen")
def test_r_record_provenance():
    prov = R.data_provenance("r_eigen")
    assert prov["RSpectra"] == "0.16.2" and prov["irlba"] == "2.4.1"


@pytest.mark.parametrize("key", RRECORDS)
def test_r_comparators_agree(key):
    """Each of RSpectra's (irlba's) values is within 1e-9 relative of a true one (numpy's dense
    eigenvalues), and sd's k values are the true wanted ones: where R finds the wanted set (it may
    miss a multiplicity, as on lap2d/30), R and sd agree."""
    rec = R.data("r_eigen")[key]
    parts = key.split("/")
    fn, mid = parts[1], "/".join(parts[2:4])
    m = M.get(mid)
    assert rec["digest"] == M.digest(m)
    D = m.dense()
    if fn == "svds":
        true = np.linalg.svd(D, compute_uv=False)
        mine = np.sort(npy(S.svds(sparse(m), k=6).S))[::-1]
        np.testing.assert_allclose(mine, true[:6], rtol=1e-11)
        for lib in ("RSpectra", "irlba"):
            for v in rec[lib]:
                assert np.abs(true - v).min() <= 1e-9 * true[0]
        np.testing.assert_allclose(rec["RSpectra"], mine, rtol=1e-9)
        return
    which = parts[4]
    if fn == "eigs_sym":
        true = np.linalg.eigvalsh(D)
        kw = {"sigma": 0.0} if which == "sigma0" else {"which": which}
        mine = np.sort(npy(S.eigsh(sparse(m), k=6, **kw).eigenvalues))
        for v in rec["eigenvalues"]:
            assert np.abs(true - v).min() <= 1e-9 * np.abs(true).max()
        if len(set(np.round(rec["eigenvalues"], 8))) == len(set(np.round(mine, 8))):
            np.testing.assert_allclose(rec["eigenvalues"], mine, rtol=1e-9, atol=1e-9)
        return
    true = np.linalg.eigvals(D)
    got = np.asarray(rec["real"]) + 1j * np.asarray(rec["imag"])
    mine = npy(S.eigs(sparse(m), k=6, which=which).eigenvalues)
    mine = mine[np.lexsort((mine.imag, mine.real))]
    for v in got:
        assert np.abs(true - v).min() <= 1e-7 * np.abs(true).max()
    np.testing.assert_allclose(got, mine, rtol=1e-7)


# ---------------------------------------------------------------- svds


@pytest.mark.parametrize("shape", [(200, 120), (120, 200), (150, 150)])
@pytest.mark.parametrize("which", ["LM", "SM"])
@pytest.mark.parametrize("complex_", [False, True])
def test_svds_matches_the_dense_svd(shape, which, complex_):
    """Tall, wide and square, real and complex: numpy's singular values, ascending; the vectors'
    residuals; return_singular_vectors 'u' / 'vh' / False leave the others None."""
    g = np.random.default_rng(11)
    A = sp.random_array(shape, density=0.05, rng=g)
    if complex_:
        A = A + 1j * sp.random_array(shape, density=0.05, rng=g)
    A = (A + sp.eye_array(*shape)).tocsr()
    s_ref = np.linalg.svd(A.toarray(), compute_uv=False)
    k = 5
    r = S.svds(_sd(A), k=k, which=which)
    assert r.converged
    want = np.sort(s_ref[:k] if which == "LM" else s_ref[-k:])
    np.testing.assert_allclose(npy(r.S), want, rtol=1e-10)
    U, Vh = npy(r.U), npy(r.Vh)
    assert U.shape == (shape[0], k) and Vh.shape == (k, shape[1])
    assert np.linalg.norm(A @ Vh.conj().T - U * npy(r.S)) <= 1e-12 * s_ref[0]
    for opt, has_u, has_v in (("u", True, False), ("vh", False, True), (False, False, False)):
        q = S.svds(_sd(A), k=k, which=which, return_singular_vectors=opt)
        assert (q.U is not None) == has_u and (q.Vh is not None) == has_v
        assert bits_equal(npy(q.S), npy(r.S))


@LIVE
def test_svds_agrees_with_scipy():
    """The recorded svds values (scipy's default solver, AA^H's eigsh) and scipy live."""
    for key in sorted(k for k in R.data("scipy") if k.startswith("eigen/svds")):
        rec = R.data("scipy")[key]
        m = M.get("/".join(key.split("/")[2:4]))
        s = npy(S.svds(sparse(m), k=6).S)
        np.testing.assert_allclose(s[::-1], rec["singular_values"], rtol=1e-10)


def test_svds_on_a_linear_operator():
    """A LinearOperator with rmatvec works (its products go through Python), the same values."""
    m = M.get("rect/60x20")
    A = sparse(m)
    L = S.LinearOperator(A.shape, lambda x: A @ x, rmatvec=lambda x: A.T @ x, dtype=sd.float64)
    assert bits_equal(npy(S.svds(L, k=3).S), npy(S.svds(A, k=3).S))
    with pytest.raises(ValueError, match="without 'rmatvec'"):
        S.svds(S.LinearOperator(A.shape, lambda x: A @ x, dtype=sd.float64), k=3)


# ---------------------------------------------------------------- the start vector, the bits


def test_default_start_vector_is_fixed_and_documented():
    """v0 = None is 2 u - 1, u = sd.random.default_rng(0).random(n) (complex: consecutive pairs);
    rng= draws it from that generator instead."""
    A = _sd(_RSYM)
    a, b = S.eigsh(A, k=3), S.eigsh(A, k=3)
    assert bits_equal(npy(a.eigenvalues), npy(b.eigenvalues))
    u = sd.random.default_rng(0).random(_N) * 2.0 - 1.0
    c = S.eigsh(A, k=3, v0=u)
    assert bits_equal(npy(a.eigenvalues), npy(c.eigenvalues)) and a.n_op == c.n_op
    d = S.eigsh(A, k=3, rng=sd.random.default_rng(5))
    e = S.eigsh(A, k=3, v0=sd.random.default_rng(5).random(_N) * 2.0 - 1.0)
    assert bits_equal(npy(d.eigenvalues), npy(e.eigenvalues))
    np.testing.assert_allclose(npy(d.eigenvalues), npy(a.eigenvalues), rtol=1e-10)
    z = sd.random.default_rng(0).random(2 * 300) * 2.0 - 1.0
    zc = (z[0::2] + z[1::2] * 1j).copy()
    assert bits_equal(npy(S.eigs(_sd(_CPLX), k=3).eigenvalues), npy(S.eigs(_sd(_CPLX), k=3, v0=zc).eigenvalues))


def _big():
    n = 40000
    t = np.arange(n, dtype=np.float64)
    return S.diags_array([sd.asarray(1.0 + t / n * 3.0), sd.full(n - 1, 0.2), sd.full(n - 1, 0.2)],
                         offsets=[0, 1, -1], format="csr")  # fmt: skip


@pytest.mark.usefixtures("restore_threads")
def test_same_bits_on_every_thread_count():
    """n = 4e4: the pool splits the products, the inner products and the basis updates."""
    A = _big()
    U = A + S.diags_array([sd.full(39999, 0.1)], offsets=[1], format="csr")
    first = None
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        got = [S.eigsh(A, k=3, ncv=12), S.eigs(U, k=3, ncv=12), S.svds(A, k=3, ncv=12)]
        vals = [npy(got[0].eigenvectors), npy(got[1].eigenvectors), npy(got[2].U)]
        if first is None:
            first = vals
            assert all(g.converged for g in got)
        else:
            assert all(bits_equal(a, f) for a, f in zip(vals, first))


@pytest.mark.parametrize("fn", ["eigsh", "eigs", "svds"])
def test_same_bits_for_every_format_and_operator(fn):
    """CSR, CSC, COO, DOK, a dense sd.ndarray, aslinearoperator(A) and a Python LinearOperator
    v -> A @ v give the same bits."""
    m = M.get({"eigsh": "random_spd/60", "eigs": "random_unsym/50x5", "svds": "rect/60x20"}[fn])
    A = sparse(m)
    forms = [A, A.tocsc(), A.tocoo(), A.todok(), A.toarray(), S.aslinearoperator(A),
             S.LinearOperator(A.shape, lambda x: A @ x, rmatvec=lambda x: A.T @ x, dtype=sd.float64)]  # fmt: skip
    f = getattr(S, fn)
    out = [f(X, k=4) for X in forms]
    key = "S" if fn == "svds" else "eigenvalues"
    first = npy(getattr(out[0], key))
    for r in out[1:]:
        assert bits_equal(npy(getattr(r, key)), first)


# ---------------------------------------------------------------- the results


def test_results_unpack_like_scipy():
    A = _sd(_RSYM)
    r = S.eigsh(A, k=3)
    w, v = r
    assert len(r) == 2 and r[0] is r.eigenvalues and v is r.eigenvectors
    assert tuple(v.shape) == (_N, 3)
    w2, v2 = S.eigsh(A, k=3, return_eigenvectors=False)
    assert v2 is None and tuple(w2.shape) == (3,)
    u, s, vh = S.svds(A, k=3)
    assert tuple(u.shape) == (_N, 3) and tuple(vh.shape) == (3, _N)
    with pytest.raises(ValueError):
        r.eigenvalues[0] = 1.0  # read-only
    with pytest.raises(AttributeError):
        r.eigenvalues = None
    assert repr(r).startswith("EigshResult(converged=True, status=0, n_iter=")
    # a short real array written out, as every result's repr (API.md §2.3, SP113); complex by shape
    assert repr(r).endswith(", eigenvalues=[" + ", ".join(f"{e:.6g}" for e in r.eigenvalues.tolist()) + "])")
    assert "eigenvalues=<array of shape (2,)>" in repr(S.eigs(_sd(_UNSYM), k=2))
    text = r.summary()
    assert "Lanczos Eigenvalue Results" in text and "the 3 wanted eigenvalues converged" in text
    assert "Arnoldi Eigenvalue Results" in S.eigs(_sd(_UNSYM), k=2).summary()
    assert "Singular Value Results" in S.svds(A, k=2).summary()


def test_iteration_limit_reports_the_converged_pairs():
    """At max_iter: status 1, converged False, the pairs that converged (scipy raises
    ArpackNoConvergence with them), the message says how many."""
    A = _sd(_RSYM)
    r = S.eigsh(A, k=6, which="SM", max_iter=3)
    assert r.status == 1 and not r.converged
    assert r.n_converged == r.eigenvalues.shape[0] < 6
    assert r.message == f"the iteration limit was reached: {r.n_converged} of 6 eigenvalues converged"
    assert tuple(r.eigenvectors.shape) == (_N, r.n_converged)
    q = S.svds(_sd(_lap1(200)), k=5, which="SM", max_iter=2)
    assert q.status == 1 and not q.converged and "singular values converged" in q.message


# ---------------------------------------------------------------- inputs and errors


def test_errors():
    A = _sd(_RSYM)
    with pytest.raises(TypeError, match=r"eigsh\(\): 'A' must be a sparse array, an sd.ndarray or a LinearOperator"):
        S.eigsh([[1.0]])
    with pytest.raises(ValueError, match=r"'A' must be square"):
        S.eigsh(_sd(sp.random_array((5, 4), density=0.5, rng=np.random.default_rng(0))))
    with pytest.raises(
        ValueError,
        match=r"eigsh\(\): 'k' must be < n = 400 for 'A' of shape \(400, 400\); for all "
        r"eigenvalues use sd.linalg.eigh\(A.toarray\(\)\)",
    ):
        S.eigsh(A, k=400)
    with pytest.raises(ValueError, match=r"eigs\(\): 'k' must be < n - 1 = 399"):
        S.eigs(A, k=399)
    with pytest.raises(ValueError, match=r"'k' must be >= 1"):
        S.eigsh(A, k=0)
    with pytest.raises(TypeError, match=r"'k' must be an int"):
        S.eigsh(A, k=2.0)
    with pytest.raises(ValueError, match=r"'ncv' must satisfy k < ncv <= n = 400, got 3"):
        S.eigsh(A, k=3, ncv=3)
    with pytest.raises(ValueError, match=r"'ncv' must satisfy k \+ 1 < ncv"):
        S.eigs(A, k=3, ncv=4)
    with pytest.raises(ValueError, match=r"eigsh\(\): 'which' must be 'LM', 'SM', 'LA', 'SA' or 'BE', not 'LR'"):
        S.eigsh(A, which="LR")
    with pytest.raises(TypeError, match=r"'which' must be"):
        S.eigs(A, which=None)
    with pytest.raises(ValueError, match=r"mode='cayley' needs 'sigma'"):
        S.eigsh(A, mode="cayley")
    with pytest.raises(ValueError, match=r"'mode' must be 'normal', 'buckling' or 'cayley'"):
        S.eigsh(A, sigma=1.0, mode="x")
    with pytest.raises(ValueError, match=r"'OPinv' needs 'sigma'"):
        S.eigsh(A, OPinv=A)
    with pytest.raises(ValueError, match=r"'Minv' needs 'M'"):
        S.eigsh(A, Minv=A)
    with pytest.raises(ValueError, match=r"'Minv' cannot be given with 'sigma'"):
        S.eigsh(A, sigma=1.0, M=A, Minv=A)
    with pytest.raises(ValueError, match=r"'tol' must be a finite number >= 0"):
        S.eigsh(A, tol=-1.0)
    with pytest.raises(ValueError, match=r"'max_iter' must be >= 1"):
        S.eigsh(A, max_iter=0)
    with pytest.raises(ValueError, match=r"'v0' must not be zero"):
        S.eigsh(A, v0=sd.zeros(400))
    with pytest.raises(ValueError, match=r"'v0' must have shape \(400,\)"):
        S.eigsh(A, v0=sd.ones(3))
    with pytest.raises(TypeError, match=r"'rng' must be an sd.random.Generator or None"):
        S.eigsh(A, rng=0)
    with pytest.raises(TypeError, match=r"'sigma' must be an int or a float or None, not complex"):
        S.eigsh(A, sigma=1j)
    with pytest.raises(ValueError, match=r"OPpart='i' needs a complex 'sigma'"):
        S.eigs(A, sigma=1.0, OPpart="i")
    with pytest.raises(ValueError, match=r"which='BE' cannot be used with a complex problem"):
        S.eigsh(A.astype(sd.complex128), which="BE")
    with pytest.raises(ValueError, match=r"svds\(\): 'k' must satisfy 1 <= k < min\(A.shape\) = 400, got 400"):
        S.svds(A, k=400)
    with pytest.raises(ValueError, match=r"'return_singular_vectors' must be True, False, 'u' or 'vh', not 'x'"):
        S.svds(A, return_singular_vectors="x")
    with pytest.raises(ValueError, match=r"svds\(\): 'which' must be 'LM' or 'SM', not 'LA'"):
        S.svds(A, which="LA")
    with pytest.raises(TypeError, match="unexpected keyword argument 'maxiter'"):
        S.eigsh(A, maxiter=10)
    with pytest.raises(TypeError, match="unexpected keyword argument 'solver'"):
        S.svds(A, solver="arpack")
    with pytest.raises(ValueError, match=r"'A' contains NaN or infinity"):
        S.eigsh(_sd(sp.diags_array(np.r_[np.nan, np.ones(9)]).tocsr()), k=2)
    with pytest.raises(sd.linalg.LinAlgError, match=r"eigsh\(\): singular matrix"):
        S.eigsh(_sd(sp.diags_array(np.r_[0.0, np.arange(1.0, 10.0)]).tocsr()), k=2, sigma=0.0)


def _ring(m, diagonal=0.1, cycle=0.9):
    """diagonal on the diagonal and cycle around the cycle: partial pivoting grows the entries."""
    shift = S.diags_array([sd.ones(m - 1), sd.ones(1)], offsets=[-1, m - 1], format="csr")
    return diagonal * S.eye_array(m, format="csr") + cycle * shift


def _shift_invert(A, **kw):
    """(result, whether the shift-invert operator of the call refined its solves)."""
    from stardust.sparse import _eigen

    made = []
    init = _eigen._Inverse.__init__

    def spy(self, *a, **k):
        init(self, *a, **k)
        made.append(self)

    _eigen._Inverse.__init__ = spy
    try:
        r = S.eigs(A, **kw) if kw.pop("general", False) else S.eigsh(A, **kw)
    finally:
        _eigen._Inverse.__init__ = init
    (op,) = made
    return r, op.refine


# (m, diagonal, cycle): the rings whose eigenpairs miss the contract when the LU solve is not refined
# (residuals 3.7e3 - 7.1e4 eps ||A||_2; the first solve's normwise backward error 1.5e3 - 4.2e4 eps)
RINGS_OVER = [(10, 0.1, 0.9), (40, 0.1, 0.9), (60, 0.1, 0.9), (10, 0.02, 0.98), (20, 0.02, 0.98), (40, 0.02, 0.98)]


@pytest.mark.parametrize("m, diagonal, cycle", RINGS_OVER)
def test_shift_invert_refines_an_inaccurate_solve(m, diagonal, cycle):
    """The operator's first product measures the normwise backward error of its solve and refines
    from then on when it is past the eigenpairs' contract: the residuals are then inside it."""
    A = _ring(m, diagonal, cycle)
    r, refined = _shift_invert(A, k=3, sigma=0, ncv=min(30, m - 1), general=True)
    assert refined is True and r.converged
    w, v = npy(r.eigenvalues), npy(r.eigenvectors)
    C = npy(A.toarray())
    bound = R.C_EIG * EPS * np.linalg.norm(C, 2)
    assert max(np.linalg.norm(C @ v[:, j] - w[j] * v[:, j]) for j in range(3)) <= bound
    exact = diagonal + cycle * np.exp(2j * np.pi * np.arange(m) / m)  # the circulant's eigenvalues
    assert max(np.abs(exact - z).min() for z in w) <= bound


def test_shift_invert_ring_values():
    """The reported ring (n = 40, 0.1 and 0.9): eigenvalues and residuals to 100 eps (1e-11 unrefined)."""
    A = _ring(40)
    r, _ = _shift_invert(A, k=3, sigma=0, ncv=30, general=True)
    w, v = npy(r.eigenvalues), npy(r.eigenvectors)
    C = npy(A.toarray())
    exact = 0.1 + 0.9 * np.exp(2j * np.pi * np.arange(40) / 40)
    exact = exact[np.argsort(np.abs(exact))[:3]]
    assert np.max(np.abs(np.sort_complex(w) - np.sort_complex(exact))) <= 100 * EPS
    assert max(np.linalg.norm(C @ v[:, j] - w[j] * v[:, j]) for j in range(3)) <= 100 * EPS


@pytest.mark.parametrize("id_, sigma", [("lap2d/30", 0.0), ("lap2d/30", 0.5), ("lap2d/100", 0.5), ("lap2d/100", 2.7),
                                        ("lap3d/10", 1.3), ("saddle/600x150", 0.5)])  # fmt: skip
def test_shift_invert_keeps_one_solve_inside_the_contract(id_, sigma):
    """A shift inside the spectrum (and the saddle's pivoting) leaves a large componentwise error but
    eigenpairs well inside the contract: the fast path, one solve a product."""
    A = sparse(M.get(id_))
    r, refined = _shift_invert(A, k=4, sigma=sigma)
    assert refined is False and r.converged
    w, V = npy(r.eigenvalues), npy(r.eigenvectors)
    As = sp.csr_array(npy(A.toarray()))
    norm = spla.norm(As, 1)  # >= ||A||_2 for a symmetric matrix
    assert np.abs(As @ V - V * w).max() <= R.C_EIG * EPS * norm


def test_start_vector_in_the_null_space():
    """OP v0 = 0: ARPACK's 'starting vector is zero' (scipy's ArpackError) is a ValueError."""
    D = _sd(sp.diags_array(np.r_[np.zeros(5), np.ones(5)]).tocsr())
    with pytest.raises(ValueError, match=r"the start vector OP v0 is zero"):
        S.eigsh(D, k=2, v0=sd.asarray(np.r_[np.ones(5), np.zeros(5)]))


def test_linalg_points_to_the_eigensolvers():
    A = _sd(_RSYM)
    for fn, hint in (("eigh", "eigsh"), ("eigvalsh", "eigsh"), ("eig", "eigs"), ("svd", "svds")):
        with pytest.raises(TypeError, match=rf"for a sparse array use sd.sparse.{hint}$"):
            getattr(sd.linalg, fn)(A)


def test_complex_shift_without_vectors_still_gives_the_values():
    """A real A with a complex sigma: the values are Rayleigh quotients of the Ritz vectors, so
    return_eigenvectors=False computes the vectors inside (scipy 1.18.1 returns zeros there)."""
    A = _sd(_UNSYM[:200, :200].tocsr())
    for part in ("r", "i"):
        a = S.eigs(A, k=3, sigma=0.5 + 0.3j, OPpart=part)
        b = S.eigs(A, k=3, sigma=0.5 + 0.3j, OPpart=part, return_eigenvectors=False)
        assert b.eigenvectors is None
        assert bits_equal(npy(a.eigenvalues), npy(b.eigenvalues))
        assert np.abs(npy(b.eigenvalues)).min() > 0.1


def test_message_singular_for_one_and_n_iter_at_the_limit():
    """k = 1 says "eigenvalue" (R231); at the iteration limit eigsh and eigs count max_iter + 1
    restarts (ARPACK's iparam(3), scipy's), svds max_iter (R232, docs/api/sparse.md §7)."""
    A = S.eye_array(30, format="csr") * 0.5 + S.diags_array(0.1, offsets=1, shape=(30, 30), format="csr")
    assert S.eigs(A, k=1).message == "the 1 wanted eigenvalue converged"
    L = S.diags_array([-1.0, 2.0, -1.0], offsets=[-1, 0, 1], shape=(100, 100), format="csr")
    assert S.eigsh(L, k=1).message == "the 1 wanted eigenvalue converged"
    assert S.svds(L, k=1).message == "the 1 wanted singular value converged"
    assert S.eigsh(L, k=2).message == "the 2 wanted eigenvalues converged"
    r = (S.eigsh(L, k=3, which="SA", max_iter=5), S.eigs(L, k=3, which="SR", max_iter=5),
         S.svds(L, k=3, which="SM", max_iter=5))  # fmt: skip
    assert [x.n_iter for x in r] == [6, 6, 5] and not any(x.converged for x in r)
    assert r[0].message.endswith(" of 3 eigenvalues converged") and r[2].message.endswith(
        " of 3 singular values converged"
    )


@pytest.mark.usefixtures("restore_threads")
def test_complex_hermitian_sorted_on_every_thread_count():
    """R229: the complex Hermitian eigsh's ascending values and their vectors, the same bits for
    1 and 10 threads, 'LA' and 'SA' both ascending."""
    n = 100
    L = S.diags_array([-1.0, 2.0, -1.0], offsets=[-1, 0, 1], shape=(n, n), format="csr")
    sh = S.diags_array(sd.ones(n - 1), offsets=1, shape=(n, n), format="csr")
    H = (L + 1j * (sh - sh.T) * 0.5).tocsr()
    Hd = npy(H.toarray())
    ref = np.linalg.eigvalsh(Hd)
    out = []
    for t in (1, 10):
        sd.set_num_threads(t)
        got = []
        for which, want in (("LA", ref[-3:]), ("SA", ref[:3])):
            r = S.eigsh(H, k=3, which=which)
            w, V = npy(r.eigenvalues), npy(r.eigenvectors)
            assert np.all(np.diff(w) > 0)
            np.testing.assert_allclose(w, want, rtol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(Hd @ V - V * w, axis=0), 0, atol=1e-12)
            got += [w.tobytes(), V.tobytes()]
        out.append(got)
    assert out[0] == out[1]
