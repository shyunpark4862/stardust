"""laplacian (docs/api/graph.md §5, docs/design/graph.md §2.5, §3.6, §4; decisions GR50, GR51, GR53-GR55;
milestone GR4): every entry within the contract of the Arb references (tests/graph/ref/laplacian.npz) for
sparse and dense inputs, integer inputs exact, the pattern (the off-diagonal entries of A, or of A + A^T
by sd.sparse's sum rule, and the whole diagonal) and the values against scipy 1.18.1, -0.0 where A holds
0, the dtype rules, form='lo' and the errors."""

import math

import numpy as np
import pytest
import scipy.sparse as S
import scipy.sparse.csgraph as CS

import stardust as sd

from . import graphs as G
from . import refdata
from .conftest import LIVE
from .forms import csr, square_ids

GR = sd.graph
OPTIONS = [(nm, out, sym) for nm in (False, True) for out in (False, True) for sym in (False, True)]


def _dense_of(g):
    """The values of a test graph as a dense float64 matrix (0 where nothing is stored)."""
    return g.dense(0.0)


def _abs_pairs(g):
    a = {}
    for i, j, w in g.edges():
        if i != j:
            a[(i, j)] = abs(float(w))
    return a


def test_reference_entries_within_the_contract():
    r = refdata.ref("laplacian")
    m = r.methods["L"]
    worst = 0.0
    for j, k in enumerate(m.index):
        g = G.get(str(r.inputs["graph"][k]))
        normed = bool(r.inputs["normed"][k])
        out = bool(r.inputs["use_out_degree"][k])
        sym = bool(r.inputs["symmetrized"][k])
        n = g.n
        hi, lo = m.hi[j].reshape(n, n), m.lo[j].reshape(n, n)
        terms = refdata.laplacian_terms(n, g.edges(), sym, out)
        absA = _abs_pairs(g)
        dense_in = _dense_of(g) if g.data.dtype == np.float64 else _dense_of(g).astype(g.data.dtype)
        for form, A in (("sparse", csr(g)), ("dense", dense_in)):
            L = GR.laplacian(A, normed=normed, use_out_degree=out, symmetrized=sym)
            got = np.asarray(L.toarray() if form == "sparse" else L, dtype=np.float64)
            for a in range(n):
                for b in range(n):
                    pair = absA.get((a, b), 0.0) + absA.get((b, a), 0.0)
                    bd = refdata.laplacian_bound(normed, sym, a, b, terms, hi[a, b], pair)
                    u = refdata.laplacian_units(float(got[a, b]), hi[a, b], lo[a, b], bd)
                    assert u <= 1.0, (g.id, form, normed, out, sym, a, b, got[a, b], hi[a, b])
                    worst = max(worst, u)
            if g.data.dtype != np.float64 or "int" in g.tags or "dyadic" in g.tags:
                exact = hi + lo
                assert np.array_equal(got, exact) or normed, (g.id, form)
    assert worst <= 1.0


@pytest.mark.parametrize("gid", square_ids())
def test_sparse_and_dense_agree(gid):
    """A dense matrix of the stored values gives the sparse result's values (its -0.0 aside)."""
    g = G.get(gid)
    if g.n == 0 or g.matrix is not None or not np.all(np.isfinite(g.data.astype(np.float64))):
        return
    A = csr(g)
    D = g.dense(0.0) if g.data.dtype == np.float64 else g.dense(0.0).astype(np.int64)
    for normed, out, sym in OPTIONS:
        if normed and np.any(g.data.astype(np.float64) < 0):
            continue
        s = GR.laplacian(A, normed=normed, use_out_degree=out, symmetrized=sym, return_diag=True)
        d = GR.laplacian(D, normed=normed, use_out_degree=out, symmetrized=sym, return_diag=True)
        assert np.array_equal(np.asarray(s.laplacian.toarray()), np.asarray(d.laplacian)), (normed, out, sym)
        assert np.asarray(s.diag).tobytes() == np.asarray(d.diag).tobytes(), (normed, out, sym)


def _scipy_csr(x):
    x = x.tocsr()
    x.sort_indices()
    return x


@LIVE
@pytest.mark.parametrize("gid", square_ids())
def test_against_scipy(gid):
    """The same pattern as scipy's result (as CSR) and the same values: the off-diagonal entries of the
    unnormalized laplacian bit for bit, the degrees and the normalized entries to a few rounding errors
    (scipy subtracts the diagonal from the full sum), the normalized diagonal exactly."""
    g = G.get(gid)
    if g.n == 0 or g.matrix is not None or g.data.dtype == np.bool_:
        return
    a = g.scipy().copy()
    A = csr(g)
    for normed, out, sym in OPTIONS:
        with np.errstate(all="ignore"):
            ref, rdiag = CS.laplacian(a, normed=normed, use_out_degree=out, symmetrized=sym, return_diag=True)
        ref = _scipy_csr(ref)
        L, diag = GR.laplacian(A, normed=normed, use_out_degree=out, symmetrized=sym, return_diag=True)
        assert np.array_equal(np.asarray(L.indptr), ref.indptr), (normed, out, sym)
        assert np.array_equal(np.asarray(L.indices), ref.indices), (normed, out, sym)
        assert str(L.dtype) == ("float64" if normed else str(a.dtype))
        x, y = np.asarray(L.data, dtype=np.float64), ref.data.astype(np.float64)
        rows = np.repeat(np.arange(g.n), np.diff(ref.indptr))
        diag_place = rows == ref.indices
        if not normed and not sym:
            assert x[~diag_place].tobytes() == y[~diag_place].tobytes()
        assert np.allclose(x, y, rtol=1e-12, atol=1e-300, equal_nan=True), (normed, out, sym)
        if normed:
            assert np.array_equal(x[diag_place], y[diag_place], equal_nan=True)
        assert np.allclose(np.asarray(diag, dtype=np.float64), rdiag, rtol=1e-12, atol=1e-300, equal_nan=True)


@LIVE
def test_negative_zero_where_a_holds_zero():
    """scipy's -0.0 (the GR0 probe read it through toarray(), which adds onto +0.0 and lost it): a stored 0
    gives -0.0 in the sparse result, every 0 of a dense matrix -0.0 off the diagonal."""
    a = S.csr_array((np.array([0.0, 1.0]), ([0, 1], [1, 2])), shape=(3, 3))
    for normed in (False, True):
        ref = _scipy_csr(CS.laplacian(a, normed=normed))
        L = GR.laplacian(sd.sparse.csr_array(a), normed=normed)
        assert np.asarray(L.data).tobytes() == ref.data.tobytes()
        assert np.signbit(np.asarray(L.data)[np.asarray(L.indices) == 1][0])
        dense = np.array([[0, 2.0, 0], [2, 5, 3], [0, 3, 0]])
        ours = np.asarray(GR.laplacian(dense, normed=normed))
        theirs = CS.laplacian(dense, normed=normed)
        assert np.array_equal(np.signbit(ours), np.signbit(theirs))
        assert np.allclose(ours, theirs, rtol=1e-15, atol=0)


def test_values_as_they_are():
    # NaN and inf propagate; the diagonal is left out of the degree (also an inf or NaN one)
    A = np.array([[np.inf, 1.0, 0], [2, np.nan, np.nan], [0, 3, 0]])
    L = np.asarray(GR.laplacian(A))
    assert L[0, 0] == 2 and L[1, 1] == 4 and math.isnan(L[2, 2]) and math.isnan(L[1, 2]) and L[0, 1] == -1
    Ln = np.asarray(GR.laplacian(A, normed=True))
    assert Ln[0, 0] == 1 and Ln[1, 1] == 1  # the diagonal is 1 unless the degree is 0 (scipy's rule)
    # a negative degree: NaN off the diagonal (sqrt), 1 on it
    neg = np.asarray(GR.laplacian(np.array([[0, -1.0], [-1, 0]]), normed=True))
    assert np.isnan(neg[0, 1]) and neg[0, 0] == 1
    # degree 0: divided by 1, 0 on the diagonal
    z = GR.laplacian(np.array([[0, 0, 0], [1.0, 0, 0], [0, 0, 0]]), normed=True, return_diag=True)
    assert np.asarray(z.diag).tolist() == [1.0, 1.0, 1.0]
    assert np.asarray(z.laplacian).tolist() == [[1.0, -0.0, -0.0], [-1.0, 0.0, -0.0], [-0.0, -0.0, 0.0]]


def test_symmetrized_drops_exact_zero_sums():
    """A + A^T by sd.sparse's sum rule: an exact zero sum is not stored (scipy's too); without
    symmetrized a stored 0 stays (as -0.0)."""
    a = sd.sparse.csr_array(
        (sd.asarray([1.0, -1.0, 0.0, 2.0]), sd.asarray([1, 0, 0, 1]), sd.asarray([0, 1, 2, 4])), shape=(3, 3)
    )
    L = GR.laplacian(a, symmetrized=True)
    assert np.asarray(L.indices).tolist() == [0, 1, 2, 1, 2] and np.asarray(L.indptr).tolist() == [0, 1, 3, 5]
    assert np.asarray(L.data).tolist() == [0.0, 2.0, -2.0, -2.0, 2.0]
    S2 = a + a.T
    assert S2.nnz == 2  # the same rule
    L0 = GR.laplacian(a)
    assert np.asarray(L0.indices).tolist() == [0, 1, 0, 1, 0, 1, 2]


def test_dtypes():
    i = np.array([[0, 2, 0], [1, 0, 5], [0, 3, 0]], dtype=np.int64)
    L = GR.laplacian(i)
    assert L.dtype == sd.int64 and np.asarray(L).tolist() == [[1, -2, 0], [-1, 5, -5], [0, -3, 5]]
    assert GR.laplacian(i, use_out_degree=True, return_diag=True).diag.tolist() == [2, 6, 3]
    b = GR.laplacian(i != 0, return_diag=True)
    assert b.laplacian.dtype == sd.int64 and b.diag.dtype == sd.int64
    assert GR.laplacian(sd.sparse.csr_array(i != 0)).dtype == sd.int64
    assert GR.laplacian(i, normed=True).dtype == sd.float64
    assert GR.laplacian(i, dtype=sd.float64).dtype == sd.float64
    assert GR.laplacian(i, dtype="float64", return_diag=True).diag.dtype == sd.float64
    assert GR.laplacian(i.astype(np.float64), normed=True, dtype="int64").dtype == sd.int64
    big = np.array([[0, 2**62], [2**62, 0]], dtype=np.int64)
    assert np.asarray(GR.laplacian(big + big)).tolist()[0][0] == -(2**63)  # int64 wraps (numpy's sum)
    assert GR.laplacian(sd.sparse.csr_array(i)).dtype == sd.int64
    with pytest.raises(TypeError, match=r"laplacian\(\): 'dtype' must be None, int64 or float64, not bool"):
        GR.laplacian(i, dtype="bool")
    with pytest.raises(TypeError, match="'dtype' must be None, int64 or float64, not complex128"):
        GR.laplacian(i, dtype="complex128")
    with pytest.raises(TypeError, match="must be real"):
        GR.laplacian(np.eye(2) * 1j)


def test_results_and_readonly():
    A = sd.sparse.csr_array(np.array([[0, 1.0], [1, 0]]))
    r = GR.laplacian(A, return_diag=True)
    assert type(r).__name__ == "LaplacianResult" and r._fields == ("laplacian", "diag")
    L, d = r
    assert isinstance(L, sd.sparse.csr_array)
    with pytest.raises(ValueError):
        d[0] = 1.0
    dense = GR.laplacian(np.eye(2), return_diag=True)
    with pytest.raises(ValueError):
        dense.laplacian[0, 0] = 1.0
    plain = GR.laplacian(np.eye(2))
    plain[0, 0] = 3.0  # a plain result is a new writable array
    assert GR.laplacian(sd.sparse.csr_array((0, 0))).shape == (0, 0)
    assert GR.laplacian(np.zeros((0, 0)), return_diag=True).diag.shape == (0,)


def _b(x):
    return np.asarray(x).tobytes()


def test_form_lo():
    """form='lo' wraps the matrix of form='array' (decision GR51): its products are that matrix's, bit
    for bit."""
    r = np.random.Generator(np.random.PCG64(7))
    W = r.uniform(0, 1, (40, 40)) * (r.random((40, 40)) < 0.2)
    x = sd.asarray(r.normal(size=40))
    X = sd.asarray(r.normal(size=(40, 3)))
    for A in (sd.sparse.csr_array(W), W):
        for normed, out, sym in OPTIONS:
            L = GR.laplacian(A, normed=normed, use_out_degree=out, symmetrized=sym)
            op, diag = GR.laplacian(A, normed=normed, use_out_degree=out, symmetrized=sym, form="lo", return_diag=True)
            assert isinstance(op, sd.sparse.LinearOperator) and op.shape == (40, 40)
            assert _b(op @ x) == _b(L @ x)
            assert _b(op.matmat(X)) == _b(L @ X)
            assert _b(op.rmatvec(x)) == _b(L.T @ x)
            again = GR.laplacian(A, normed=normed, use_out_degree=out, symmetrized=sym, return_diag=True)
            assert _b(diag) == _b(again.diag)
    assert GR.laplacian(W.astype(np.int64), form="lo").dtype == sd.int64


def test_errors():
    with pytest.raises(ValueError, match=r"laplacian\(\): 'csgraph' must be a square matrix"):
        GR.laplacian(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="'form' must be 'array' or 'lo', not 'function'"):
        GR.laplacian(np.eye(2), form="function")
    with pytest.raises(TypeError, match="'normed' must be True or False"):
        GR.laplacian(np.eye(2), normed=1)
    with pytest.raises(TypeError, match="copy"):
        GR.laplacian(np.eye(2), copy=False)
    with pytest.raises(TypeError, match="positional"):
        GR.laplacian(np.eye(2), True)
    with pytest.raises(TypeError, match=r"sd.sparse.csr_array\(A\)"):
        GR.laplacian(S.eye_array(3, format="csr"))
