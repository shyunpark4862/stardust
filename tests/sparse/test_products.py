"""The products of SP2 (docs/api/sparse.md §3.1, docs/design/sparse.md §2.6, §3.2, §4): sparse @
dense (SpMV, SpMM), dense @ sparse and sparse @ sparse (SpGEMM).

The order of a sum is the definition (§2.6): each output entry adds its terms in ascending index
order from zero, fma(a, x, acc) per real term, acc + a*x (numpy's complex product) per complex
term, the wrapping product and sum per integer term; a line of more than L = 4096 terms sums
pieces of L and joins them along the pairwise tree. So the tests compare bit for bit with a Python
model of that order, with scipy 1.18.1 (whose csr_matvec and csr_matmat are the same order on
lines up to L), across the four formats and across sd thread counts; and against the Arb
references of tests/sparse/ref/spmv.npz in the contract's units (refdata.spmv_error <= 1)."""

import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import matrices as M
from . import refdata as R
from .conftest import LIVE, coo, npy

S = sd.sparse
FORMATS = ("coo", "csr", "csc", "dok")
SPMV_POINTS = R.points("spmv")
L = R.L


def cmul(a, b):
    """numpy's complex product (array/kernels/scalar_ops.hpp cmul)."""
    return complex(math.fma(a.real, b.real, -(a.imag * b.imag)), math.fma(a.real, b.imag, a.imag * b.real))


def term(a, x, acc):
    if isinstance(acc, complex):
        p = cmul(complex(a), complex(x))
        return complex(acc.real + p.real, acc.imag + p.imag)
    if isinstance(acc, int):
        return ((acc + int(a) * int(x) + 2**63) % 2**64) - 2**63
    return math.fma(float(a), float(x), acc)


def ordered_sum(pairs, zero):
    """sum of a*x over the pairs in order, pieces of L joined as the pairwise tree does for fewer
    than 16 pieces (a sequential sum from 0)."""
    if len(pairs) <= L:
        acc = zero
        for a, x in pairs:
            acc = term(a, x, acc)
        return acc
    parts = [ordered_sum(pairs[i : i + L], zero) for i in range(0, len(pairs), L)]
    assert len(parts) < 16
    acc = zero
    for p in parts:
        acc = acc + p
    return acc


def model_matvec(m, x, transpose=False):
    """y = A x (or A^T x) in the order of §2.6, from the canonical entries of matrix m."""
    r, c, v = m.canonical()
    if transpose:
        r, c = c, r
    n = m.shape[1] if transpose else m.shape[0]
    zero = 0j if (v.dtype.kind == "c" or np.asarray(x).dtype.kind == "c") else (0 if v.dtype.kind in "ib" else 0.0)
    order = np.lexsort((c, r))
    rows = [[] for _ in range(n)]
    for k in order:
        rows[r[k]].append((v[k], x[c[k]]))
    return np.array([ordered_sum(p, zero) for p in rows], dtype=type(zero) if n else np.float64)


def sd_vec(x):
    return sd.asarray(np.ascontiguousarray(x))


def as_format(m, fmt):
    return coo(m).asformat(fmt)


# ---- SpMV and x @ A: the order, every format ----


@pytest.mark.parametrize("p", SPMV_POINTS[::3], ids=lambda p: f"{p['matrix']}:{p['kind']}")
def test_matvec_is_the_ordered_sum_in_every_format(p):
    m = M.get(p["matrix"])
    if m.data.dtype == bool:
        return
    x, w = p["x"], p["w"]
    want_y, want_z = model_matvec(m, x), model_matvec(m, w, transpose=True)
    for fmt in FORMATS:
        A = as_format(m, fmt)
        assert bits_equal(npy(A @ sd_vec(x)), want_y.astype(npy(A @ sd_vec(x)).dtype)), fmt
        if m.shape[0] or m.shape[1]:
            z = sd_vec(w) @ A
            assert bits_equal(npy(z), want_z.astype(npy(z).dtype)), fmt
            assert bits_equal(npy(A.T @ sd_vec(w)), npy(z))


@pytest.mark.parametrize("p", SPMV_POINTS, ids=lambda p: f"{p['matrix']}:{p['kind']}")
def test_matvec_meets_the_contract(p):
    m = M.get(p["matrix"])
    if m.data.dtype == bool:
        return
    A = as_format(m, "csr")
    x, w = p["x"], p["w"]
    y, z = npy(A @ sd_vec(x)), npy(sd_vec(w) @ A)
    r, c, v = m.canonical()
    abs_y = np.zeros(m.shape[0])
    np.add.at(abs_y, r, np.abs(v * x[c]) if len(v) else 0.0)
    abs_z = np.zeros(m.shape[1])
    np.add.at(abs_z, c, np.abs(v * w[r]) if len(v) else 0.0)
    row_len, col_len = np.bincount(r, minlength=m.shape[0]), np.bincount(c, minlength=m.shape[1])
    assert R.spmv_error(y, p["matvec_hi"], p["matvec_lo"], abs_y, row_len).max(initial=0) <= 1
    assert R.spmv_error(z, p["rmatvec_hi"], p["rmatvec_lo"], abs_z, col_len).max(initial=0) <= 1


@pytest.mark.parametrize(
    "mid", ["lap2d/30", "random_unsym/1000x8", "cancel/400", "random_complex/800", "coo/int", "rect/1200x300",
            "banded/2000x5x5", "long_rows/300"]
)  # fmt: skip
@LIVE
def test_matvec_is_scipys_bits(mid):
    m = M.get(mid)
    s = m.scipy("csr")
    g = np.random.default_rng(5)
    x = g.standard_normal(m.shape[1]) * 10.0 ** g.integers(-8, 9, m.shape[1])
    if m.data.dtype.kind == "c":
        x = x + 1j * g.standard_normal(m.shape[1])
    if m.data.dtype.kind == "i":
        x = g.integers(-9, 10, m.shape[1])
    w = g.standard_normal(m.shape[0])
    for fmt in ("csr", "csc", "coo"):
        A = as_format(m, fmt)
        assert bits_equal(npy(A @ sd_vec(x)), s @ x), fmt
        assert bits_equal(npy(sd_vec(w) @ A), w @ s), fmt
    X = np.stack([x, 2 * x, x[::-1]], axis=1)
    assert bits_equal(npy(as_format(m, "csr") @ sd.asarray(X)), s @ X)


def test_long_lines_are_pieces_joined_by_the_tree():
    # one row and one column of 3 L + 5 terms: the scatter (x @ CSR) on one thread tracks the
    # pieces per output; on several threads it transposes and gathers; both give the gather's bits
    n = 3 * L + 5
    g = np.random.default_rng(8)
    vals = g.standard_normal(n) * 10.0 ** g.integers(-6, 7, n)
    row = np.concatenate([np.zeros(n, np.int64), np.arange(1, 50)])
    col = np.concatenate([np.arange(n), np.zeros(49, np.int64)])
    data = np.concatenate([vals, g.standard_normal(49)])
    A = S.coo_array((sd.asarray(data), (sd.asarray(row), sd.asarray(col))), shape=(n, n)).tocsr()
    x = g.standard_normal(n)
    want = ordered_sum(list(zip(vals, x)), 0.0)
    assert npy(A @ sd.asarray(x))[0] == want
    sequential = 0.0
    for a, b in zip(vals, x):
        sequential = math.fma(a, b, sequential)
    assert want != sequential  # the pieces matter here
    T = A.T.tocsr()  # its column 0 has the long line
    before = sd.get_num_threads()
    try:
        outs = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            outs.append(npy(sd.asarray(x) @ T))
        assert all(bits_equal(o, outs[0]) for o in outs)
        assert outs[0][0] == want
        assert bits_equal(outs[0], npy(A @ sd.asarray(x)))  # x @ A^T scatters, A @ x gathers: one order
    finally:
        sd.set_num_threads(before)


# ---- SpMM and dense @ sparse ----


@pytest.mark.parametrize("fmt", FORMATS)
def test_matmat_columns_are_matvecs(fmt):
    m = M.get("random_complex/40")
    A = as_format(m, fmt)
    g = np.random.default_rng(6)
    X = g.standard_normal((40, 37)) + 1j * g.standard_normal((40, 37))
    Y = npy(A @ sd.asarray(X))
    for j in range(37):
        assert bits_equal(Y[:, j], npy(A @ sd.asarray(np.ascontiguousarray(X[:, j]))))
    Xs = sd.asarray(X)[:, ::2]  # strided
    assert bits_equal(npy(A @ Xs), Y[:, ::2])
    Z = npy(sd.asarray(X.T) @ A)  # (37, 40)
    for j in range(37):
        assert bits_equal(Z[j], npy(sd.asarray(np.ascontiguousarray(X[:, j])) @ A))


def test_shapes_and_scalars_of_one_dimensional_products():
    a = S.csr_array(sd.asarray([1.0, 0.0, 2.0]))
    x = sd.asarray([3.0, 4.0, 5.0])
    r = a @ x
    assert type(r) is float and r == 13.0
    assert type(x @ a) is float and x @ a == 13.0
    X = sd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert (a @ X).tolist() == [11.0, 14.0]
    assert (X.mT @ a).tolist() == [11.0, 14.0]
    B = S.csr_array(sd.asarray([[1, 0], [0, 1], [1, 1]], dtype=sd.bool))
    assert (B @ sd.asarray([True, True])).dtype == sd.int64  # bool @ bool is int64 (the array's rule)
    with pytest.raises(
        ValueError, match=r"^the last axis of 'x1' \(3\) must match the second-to-last axis of 'x2' \(2\)$"
    ):
        a @ sd.asarray([1.0, 2.0])
    with pytest.raises(TypeError):
        a @ 2.0  # scipy: ValueError ("use '*'"); here as for sd.ndarray
    with pytest.raises(TypeError):
        a @ np.ones(3)


# ---- sparse @ sparse ----


def model_matmat(ma, mb):
    """C = A B in the order of §2.6: C_ij = sum over k ascending of a_ik b_kj, from zero."""
    ra, ca, va = ma.canonical()
    rb, cb, vb = mb.canonical()
    complex_ = va.dtype.kind == "c" or vb.dtype.kind == "c"
    zero = 0j if complex_ else (0 if va.dtype.kind in "ib" and vb.dtype.kind in "ib" else 0.0)
    rows_b = {}
    for k, j, v in zip(rb, cb, vb):
        rows_b.setdefault(int(k), []).append((int(j), v))
    out = {}
    for i, k, a in sorted(zip(ra, ca, va), key=lambda t: (t[0], t[1])):
        for j, b in rows_b.get(int(k), []):
            out[(int(i), j)] = term(a, b, out.get((int(i), j), zero))
    C = np.zeros((ma.shape[0], mb.shape[1]), dtype=type(zero) if not isinstance(zero, int) else np.int64)
    for (i, j), v in out.items():
        C[i, j] = v
    return C, {k for k, v in out.items() if v != 0}


@pytest.mark.parametrize("mid", ["random_unsym/50x5", "cancel/50", "random_complex/40", "coo/int", "rect/60x20"])
def test_spgemm_is_the_ordered_sum_in_every_format(mid):
    m = M.get(mid)
    mt = M.Matrix(mid + "^T", m.shape[::-1], m.col, m.row, m.data, frozenset())
    want, stored = model_matmat(m, mt)
    results = []
    for fa in FORMATS:
        for fb in ("csr", "csc"):
            C = as_format(m, fa) @ as_format(mt, fb)
            assert C.format == ("csc" if fa == "csc" else "csr")
            assert C.nnz == len(stored) and C.count_nonzero() == C.nnz  # exact zeros dropped
            results.append(npy(C.toarray()))
    for r in results:
        assert bits_equal(r, want + np.zeros_like(want))


@pytest.mark.parametrize("mid", ["random_unsym/1000x8", "lap2d/30", "random_complex/800", "coo/int"])
@LIVE
def test_spgemm_is_scipys(mid):
    m = M.get(mid)
    A = as_format(m, "csr")
    s = m.scipy("csr")
    C, c = A @ A.T, s @ s.T
    c.sum_duplicates()
    assert C.nnz == c.nnz
    assert bits_equal(npy(C.indptr), c.indptr.astype(np.int64))
    assert bits_equal(npy(C.indices), c.indices.astype(np.int64))
    assert bits_equal(npy(C.data), c.data)


def test_spgemm_one_dimensional():
    v = S.csr_array(sd.asarray([1.0, 0.0, 2.0]))
    A = S.csr_array(sd.asarray([[1.0, 2.0], [0.0, 0.0], [3.0, 0.0]]))
    r = v @ A
    assert r.format == "coo" and r.shape == (2,) and r.toarray().tolist() == [7.0, 2.0]  # scipy: COO
    q = A.T @ v
    assert q.format == "coo" and q.toarray().tolist() == [7.0, 2.0]
    assert v @ v == 5.0 and type(v @ v) is float
    assert S.csr_array(sd.asarray([1.0, 0.0])) @ S.csr_array(sd.asarray([0.0, 1.0])) == 0.0
    with pytest.raises(ValueError, match="must match"):
        A @ A


# ---- every thread count ----


@pytest.mark.usefixtures("restore_threads")
def test_products_and_operations_have_the_same_bits_on_every_thread_count():
    g = np.random.default_rng(11)
    n, nnz = 3000, 200000
    row, col = g.integers(0, n, nnz), g.integers(0, n, nnz)
    data = g.standard_normal(nnz) * 10.0 ** g.integers(-8, 9, nnz)
    A = S.coo_array((sd.asarray(data), (sd.asarray(row), sd.asarray(col))), shape=(n, n)).tocsr()
    B = A.T.tocsr()
    x = sd.asarray(g.standard_normal(n))
    X = sd.asarray(g.standard_normal((n, 5)))

    def run():
        return [npy(A @ x), npy(x @ A), npy(A.tocsc() @ x), npy(A @ X), npy(X.mT @ A), npy((A + B).data),
                npy((A * B).data), npy((A @ B).data), npy(A.sum(axis=0)), npy(A.sum(axis=1)), npy(A.mean(axis=0)),
                npy(A.max(axis=0)), npy(A.argmin(axis=1)), A.sum()]  # fmt: skip

    outs = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        outs.append(run())
    for o in outs[1:]:
        for a, b in zip(o, outs[0]):
            assert bits_equal(a, b)
