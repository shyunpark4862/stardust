"""The canonical form (docs/design/sparse.md §2.3, §2.6): CSR and CSC are always sorted without
repeats; repeated positions are summed left to right in input order; the result is the same bit
for bit for every sd thread count and does not depend on which of the two kernel paths ran (the
direct one below 2^16 entries, the bucket one above, on one thread from 2^17, or 2^20 for a
transpose: csrc/sparse/canonical.cpp).

The reference is tests/sparse/matrices.py's _canonical (a stable sort, then a sequential sum per
position). scipy's own order of summing differs on long lines and in COO.sum_duplicates; the tests
here show where (docs/design/sparse.md §2.3)."""

import numpy as np
import pytest
import scipy.sparse as sp

import stardust as sd

from .._bits import bits_equal
from . import matrices as M
from .conftest import LIVE, coo, npy, scipy_coo

S = sd.sparse
THREADS = (1, 2, 3, 7, 10)


def reference(shape, row, col, data, by_row=True):
    """(indptr, indices, data) of the canonical compressed form (input-order sums; int64 wraps)."""
    with np.errstate(over="ignore"):
        r, c, v = M._canonical(shape, row, col, data, by_row=by_row)
    major, minor = (r, c) if by_row else (c, r)
    n_major = shape[0] if by_row else shape[1]
    indptr = np.zeros(n_major + 1, dtype=np.int64)
    np.cumsum(np.bincount(major, minlength=n_major), out=indptr[1:])
    return indptr, minor, v


def random_coo(shape, nnz, kind, seed):
    g = np.random.Generator(np.random.PCG64([seed, nnz]))
    row, col = g.integers(0, shape[0], nnz), g.integers(0, shape[1], nnz)
    if kind == "float":
        data = g.standard_normal(nnz) * 10.0 ** g.integers(-15, 16, nnz)
    elif kind == "complex":
        data = g.standard_normal(nnz) * 10.0 ** g.integers(-8, 9, nnz) + 1j * g.standard_normal(nnz)
    elif kind == "int":
        data = g.integers(-(2**62), 2**62, nnz)
    else:
        data = g.integers(0, 2, nnz).astype(bool)
    return row, col, data


def sd_coo(shape, row, col, data):
    return S.coo_array((sd.asarray(data), (sd.asarray(row), sd.asarray(col))), shape=shape)


def parts(A):
    return tuple(npy(x) for x in (A.indptr, A.indices, A.data))


# ---- the sum of repeats ----


def test_repeats_sum_left_to_right():
    # [1, 1e16, 1, -1e16] at one position: ((1 + 1e16) + 1) - 1e16 = 0 in float64
    A = S.coo_array(([1.0, 1e16, 1.0, -1e16], ([0, 0, 0, 0], [1, 1, 1, 1])), shape=(2, 2))
    for B in (A.tocsr(), A.tocsc()):
        assert bits_equal(npy(B.data), np.array([0.0]))  # a stored zero: kept
    A.sum_duplicates()
    assert bits_equal(npy(A.data), np.array([0.0])) and A.has_canonical_format
    assert bits_equal(npy(S.coo_array(([1.0, 1e16, 1.0, -1e16], ([0, 0, 0, 0], [1, 1, 1, 1]))).toarray()),
                      np.array([[0.0, 0.0]]))  # fmt: skip


def test_repeats_in_compressed_parts_sum_in_input_order():
    A = S.csr_array(([1e16, 1.0, 1.0, -1e16, 5.0], [3, 3, 0, 3, 1], [0, 5]), shape=(1, 4))
    assert bits_equal(npy(A.indices), np.array([0, 1, 3]))
    assert bits_equal(npy(A.data), np.array([1.0, 5.0, 0.0]))


@pytest.mark.parametrize("kind", ["float", "complex", "int", "bool"])
def test_kinds_of_sum(kind):
    row, col, data = random_coo((40, 30), 3000, kind, 11)
    want = reference((40, 30), row, col, data)
    got = parts(sd_coo((40, 30), row, col, data).tocsr())
    assert all(bits_equal(g, w) for g, w in zip(got, want))


def test_int_repeats_wrap_and_bool_repeats_or():
    A = S.coo_array(([2**62, 2**62], ([0, 0], [0, 0])))
    assert npy(A.tocsr().data)[0] == -(2**63)  # as scipy and numpy
    B = S.coo_array(([True, True, False], ([0, 0, 0], [0, 0, 1])))
    assert bits_equal(npy(B.tocsr().data), np.array([True, False]))


@LIVE
@pytest.mark.parametrize("mid", M.ids(tag="dups") + M.ids(tag="coo"))
def test_matches_scipy_on_short_lines(mid):
    # scipy's tocsr sums a line of up to 15 entries in input order too (SP0 record)
    m = M.get(mid)
    s = scipy_coo(m).tocsr()
    s.sum_duplicates()
    got = coo(m).tocsr()
    assert bits_equal(npy(got.indptr), s.indptr.astype(np.int64))
    assert bits_equal(npy(got.indices), s.indices.astype(np.int64))
    assert bits_equal(npy(got.data), s.data)
    want = reference(m.shape, m.row, m.col, m.data)
    assert bits_equal(npy(got.data), want[2])


@LIVE
def test_where_scipy_sums_in_another_order():
    # one row of 40 entries in 8 positions, values of mixed magnitude: input order (sd) and
    # scipy's unstable csr_sort_indices / COO reduceat give different roundings on some rows
    g = np.random.Generator(np.random.PCG64(7))
    differ_tocsr = differ_coo = 0
    for _ in range(50):
        col = g.integers(0, 8, 40)
        data = g.standard_normal(40) * 10.0 ** g.integers(-16, 17, 40)
        row = np.zeros(40, dtype=np.int64)
        want = reference((1, 8), row, col, data)[2]
        A = S.coo_array((sd.asarray(data), (sd.asarray(row), sd.asarray(col))), shape=(1, 8))
        assert bits_equal(npy(A.tocsr().data), want)
        s = sp.coo_array((data, (row, col)), shape=(1, 8))
        c = s.tocsr()
        c.sum_duplicates()
        differ_tocsr += not bits_equal(c.data, want)
        s.sum_duplicates()
        differ_coo += not bits_equal(s.data, want)
    assert differ_tocsr > 0 and differ_coo > 0


# ---- canonical output, every path, every thread count ----


@pytest.mark.usefixtures("restore_threads")
@pytest.mark.parametrize(
    "shape, nnz, kind",
    [
        ((300, 200), 20000, "float"),  # direct on every thread count
        ((10000, 10000), 100000, "float"),  # direct on one thread, buckets on several
        ((10000, 10000), 300000, "float"),  # buckets everywhere
        ((40, 10**6), 200000, "float"),  # few long lines
        ((3, 3), 70000, "float"),  # nearly all repeats
        ((5 * 10**6, 20), 70000, "float"),  # many empty lines, few entries per bucket
        ((5000, 5000), 100000, "complex"),
        ((5000, 5000), 100000, "int"),
        ((300, 300), 100000, "bool"),
    ],
)
def test_same_bits_for_every_thread_count(shape, nnz, kind):
    row, col, data = random_coo(shape, nnz, kind, 5)
    want_r = reference(shape, row, col, data, by_row=True)
    want_c = reference(shape, row, col, data, by_row=False)
    for t in THREADS:
        sd.set_num_threads(t)
        A = sd_coo(shape, row, col, data)
        csr, csc = A.tocsr(), A.tocsc()
        assert all(bits_equal(g, w) for g, w in zip(parts(csr), want_r)), t
        assert all(bits_equal(g, w) for g, w in zip(parts(csc), want_c)), t
        assert all(bits_equal(g, w) for g, w in zip(parts(csr.tocsc()), want_c)), t  # the transpose
        assert all(bits_equal(g, w) for g, w in zip(parts(csc.tocsr()), want_r)), t
        B = A.copy()
        B.sum_duplicates()
        assert bits_equal(npy(B.data), want_r[2]) and B.has_canonical_format


@pytest.mark.usefixtures("restore_threads")
@pytest.mark.parametrize("nnz", [2**16 - 1, 2**16, 2**17 + 5, 2**20 + 3])
def test_transpose_paths_agree(nnz):
    row, col, data = random_coo((nnz // 7 + 1, nnz // 9 + 1), nnz, "float", 9)
    csr = sd_coo((nnz // 7 + 1, nnz // 9 + 1), row, col, data).tocsr()
    got = []
    for t in (1, 10):
        sd.set_num_threads(t)
        got.append(parts(csr.tocsc()))
    assert all(bits_equal(a, b) for a, b in zip(*got))


@LIVE
@pytest.mark.parametrize("mid", M.ids(small=True))
def test_conversions_of_every_test_matrix(mid):
    m = M.get(mid)
    A = coo(m)
    ip, ix, v = m.csr()
    got = A.tocsr()
    assert bits_equal(npy(got.indptr), ip) and bits_equal(npy(got.indices), ix) and bits_equal(npy(got.data), v)
    ip, ix, v = m.csc()
    got = A.tocsc()
    assert bits_equal(npy(got.indptr), ip) and bits_equal(npy(got.indices), ix) and bits_equal(npy(got.data), v)
    s = scipy_coo(m).tocsc()
    s.sum_duplicates()
    assert bits_equal(npy(got.indptr), s.indptr.astype(np.int64)) and bits_equal(
        npy(got.indices), s.indices.astype(np.int64)
    )


def test_an_already_canonical_input_is_not_copied():
    A = S.coo_array(np.eye(4))
    assert A.has_canonical_format
    B = A.tocsr()
    assert B.data is A.data and B.indices is A.coords[1]
    C = A.tocsr(copy=True)
    assert not np.shares_memory(npy(C.data), npy(A.data))


def test_csr_from_compressed_parts_canonicalizes():
    # unsorted lines with repeats; empty lines; lines at the end
    A = S.csr_array(([1.0, 2.0, 3.0, 4.0, 5.0], [2, 0, 2, 1, 0], [0, 3, 3, 5, 5]), shape=(4, 3))
    assert bits_equal(npy(A.indptr), np.array([0, 2, 2, 4, 4]))
    assert bits_equal(npy(A.indices), np.array([0, 2, 0, 1]))
    assert bits_equal(npy(A.data), np.array([2.0, 4.0, 5.0, 4.0]))
