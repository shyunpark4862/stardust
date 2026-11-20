"""The test matrices (tests/sparse/matrices.py) are what their tags say, and their canonical forms
follow docs/design/sparse.md §2.3: row-major (column-major for CSC) order, no duplicates,
duplicates summed left to right in input order."""

import numpy as np
import pytest

from .._bits import bits_equal
from . import matrices as M

SMALL = M.ids(small=True)


def test_ids_are_unique_and_heavy_ones_are_left_out():
    assert len(set(M.IDS)) == len(M.IDS)
    assert M.HEAVY <= set(M.IDS)
    assert not set(M.ids()) & M.HEAVY
    with pytest.raises(KeyError):
        M.get("lap2d/7")


@pytest.mark.parametrize("mid", M.ids())
def test_canonical_forms(mid):
    a = M.get(mid)
    m, n = a.shape
    assert a.row.dtype == a.col.dtype == np.int64
    for kind, (indptr, indices, data) in (("csr", a.csr()), ("csc", a.csc())):
        lines = m if kind == "csr" else n
        assert indptr.dtype == indices.dtype == np.int64 and indptr.shape == (lines + 1,)
        assert indptr[0] == 0 and (np.diff(indptr) >= 0).all() and indptr[-1] == data.size
        for i in range(0, lines, max(1, lines // 50)):
            seg = indices[indptr[i] : indptr[i + 1]]
            assert (np.diff(seg) > 0).all()  # sorted, no duplicates
    r, c, v = a.canonical()
    if r.size:
        key = r * n + c
        assert (np.diff(key) > 0).all()
    if "coo" not in a.tags:  # generated matrices are already canonical
        assert bits_equal(r, a.row) and bits_equal(c, a.col) and bits_equal(v, a.data)


def test_duplicates_sum_left_to_right():
    a = M.get("coo/cancel")
    d = a.dense()
    assert d[0, 1] == 0.0  # ((1 + 1e16) + 1) - 1e16, not 2
    assert d[1, 0] == 3.0 and d[2, 2] == 0.0
    r, c, v = a.canonical()
    assert (2, 2) in set(zip(r.tolist(), c.tolist()))  # the explicit zero stays stored


@pytest.mark.parametrize("mid", [i for i in SMALL if M.get(i).shape[0] == M.get(i).shape[1] and M.get(i).nnz])
def test_tags_describe_the_matrix(mid):
    a = M.get(mid)
    d = a.dense()
    t = a.tags
    if t & {"symmetric", "spd", "indefinite", "hermitian"}:
        assert np.array_equal(d, d.conj().T)
    if "spd" in t:
        assert np.linalg.eigvalsh(d).min() > 0
    if "indefinite" in t:
        w = np.linalg.eigvalsh(d)
        assert w.min() < 0 < w.max() and np.abs(w).min() > 1e-8 * np.abs(w).max()
    if "singular" in t:
        assert np.linalg.matrix_rank(d) < d.shape[0]
    if "struct_singular" in t:
        assert (np.abs(d).sum(axis=0) == 0).any() or (np.abs(d).sum(axis=1) == 0).any()
    if "ill" in t:
        assert np.linalg.cond(d) > 1e7
    if "complex" in t:
        assert a.dtype == np.complex128
    if t & {"unsym", "random", "banded", "grid", "spd", "indefinite"} and not t & {
        "singular",
        "struct_singular",
        "ill",
    }:
        assert np.linalg.cond(d) < 1e7


def test_rectangular_and_rank_deficient():
    for mid in ("rect/60x20", "rect/1200x300"):
        d = M.get(mid).dense()
        assert np.linalg.matrix_rank(d) == min(d.shape)
    d = M.get("rect/80x30d").dense()
    assert np.linalg.matrix_rank(d) == 29


def test_long_rows_pass_the_summation_chunk():
    a = M.get("long_rows/9000")
    lengths = np.diff(a.csr()[0])
    assert (lengths > 4096).sum() == 3


def test_empty_shapes():
    for mid, shape in (("empty/0x5", (0, 5)), ("empty/5x0", (5, 0)), ("empty/0x0", (0, 0)), ("empty/4x6", (4, 6))):
        a = M.get(mid)
        assert a.shape == shape and a.nnz == 0 and a.csr()[0].shape == (shape[0] + 1,)
    d = M.get("empty/rows").dense()
    assert (np.abs(d).sum(axis=1) == 0).any() and (np.abs(d).sum(axis=0) == 0).any()


def test_dtypes_of_the_coo_inputs():
    assert M.get("coo/int").dtype == np.int64 and M.get("coo/bool").dtype == np.bool_
    assert M.get("coo/complex").dtype == np.complex128
    d = M.get("coo/nan_inf").data
    assert np.isnan(d).sum() == 1 and np.isinf(d).sum() == 2 and np.signbit(d[-1])


def test_digest_and_rhs_are_fixed():
    a = M.get("lap2d/4")
    assert M.digest(a) == M.digest(M._build("lap2d/4"))
    assert bits_equal(M.rhs(a), M.rhs(a)) and M.rhs(a, 3).shape == (16, 3)
    assert M.rhs(M.get("random_complex/40")).dtype == np.complex128
