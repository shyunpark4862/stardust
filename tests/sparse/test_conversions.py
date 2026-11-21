"""Conversions between the formats, dense arrays and scipy.sparse, and the pattern methods of SP1
(docs/api/sparse.md §2, docs/design/sparse.md §2.5, §3.1, §3.5).

Every method is compared with scipy 1.18.1 on the test matrices: the result's format (scipy's:
reshape gives COO, a DOK's real/imag and complex conj give CSR, CSR's transpose is CSC), the
stored entries, the values bit for bit, the order of nonzero(). Where sd decided otherwise, the
test says so: a DOK's entries come out in row-major order (scipy: insertion order), COO repeats
are summed in input order, a sum_duplicates()/count_nonzero() does not change scipy's way of
summing, 1-D .T is a ValueError as for sd.ndarray (scipy: the array itself), resize keeps the
number of dimensions."""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import stardust as sd

from .._bits import bits_equal
from . import matrices as M
from .conftest import LIVE, coo, i64, npy, scipy_coo

S = sd.sparse
FORMATS = ("coo", "csr", "csc", "dok")
SMALL = M.ids(small=True)
# coo/cancel's repeats [1, 1e16, 1, -1e16] sum to 0 in input order; scipy's COO sum_duplicates and
# todok give 2 (np.add.reduceat), so the COO-summing comparisons leave it out
NOT_CANCEL = [i for i in SMALL if i != "coo/cancel"]


def both(mid, fmt):
    """Matrix mid in format fmt: (sd, scipy)."""
    m = M.get(mid)
    A, s = coo(m), scipy_coo(m)
    if fmt == "coo":
        return A, s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = s.asformat(fmt)
        if fmt in ("csr", "csc"):
            s.sum_duplicates()
    return A.asformat(fmt), s


def same_entries(A, s):
    """The same stored entries in the same order: COO coords and data, compressed arrays, dense."""
    assert A.format == s.format and A.shape == s.shape and A.nnz == s.nnz
    if A.format == "coo":
        assert all(bits_equal(npy(a), b.astype(np.int64)) for a, b in zip(A.coords, s.coords))
        assert bits_equal(npy(A.data), s.data)
    elif A.format in ("csr", "csc"):
        assert bits_equal(npy(A.indptr), s.indptr.astype(np.int64))
        assert bits_equal(npy(A.indices), s.indices.astype(np.int64))
        assert bits_equal(npy(A.data), s.data)
    assert bits_equal(npy(A.toarray()), s.toarray())


# ---- dense ----


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("mid", SMALL)
def test_toarray_matches_scipy(mid, fmt):
    A, s = both(mid, fmt)
    if fmt == "dok" and mid == "coo/cancel":
        return  # scipy's todok sums the repeats its own way (module docstring)
    assert bits_equal(npy(A.toarray()), s.toarray())


def test_toarray_adds_the_values_to_zeros():
    # a stored -0.0 is +0.0 in the dense array, as scipy's toarray (zeros += values)
    A = S.csr_array(([-0.0, complex(-0.0, -0.0).real], [0, 1], [0, 2]), shape=(1, 2))
    assert not np.signbit(npy(A.toarray())).any()
    Z = S.coo_array(([complex(-0.0, -0.0)], ([0], [0])), shape=(1, 1))
    z = npy(Z.toarray())[0, 0]
    assert not np.signbit(z.real) and not np.signbit(z.imag)
    assert np.signbit(npy(Z.tocsr().data)[0].real)  # the stored value itself keeps its sign


@pytest.mark.usefixtures("restore_threads")
def test_dense_conversions_for_every_thread_count():
    g = np.random.Generator(np.random.PCG64(3))
    x = g.standard_normal((600, 500))
    x[g.uniform(size=x.shape) < 0.9] = 0.0
    x[3, 4] = np.nan
    out = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        r = [S.csr_array(x), S.csc_array(x), S.coo_array(x.T)]
        out.append([npy(a.toarray()).tobytes() for a in r] + [npy(r[0].indices).tobytes(), npy(r[1].indices).tobytes()])
    assert out[0] == out[1] == out[2]
    assert bits_equal(npy(S.csr_array(x).toarray()), x + 0.0)


# ---- between formats ----


@LIVE
@pytest.mark.parametrize("src", FORMATS)
@pytest.mark.parametrize("dst", FORMATS)
@pytest.mark.parametrize("mid", NOT_CANCEL)
def test_every_conversion_matches_scipy(mid, src, dst):
    A, s = both(mid, src)
    B = A.asformat(dst)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = s.asformat(dst)
        if dst in ("csr", "csc"):
            t.sum_duplicates()
    if dst == "coo" and src in ("dok", "csc"):
        # sd: a DOK comes out row-major (scipy: insertion order); a CSC column by column, as scipy
        assert bits_equal(npy(B.toarray()), t.toarray())
        if src == "csc":
            same_entries(B, t)
        return
    if dst == "dok":
        assert B.nnz == t.nnz and bits_equal(npy(B.toarray()), t.toarray())
        return
    same_entries(B, t)


@pytest.mark.parametrize("fmt", FORMATS)
def test_same_format_and_copy(fmt):
    A = S.csr_array(np.eye(3)).asformat(fmt)
    assert getattr(A, "to" + fmt)() is A and A.asformat(fmt) is A and A.asformat(None) is A
    B = getattr(A, "to" + fmt)(copy=True)
    assert B is not A and bits_equal(npy(B.toarray()), npy(A.toarray()))
    if fmt != "dok":
        assert not np.shares_memory(npy(B.data), npy(A.data))


def test_conversion_copy_rules():
    A = S.csr_array(np.array([[1.0, 0], [2, 3]]))
    C = A.tocoo()
    assert C.data is A.data and C.coords[1] is A.indices  # CSR -> COO shares
    assert A.tocoo(copy=False).data is A.data
    assert not np.shares_memory(npy(A.tocoo(copy=True).data), npy(A.data))
    with pytest.raises(ValueError, match=r"^tocsc\(\): copy=False, but converting csr to csc requires a copy$"):
        A.tocsc(copy=False)
    with pytest.raises(ValueError, match=r"^todok\(\): copy=False, but a dok_array requires a copy$"):
        A.todok(copy=False)
    with pytest.raises(ValueError, match=r"^tocsr\(\): copy=False, but sorting or summing duplicates requires a copy$"):
        S.coo_array(([1.0, 2.0], ([1, 0], [0, 0]))).tocsr(copy=False)
    K = A.todok()
    with pytest.raises(ValueError, match=r"^tocoo\(\): copy=False, but converting dok to coo requires a copy$"):
        K.tocoo(copy=False)
    with pytest.raises(TypeError, match=r"^tocoo\(\): 'copy' must be True, False or None, not int$"):
        A.tocoo(copy=0)


def test_asformat_errors():
    A = S.csr_array(np.eye(2))
    with pytest.raises(
        ValueError, match=r"^asformat\(\): 'format' must be 'coo', 'csr', 'csc', 'dok' or None, not 'lil'$"
    ):
        A.asformat("lil")
    with pytest.raises(
        TypeError, match=r"^asformat\(\): 'format' must be 'coo', 'csr', 'csc', 'dok' or None, not int$"
    ):
        A.asformat(1)


@pytest.mark.parametrize("fmt", ["coo", "csr", "dok"])
def test_one_dimensional_conversions(fmt):
    x = np.array([0.0, 3, 0, -1, 0])
    A = S.coo_array(x).asformat(fmt)
    assert A.shape == (5,)
    for g in ("coo", "csr", "dok"):
        assert bits_equal(npy(A.asformat(g).toarray()), x)
    with pytest.raises(ValueError, match=r"^tocsc\(\): a csc_array has 2 dimensions, the array has 1$"):
        A.tocsc()


# ---- scipy ----


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
def test_to_scipy_is_a_view(fmt):
    A = S.csr_array(np.array([[1.0, 0, 2], [0, 3, 0]])).asformat(fmt)
    s = A.to_scipy()
    assert isinstance(s, getattr(sp, f"{fmt}_array")) and s.shape == A.shape
    assert bits_equal(s.toarray(), npy(A.toarray()))
    if fmt == "dok":
        with pytest.raises(ValueError, match=r"^to_scipy\(\): copy=False, but a scipy dok_array requires a copy$"):
            A.to_scipy(copy=False)
        return
    assert np.shares_memory(s.data, npy(A.data))
    s.data[0] = 7.0  # the values are shared and writable
    assert npy(A.data)[0] == 7.0
    idx = s.coords if fmt == "coo" else (s.indices, s.indptr)
    for a in idx:
        assert a.dtype == np.int64 and not a.flags.writeable
    c = A.to_scipy(copy=True)
    assert not np.shares_memory(c.data, npy(A.data)) and all(
        a.flags.writeable for a in (c.coords if fmt == "coo" else (c.indices, c.indptr))
    )


@LIVE
def test_to_scipy_one_dimensional_and_canonical_flags():
    v = S.csr_array(np.array([0.0, 1, 0, 2]))
    s = v.to_scipy()
    assert s.shape == (4,) and bits_equal(s.toarray(), np.array([0.0, 1, 0, 2])) and s.has_canonical_format
    c = S.coo_array(([1.0, 2.0], ([1, 0], [0, 0]))).to_scipy()
    assert not c.has_canonical_format
    c.sum_duplicates()  # scipy works on copies of the read-only views it replaces
    assert bits_equal(c.toarray(), np.array([[2.0], [1.0]]))


@LIVE
def test_scipy_input_is_always_copied():
    s = sp.csr_array(np.array([[1.0, 0, 2], [0, 3, 0]]))
    A = S.csr_array(s)
    s.data[:] = 0
    s.indices[:] = 0
    assert bits_equal(npy(A.toarray()), np.array([[1.0, 0, 2], [0, 3, 0]]))


@LIVE
def test_scipy_round_trip_every_matrix():
    for mid in SMALL:
        A = coo(M.get(mid)).tocsr()
        back = S.csr_array(A.to_scipy())
        assert bits_equal(npy(back.data), npy(A.data)) and bits_equal(npy(back.indices), npy(A.indices)), mid


# ---- transpose, reshape, resize ----


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
def test_transpose_matches_scipy(fmt):
    A, s = both("random_unsym/50x5", fmt)
    for T, t in (
        (A.T, s.T),
        (A.mT, s.T),
        (A.transpose(), s.transpose()),
        (S.matrix_transpose(A), sp.matrix_transpose(s)),
    ):
        assert T.format == t.format and T.shape == t.shape
        assert bits_equal(npy(T.toarray()), t.toarray())
    if fmt == "csr":
        assert A.T.data is A.data and A.T.indices is A.indices and A.T.indptr is A.indptr  # a view, as scipy


def test_one_dimensional_transpose():
    v = S.csr_array(np.array([0.0, 1.0]))
    assert v.transpose() is v
    with pytest.raises(ValueError, match=r"^\.T: the array must have 2 dimensions, got 1$"):
        _ = v.T
    with pytest.raises(ValueError, match=r"^\.mT: the array must have at least 2 dimensions, got 1$"):
        _ = v.mT
    with pytest.raises(ValueError, match=r"^matrix_transpose\(\): the array must have at least 2 dimensions, got 1$"):
        S.matrix_transpose(v)
    with pytest.raises(TypeError, match=r"^matrix_transpose\(\): 'A' must be a sparse array, not ndarray$"):
        S.matrix_transpose(sd.asarray([[1.0]]))


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("new", [(30, 40), (1200,), (-1, 20), (60, -1), (1, 1200)])
def test_reshape_matches_scipy(fmt, new):
    A, s = both("coo/dups", fmt)  # 40 x 30
    R, r = A.reshape(new), s.reshape(new)
    assert R.format == r.format == "coo" and R.shape == r.shape
    assert bits_equal(npy(R.toarray()), r.toarray())
    if fmt != "dok":
        same_entries(R, r)


def test_reshape_rules():
    A = S.csr_array(np.arange(6.0).reshape(2, 3))
    assert A.reshape((2, 3)) is A and A.reshape((2, 3), copy=True) is not A
    assert A.reshape(6).data is A.data  # the values are shared
    assert not np.shares_memory(npy(A.reshape(6, copy=True).data), npy(A.data))
    for shape, kind, text in [
        ((4, 2), ValueError, "reshape(): cannot reshape array of size 6 into shape (4, 2)"),
        ((-1, -1), ValueError, "reshape(): can only specify one unknown dimension"),
        ((1, 2, 3), ValueError, "reshape(): 'shape' must have 1 or 2 dimensions, got 3"),
        ((2.0, 3), TypeError, "reshape(): 'shape' must be an int or a tuple of ints, not a tuple holding float"),
        ("6", TypeError, "reshape(): 'shape' must be an int or a tuple of ints, not str"),
    ]:
        with pytest.raises(kind) as e:
            A.reshape(shape)
        assert str(e.value) == text
    with pytest.raises(ValueError, match="^reshape\\(\\): copy=False"):
        A.todok().reshape(6, copy=False)
    big = S.coo_array(([1.0], ([2**40 - 1], [2**40 - 1])), shape=(2**40, 2**40))
    flat = big.reshape((2**20, 2**60))  # positions past int64 in between (__int128)
    assert i64(flat.coords[0])[0] == 2**20 - 1 and i64(flat.coords[1])[0] == 2**60 - 1


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("new", [(20, 30), (40, 10), (60, 50), (10, 5), (0, 0)])
def test_resize_matches_scipy(fmt, new):
    A, s = both("coo/dups", fmt)
    A.resize(new)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.resize(new)
    assert A.shape == s.shape and A.nnz == s.nnz
    assert bits_equal(npy(A.toarray()), s.toarray())


def test_resize_keeps_the_dimensions():
    v = S.csr_array(np.array([1.0, 0, 2]))
    v.resize(2)
    assert v.shape == (2,) and bits_equal(npy(v.toarray()), np.array([1.0, 0]))
    with pytest.raises(ValueError, match=r"^resize\(\): the shape must have 1 dimensions, got 2$"):
        v.resize((1, 2))
    A = S.coo_array(([1.0, 2.0], ([1, 0], [0, 3])))
    A.resize((1, 4))
    assert A.nnz == 1 and A.has_canonical_format


# ---- nonzero, count_nonzero, eliminate_zeros ----


@LIVE
@pytest.mark.parametrize("fmt", ["coo", "csr", "csc"])
@pytest.mark.parametrize("mid", SMALL)
def test_nonzero_matches_scipy(mid, fmt):
    A, s = both(mid, fmt)
    got, want = A.nonzero(), s.nonzero()
    assert len(got) == len(want) and all(bits_equal(npy(g), w.astype(np.int64)) for g, w in zip(got, want))
    for g in got:
        assert npy(g).flags.writeable


def test_nonzero_of_dok_is_row_major():
    K = S.dok_array((3, 3))
    K[2, 0] = 1.0
    K[0, 2] = 2.0
    K[1, 1] = 3.0
    r, c = K.nonzero()
    assert bits_equal(npy(r), np.array([0, 1, 2])) and bits_equal(npy(c), np.array([2, 1, 0]))


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("mid", NOT_CANCEL)
def test_count_nonzero_matches_scipy(mid, fmt):
    A, s = both(mid, fmt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert A.count_nonzero() == int(s.copy().count_nonzero())
        for axis in (0, 1, -1, -2):
            assert bits_equal(npy(A.count_nonzero(axis=axis)), s.copy().count_nonzero(axis=axis).astype(np.int64))


def test_count_nonzero_rules():
    A = S.coo_array(([1.0, -1.0, 0.0, 2.0], ([0, 0, 1, 1], [0, 0, 1, 2])))
    assert A.count_nonzero() == 1 and A.nnz == 4  # the repeats summed first; A unchanged
    assert isinstance(A.count_nonzero(), int)
    assert S.csr_array(np.array([0.0, 1, 2])).count_nonzero(axis=0) == 2
    with pytest.raises(ValueError, match=r"^count_nonzero\(\): axis 2 is out of bounds for an array of dimension 2$"):
        A.count_nonzero(axis=2)
    with pytest.raises(TypeError, match=r"^count_nonzero\(\): 'axis' must be an int or None, not float$"):
        A.count_nonzero(axis=1.0)


@LIVE
@pytest.mark.parametrize("fmt", ["coo", "csr", "csc"])
@pytest.mark.parametrize("mid", SMALL)
def test_eliminate_zeros_matches_scipy(mid, fmt):
    A, s = both(mid, fmt)
    A.eliminate_zeros()
    s.eliminate_zeros()
    same_entries(A, s)


def test_eliminate_zeros_drops_minus_zero_and_keeps_nan():
    A = S.csr_array(([-0.0, np.nan, 0.0, 2.0], [0, 1, 2, 3], [0, 4]), shape=(1, 4))
    A.eliminate_zeros()
    assert bits_equal(npy(A.indices), np.array([1, 3])) and bits_equal(npy(A.indptr), np.array([0, 2]))
    assert not hasattr(S.dok_array((2, 2)), "eliminate_zeros")  # scipy's dok_array has none


# ---- values: astype, conj, real, imag ----


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("mid", ["random_complex/40", "random_unsym/50x5", "coo/int", "coo/bool"])
def test_value_methods_match_scipy(mid, fmt):
    A, s = both(mid, fmt)
    pairs = [(A.conj(), s.conj()), (A.real, s.real), (A.imag, s.imag), (A.copy(), s.copy())]
    if A.dtype is not sd.complex128:
        pairs.append((A.astype(sd.float64), s.astype(np.float64)))
        pairs.append((A.astype(sd.complex128), s.astype(np.complex128)))
    for got, want in pairs:
        assert got.format == want.format and got.shape == want.shape and got.nnz == want.nnz
        assert str(got.dtype) == want.dtype.name
        assert bits_equal(npy(got.toarray()), want.toarray())


def test_astype_rules():
    A = S.csr_array(np.array([[1.5, 0], [np.nan, -2.5]]))
    B = A.astype(sd.int64)
    assert B.dtype is sd.int64 and bits_equal(npy(B.data), np.array([1, 0, -2])) and B.nnz == 3
    assert A.astype("float64", copy=False) is A and A.astype(sd.float64, copy=None) is A
    assert A.astype(sd.float64) is not A
    with pytest.raises(ValueError, match=r"^astype\(\): copy=False"):
        A.astype(sd.int64, copy=False)
    with pytest.raises(TypeError, match=r"^astype\(\): cannot convert complex128 to float64"):
        A.astype(sd.complex128).astype(sd.float64)
    with pytest.raises(TypeError, match=r"^astype\(\): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128"):
        A.astype("float32")
    K = A.todok().astype(sd.int64)
    assert K.format == "dok" and K[1, 0] == 0 and K.nnz == 3  # a converted 0 stays stored


def test_real_of_a_real_array_shares_the_values():
    A = S.csr_array(np.array([[1.0, 0], [0, 2]]))
    assert A.real.data is A.data
    assert A.conj().data is not A.data and A.conj().format == "csr"
    C = S.csr_array(np.array([[1 + 2j, 0], [0, 3j]]))
    assert C.real.dtype is sd.float64 and bits_equal(npy(C.imag.data), np.array([2.0, 3.0]))
