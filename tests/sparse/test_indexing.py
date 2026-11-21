"""Indexing, assignment and the diagonal of SP3 (docs/api/sparse.md §4.1, docs/design/sparse.md
§3.3).

The values of A[key] are those of the dense array: A[key].toarray() is A.toarray()[key] bit for
bit, and a key the dense array refuses is refused with its error and message (the array's
indexing rules, docs/api/array.md §5). Against scipy 1.18.1 (live): the stored entries (nnz,
repeats of a COO array kept) and the result's format -- A's format when the result has A's number
of dimensions, else COO; scipy's CSR and CSC give a dense numpy array for element picking and
2-D masks, sd a COO array. Assignment: the dense result is the dense array's assignment (a
repeated position takes the last value), and what is stored follows scipy per format (CSR and CSC
store assigned zeros, DOK erases them, COO drops them; COO's repeated position keeps scipy's
first value, sd the last)."""

import numpy as np
import pytest
import scipy.sparse as sp

import stardust as sd

from .._bits import bits_equal, value_equal
from . import matrices as M
from .conftest import LIVE, coo, npy

pytestmark = pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")

S = sd.sparse
FORMATS = ("coo", "csr", "csc", "dok")
MATS = ["coo/dups", "coo/int", "coo/bool", "coo/complex", "coo/nan_inf", "random_unsym/50x5", "rect/20x60",
        "empty/rows", "empty/0x5", "long_rows/300"]  # fmt: skip


def keys_for(shape, seed):
    """Keys of every kind for an array of `shape`: ints, slices, arrays, masks, None, ..."""
    r = np.random.default_rng(seed)
    out = []
    if len(shape) == 1:
        (n,) = shape
        if n:
            out += [0, n - 1, -1, slice(None), slice(1, None, 2), slice(None, None, -1), slice(-3, None),
                    [0, 0, n - 1], r.integers(-n, n, 7), r.random(n) < 0.3, (None, slice(None)), (slice(None), None),
                    (Ellipsis,), [], np.array([[0, n - 1], [1, 0]])]  # fmt: skip
        return out
    m, n = shape
    out += [slice(None), (slice(None), slice(None)), Ellipsis, [], (slice(None), [])]
    if m:
        out += [0, -1, (m - 1,), (slice(None), slice(2, None, 3)), (slice(None, None, -2),), [m - 1, 0, 0],
                r.integers(-m, m, 6), r.random(m) < 0.4, (None, 0), (0, None), (slice(1, None), Ellipsis)]  # fmt: skip
    if n:
        out += [(slice(None), 0), (Ellipsis, -1), (slice(None), r.integers(-n, n, 5)), (slice(None), r.random(n) < 0.5),
                (slice(None), [n - 1, 0, n - 1]), (slice(None, None, -1), slice(None, None, -3))]  # fmt: skip
    if m and n:
        out += [(0, 0), (m - 1, -1), (-m, n - 1), (np.int64(m // 2), np.int64(n // 2)),
                (1 % m, slice(None)), (slice(None), n // 2), (r.integers(0, m, 4), r.integers(0, n, 4)),
                ([0, 0, m - 1], [0, 0, n - 1]), (np.array([[0], [m - 1]]), [0, n - 1, 0]),
                (np.array([[0, m - 1]]), np.array([[n - 1], [0]])), (r.integers(0, m, 5), 0), (0, r.integers(0, n, 5)),
                (slice(m // 3, None), r.integers(0, n, 4)), (r.integers(0, m, 4), slice(None, n // 2)),
                (r.integers(0, m, (2, 3)), r.integers(0, n, (2, 3))), r.random((m, n)) < 0.3,
                (r.random(m) < 0.5, slice(None, None, 2))]  # fmt: skip
    return out


def sparse_of(m, fmt):
    """A new array of matrix m (its own values: an assignment writes them in place)."""
    return coo(m).copy().asformat(fmt)


def scipy_of(m, fmt):
    return sp.coo_array((m.data, (m.row, m.col)), shape=m.shape).asformat(fmt)


def dense_index(A, key):
    D = A.toarray()
    try:
        return D[key], None
    except (IndexError, ValueError, TypeError) as e:
        return None, e


def check_result(A, key, got):
    want, err = dense_index(A, key)
    assert err is None, (key, err)
    if isinstance(got, S.sparray):
        assert tuple(got.shape) == tuple(want.shape), key
        assert bits_equal(npy(got.toarray()), npy(want)), key
        same = got.ndim == A.ndim
        assert got.format == (A.format if same else "coo"), (key, got.format)
    else:
        assert not isinstance(want, sd.ndarray), key
        assert type(got) is type(want) or bits_equal(got, want), key
        assert bits_equal(got, want) or (got == want == 0), key


@pytest.mark.parametrize("mid", MATS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_getitem_is_the_dense_indexing(mid, fmt):
    m = M.get(mid)
    A = sparse_of(m, fmt)
    for key in keys_for(A.shape, 3):
        got = A[key]
        check_result(A, key, got)


@pytest.mark.parametrize("fmt", ("coo", "csr", "dok"))
def test_getitem_of_1d_is_the_dense_indexing(fmt):
    x = sd.asarray([0.0, 2, 0, -1, np.nan, 0, -0.0, 3])
    A = S.coo_array(x).asformat(fmt)
    for key in keys_for(A.shape, 5):
        got = A[key]
        check_result(A, key, got)


def is_pick(key):
    """Whether a key picks elements: arrays on both axes, or a 2-D mask."""
    if isinstance(key, np.ndarray) and key.ndim == 2:
        return True
    return isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, (list, np.ndarray)) for k in key)


@LIVE
@pytest.mark.parametrize("mid", MATS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_getitem_stores_what_scipy_stores(mid, fmt):
    m = M.get(mid)
    A, B = sparse_of(m, fmt), scipy_of(m, fmt)
    for key in keys_for(A.shape, 7):
        got = A[key]
        try:
            want = B[key]
        except ValueError:  # scipy 1.18.1 fails on a row mask with a strided slice (a reshape)
            continue
        if not isinstance(got, S.sparray):
            assert got == want.item() or (got != got and want != want), key
            continue
        if not sp.issparse(want):  # scipy's CSR and CSC pick elements into a dense array
            assert fmt in ("csr", "csc") and got.format == "coo", key
            assert value_equal(npy(got.toarray()), np.asarray(want)), key  # scipy keeps a stored -0.0
            continue
        assert got.shape == want.shape, key
        assert bits_equal(npy(got.toarray()), want.toarray()), key
        # scipy's CSR and CSC build a 2-D element pick from dense values, and its DOK drops stored
        # zeros on most paths: sd keeps the stored entries (a stored zero stays) in every format
        if not (is_pick(key) and fmt in ("csr", "csc")) and fmt != "dok":
            assert got.nnz == want.nnz, (key, got.nnz, want.nnz)
        if got.format == "coo" and want.format == "coo" and fmt == "coo":  # repeats and order as scipy's
            assert [npy(c).tolist() for c in got.coords] == [c.tolist() for c in want.coords], key
            assert bits_equal(npy(got.data), want.data), key


@pytest.mark.parametrize("fmt", FORMATS)
def test_errors_are_the_arrays(fmt):
    A = S.csr_array(sd.asarray(np.arange(12.0).reshape(3, 4))).asformat(fmt)
    for key in [5, (0, 7), -4, [0, 5], (slice(None), [0, 9]), [1.0], True, (0, 0, 0), (Ellipsis, Ellipsis), "a", 1.5,
                ([0, 1], [0, 1, 2]), np.array([True, False]), slice(None, None, 0), ([0, 1], None, [0, 1]),
                (0, None, [0, 1])]:  # fmt: skip
        _, err = dense_index(A, key)
        assert err is not None, key
        with pytest.raises(type(err)) as e:
            A[key]
        assert str(e.value) == str(err), (key, str(e.value), str(err))


def test_results_of_more_than_two_dimensions_are_an_index_error():
    A = S.csr_array(sd.asarray(np.eye(3)))
    for key in [(None, slice(None)), (slice(None), None), np.array([[0, 1], [1, 2]]), ([0, 1], None)]:
        with pytest.raises(IndexError, match="a sparse array has 1 or 2"):
            A[key]
    with pytest.raises(IndexError, match="a sparse array is not a valid index"):
        A[A > 0]


def test_scalar_is_a_python_scalar_and_coo_repeats_add_in_stored_order():
    A = S.coo_array(
        (sd.asarray([1.0, 1e16, 1.0, -1e16]), (sd.asarray([0, 0, 0, 0]), sd.asarray([1, 1, 1, 1]))), shape=(2, 2)
    )
    assert A[0, 1] == 0.0 and type(A[0, 1]) is float
    assert A.tocsr()[0, 1] == 0.0 and A.todok()[0, 1] == 0.0
    assert A[1, 1] == 0.0 and type(A[1, 1]) is float
    B = S.csr_array(sd.asarray([[1, 0], [0, 2]]))
    assert type(B[1, 1]) is int and B[1, 1] == 2 and type(B[0, 1]) is int
    assert type(S.csr_array(sd.asarray([True]))[0]) is bool


def test_stored_negative_zero_is_kept_by_a_scalar_read():
    A = S.csr_array((sd.asarray([-0.0]), sd.asarray([0]), sd.asarray([0, 1, 1])), shape=(2, 2))
    assert np.signbit(A[0, 0]) and not np.signbit(A[1, 1])


def test_results_are_canonical_and_bit_equal_on_every_thread_count(restore_threads):
    r = np.random.default_rng(11)
    n = 3 * 10**5
    ij = (sd.asarray(r.integers(0, 3000, n)), sd.asarray(r.integers(0, 2500, n)))
    D = S.coo_array((sd.asarray(r.standard_normal(n)), ij), shape=(3000, 2500))
    rows = r.integers(0, 3000, 4000)
    cols = r.integers(0, 2500, 3000)
    keys = [(rows, slice(None)), (slice(None), cols), (rows[:, None], cols[None, :500]), (rows[:3000], cols),
            (slice(None, None, -1), slice(10, 2000, 3)), (slice(None), sd.asarray(np.sort(cols)))]  # fmt: skip
    for fmt in FORMATS:
        A = D.asformat(fmt)
        base = None
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            res = [A[k] for k in keys]
            enc = [stored_bytes(x) for x in res]
            if base is None:
                base = enc
            assert enc == base, (fmt, t)


def stored_bytes(R):
    c = R.tocoo()
    return tuple(npy(x).tobytes() for x in c.coords) + (npy(c.data).tobytes(), R.format, R.shape)


# ---- assignment ----


def assign_cases(shape, seed):
    r = np.random.default_rng(seed)
    m, n = shape
    return [
        ((0, 1), 0.0), ((0, 0), 0.0), ((m - 1, n - 1), 2.5), ((0, slice(None)), 0.0), ((0, slice(None)), 7.0),
        ((slice(None), 1), sd.asarray(r.standard_normal(m))), (([0, 0], [1, 1]), sd.asarray([5.0, 6.0])),
        ((slice(0, 2), slice(0, 2)), sd.asarray([[0.0, 9], [0, 0]])),
        (([0, m - 1], slice(None)), sd.asarray(r.standard_normal(n))),
        ((slice(None, None, -1), 2), sd.asarray(r.standard_normal(m))),
        ((r.integers(0, m, 6), r.integers(0, n, 6)), 1.0),
        ((slice(None), [n - 1, 0, 0]), sd.asarray(np.array([[1.0, 2, 3]]))), (r.random((m, n)) < 0.2, 0.0),
        ((slice(1, None), r.integers(0, n, 3)), sd.asarray(r.standard_normal((m - 1, 3)))),
    ]  # fmt: skip


def dense_assign(A, key, value):
    D = A.toarray()
    D[key] = value
    return D


@pytest.mark.parametrize("fmt", FORMATS)
def test_setitem_is_the_dense_assignment(fmt):
    m = M.get("random_unsym/50x5")
    for key, value in assign_cases(m.shape, 2):
        A = sparse_of(m, fmt)
        want = dense_assign(A, key, value)
        A[key] = value
        assert A.format == fmt
        assert bits_equal(npy(A.toarray()), npy(want)), key
        if fmt in ("csr", "csc"):  # still canonical: the arrays agree with a rebuilt array
            B = type(A)((A.data, A.indices, A.indptr), shape=A.shape)
            assert bits_equal(npy(B.indices), npy(A.indices))


def scipy_entries(B):
    c = B.tocoo()
    return sorted(zip(c.coords[0].tolist(), c.coords[1].tolist(), c.data.tolist()))


def sd_entries(A):
    c = A.tocoo()
    return sorted(zip(npy(c.coords[0]).tolist(), npy(c.coords[1]).tolist(), npy(c.data).tolist()))


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
def test_setitem_stores_what_scipy_stores(fmt):
    m = M.get("rect/20x60")
    for key, value in assign_cases(m.shape, 4):
        if fmt == "coo" and isinstance(key, tuple) and any(isinstance(k, list) and len(set(k)) < len(k) for k in key):
            continue  # a COO repeat: scipy keeps the first value, sd the last
        if fmt == "coo" and isinstance(key, tuple) and all(isinstance(k, np.ndarray) for k in key):
            pairs = list(zip(*key))
            if len(set(pairs)) < len(pairs):
                continue
        A, B = sparse_of(m, fmt), scipy_of(m, fmt)
        A[key] = value
        B[key] = np.asarray(value) if isinstance(value, sd.ndarray) else value
        assert sd_entries(A) == scipy_entries(B), (key, fmt)


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
def test_setitem_with_a_sparse_value_stores_what_scipy_stores(fmt):
    m = M.get("random_unsym/50x5")
    x = np.array([[0.0, 9, 0], [0, 0, 4]])
    for key in [(slice(0, 2), slice(3, 6)), (slice(10, 12), [4, 1, 7])]:
        for X in (S.csr_array(sd.asarray(x)), S.coo_array(sd.asarray(x[:1]))):
            A, B = sparse_of(m, fmt), scipy_of(m, fmt)
            A[key] = X
            B[key] = X.to_scipy(copy=True)
            assert sd_entries(A) == scipy_entries(B), (key, fmt, X.shape)


def test_setitem_dtype_and_value_rules_are_the_arrays():
    A = S.csr_array(sd.asarray([[1, 0], [0, 2]]))
    with pytest.raises(TypeError, match="cannot implicitly convert float64 to int64"):
        A[0, 1] = 2.5
    with pytest.raises(TypeError, match="only a bool, int, float or complex can be assigned to a single element"):
        A[0, 1] = [3]
    with pytest.raises(ValueError, match=r"cannot broadcast a value of shape \(1, 2\) to the indexed shape \(2,\)"):
        A[0, :] = sd.asarray([[1, 2]])
    with pytest.raises(TypeError, match="cannot implicitly convert float64 to int64"):
        A[0:1, :] = S.csr_array(sd.asarray([[1.5, 0]]))
    with pytest.raises(IndexError, match="index 2 is out of bounds for axis 0 with size 2"):
        A[[0, 2], 0] = 1
    A[0, 1] = True  # a bool goes into int64 (the array's rule)
    assert A[0, 1] == 1
    for fmt in FORMATS:
        B = A.asformat(fmt)
        with pytest.raises(TypeError, match="cannot implicitly convert complex128 to int64"):
            B[1, 0] = 1j


def test_csr_assignment_changes_a_stored_value_in_place_and_rebuilds_for_a_new_one():
    A = S.csr_array(sd.asarray([[1.0, 0], [0, 2]]))
    data, indptr = A.data, A.indptr
    A[1, 1] = 5.0
    assert A.data is data and A.indptr is indptr and npy(data)[1] == 5.0
    A[0, 1] = 0.0  # a new position, stored though 0 (scipy's)
    assert A.nnz == 3 and A.indptr is not indptr and not A.indices._flags()["writeable"]
    # values given with copy=None stay untouched when read-only
    v = sd.asarray([1.0, 2.0])
    v._set_readonly()
    B = S.csr_array((v, sd.asarray([0, 1]), sd.asarray([0, 1, 2])), shape=(2, 2))
    B[0, 0] = 7.0
    assert npy(v).tolist() == [1.0, 2.0] and B[0, 0] == 7.0


def test_dok_assignment_erases_zeros_and_coo_drops_them():
    D = np.array([[1.0, 0, 2], [0, 3, 0]])
    A = S.dok_array(sd.asarray(D))
    A[0, :] = sd.asarray([0.0, 4, 0])
    assert A.nnz == 2 and A[0, 1] == 4.0
    C = S.coo_array(sd.asarray(D))
    C[:, 0] = 0.0
    assert C.nnz == 2
    C[1, 2] = 0.0  # a single element is stored even when 0 (scipy's)
    assert C.nnz == 3


# ---- the diagonal ----


@pytest.mark.parametrize("mid", ["coo/dups", "coo/int", "coo/bool", "coo/complex", "coo/nan_inf", "rect/20x60",
                                 "rect/60x20", "empty/0x5"])  # fmt: skip
@pytest.mark.parametrize("fmt", FORMATS)
def test_diagonal_and_trace_are_the_dense_ones(mid, fmt):
    m = M.get(mid)
    A = sparse_of(m, fmt)
    D = A.toarray()
    for k in (0, 1, -1, 3, -5, 100, -100):
        got = A.diagonal(k=k)
        assert isinstance(got, sd.ndarray) and bits_equal(npy(got), npy(sd.linalg.diagonal(D, k=k))), k
        t = A.trace(k=k)
        assert bits_equal(t, sd.sum(sd.linalg.diagonal(D, k=k))), k
        if m.data.dtype.kind == "f":
            assert bits_equal(t, sd.linalg.trace(D, k=k)), k


@LIVE
@pytest.mark.parametrize("fmt", FORMATS)
def test_setdiag_stores_what_scipy_stores(fmt):
    m = M.get("rect/20x60")
    for values, k in [(3.0, 0), (0.0, 2), (sd.asarray([1.0, 0, 2]), -1), (sd.asarray(np.arange(100.0)), 5), (7.0, -19)]:
        A, B = sparse_of(m, fmt), scipy_of(m, fmt)
        A.setdiag(values, k=k)
        B.setdiag(np.asarray(values) if isinstance(values, sd.ndarray) else values, k)
        assert sd_entries(A) == scipy_entries(B), (fmt, k)


def test_diagonal_errors():
    A = S.csr_array(sd.asarray(np.eye(3)))
    with pytest.raises(TypeError, match=r"diagonal\(\): 'k' must be an int, not float"):
        A.diagonal(k=1.0)
    with pytest.raises(ValueError, match=r"diagonal\(\): the array must have 2 dimensions, got 1"):
        S.csr_array(sd.asarray([1.0, 2])).diagonal()
    with pytest.raises(ValueError, match=r"setdiag\(\): k 3 exceeds"):
        A.setdiag(1.0, k=3)
    with pytest.raises(ValueError, match=r"setdiag\(\): 'values' must be a number or 1-D"):
        A.setdiag(sd.eye(2))
    with pytest.raises(TypeError, match="cannot implicitly convert float64 to int64"):
        A.astype(sd.int64).setdiag(1.5)
