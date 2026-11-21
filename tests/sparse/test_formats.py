"""The four formats and their constructors (docs/api/sparse.md §2, docs/design/sparse.md §2):
every accepted input, the dtype and copy rules of the array area, the checks of the index
arrays, the read-only index arrays, the attributes and the Python protocol.

The constructors are compared with scipy 1.18.1 where scipy defines the result (format, shape,
stored entries, values); the differences decided for sd (float32 widens, an index array is always
copied, sd's dtype and copy rules) are tested as such."""

import numpy as np
import pytest
import scipy.sparse as sp

import stardust as sd

from .._bits import bits_equal
from .conftest import LIVE, i64, npy

S = sd.sparse
CLASSES = (S.coo_array, S.csr_array, S.csc_array, S.dok_array)
SCIPY = {"coo": sp.coo_array, "csr": sp.csr_array, "csc": sp.csc_array, "dok": sp.dok_array}
D = np.array([[1.0, 0, 2], [0, 0, 3], [np.nan, -0.0, 0]])


def dense_of(A):
    return np.asarray(A.toarray())


# ---- accepted inputs ----


@LIVE
@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "make",
    [
        lambda: D,
        lambda: sd.asarray(D),
        lambda: D.tolist(),
        lambda: [[1, 0], [0, 2]],
        lambda: [[True, False], [False, True]],
        lambda: np.array([[1 + 2j, 0], [0, 0]]),
        lambda: np.zeros((0, 3)),
        lambda: np.zeros((3, 0)),
        lambda: np.asfortranarray(D),
        lambda: D[:, ::2],
    ],
    ids=["numpy", "sd", "list", "ints", "bools", "complex", "0x3", "3x0", "fortran", "strided"],
)
def test_dense_input_matches_scipy(cls, make):
    A = cls(make())
    ref = SCIPY[A.format](make() if not isinstance(make(), sd.ndarray) else D)
    assert A.format == ref.format and A.shape == ref.shape and A.nnz == ref.nnz
    assert str(A.dtype) == ref.dtype.name
    assert bits_equal(dense_of(A), ref.toarray())


@LIVE
@pytest.mark.parametrize("cls", [S.coo_array, S.csr_array, S.dok_array], ids=lambda c: c.__name__)
def test_one_dimensional_dense(cls):
    x = np.array([0.0, 1, 0, np.nan, -0.0, 2])
    A = cls(x)
    ref = SCIPY[A.format](x)
    assert A.shape == (6,) and A.ndim == 1 and A.nnz == ref.nnz == 3
    assert bits_equal(dense_of(A), ref.toarray())


def test_dense_nan_is_stored_and_minus_zero_is_not():
    A = S.csr_array(D)
    assert A.nnz == 4  # 1, 2, 3, nan
    assert bits_equal(npy(A.indices), np.array([0, 2, 2, 0]))


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_shape_tuple_is_an_empty_array(cls):
    A = cls((3, 4))
    assert A.shape == (3, 4) and A.nnz == 0 and A.dtype is sd.float64
    assert cls((3, 4), dtype="int64").dtype is sd.int64
    assert bits_equal(dense_of(A), np.zeros((3, 4)))
    if cls is not S.csc_array:
        assert cls((5,)).shape == (5,)


@LIVE
@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_coordinates_input(cls):
    data, row, col = np.array([1.0, 2, 3, 4]), np.array([1, 0, 1, 0]), np.array([2, 0, 2, 1])
    A = cls((data, (row, col)))
    ref = sp.coo_array((data, (row, col)))
    assert A.shape == ref.shape == (2, 3)
    assert bits_equal(dense_of(A), ref.toarray())
    if cls is S.coo_array:
        assert A.nnz == 4 and not A.has_canonical_format  # stored as given, like scipy
    elif cls in (S.csr_array, S.csc_array):
        assert A.nnz == 3  # canonical: the repeated (1, 2) summed


def test_one_dimensional_coordinates():
    A = S.coo_array(([1.0, 2.0], ([3, 0],)))
    assert A.shape == (4,) and bits_equal(dense_of(A), np.array([2.0, 0, 0, 1]))
    B = S.csr_array(([1.0, 2.0, 5.0], ([3, 0, 3],)), shape=(6,))
    assert (
        B.shape == (6,)
        and bits_equal(npy(B.indptr), np.array([0, 2]))
        and bits_equal(npy(B.data), np.array([2.0, 6.0]))
    )


@LIVE
@pytest.mark.parametrize("cls", [S.csr_array, S.csc_array], ids=lambda c: c.__name__)
def test_compressed_parts_input(cls):
    data, indices, indptr = np.array([1.0, 2, 3]), np.array([2, 0, 1]), np.array([0, 2, 3, 3])
    A = cls((data, indices, indptr))
    ref = SCIPY[A.format]((data, indices, indptr))
    assert A.shape == ref.shape and bits_equal(dense_of(A), ref.toarray())
    # canonical in the result even though the input was not sorted
    assert bits_equal(npy(A.indices), np.array([0, 2, 1])) and bits_equal(npy(A.data), np.array([2.0, 1, 3]))


def test_compressed_parts_one_dimensional():
    A = S.csr_array(([5.0, 7.0], [1, 3], [0, 2]), shape=(4,))
    assert A.shape == (4,) and bits_equal(dense_of(A), np.array([0.0, 5, 0, 7]))


@LIVE
@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("fmt", ["coo", "csr", "csc", "dok", "lil", "bsr", "dia"])
@pytest.mark.parametrize("kind", ["array", "matrix"])
def test_scipy_input_every_format(cls, fmt, kind):
    base = sp.random_array((7, 5), density=0.4, rng=np.random.default_rng(3), format="coo")
    src = base.asformat(fmt)
    if kind == "matrix":
        src = sp.csr_matrix(src).asformat(fmt)
    A = cls(src)
    assert A.shape == (7, 5)
    assert bits_equal(dense_of(A), base.toarray())


@LIVE
def test_scipy_one_dimensional_and_int32_indices():
    v = sp.coo_array(np.array([0.0, 3, 0, 4]))
    A = S.csr_array(v)
    assert A.shape == (4,) and bits_equal(dense_of(A), v.toarray())
    c = sp.csr_array(np.eye(3))
    assert c.indices.dtype == np.int32
    B = S.csr_array(c)
    assert B.indices.dtype is sd.int64 and bits_equal(dense_of(B), np.eye(3))


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_sparse_input_converts(cls):
    for src in (S.coo_array(D), S.csr_array(D), S.csc_array(D), S.dok_array(D)):
        A = cls(src)
        assert A.format == cls.format and A is not src
        assert bits_equal(dense_of(A), dense_of(src))


# ---- dtypes ----


def test_values_follow_asarray():
    assert S.csr_array([[1, 0]]).dtype is sd.int64
    assert S.csr_array([[True, False]]).dtype is sd.bool
    assert S.csr_array([[1.5, 0]]).dtype is sd.float64
    assert S.csr_array([[1j, 0]]).dtype is sd.complex128
    # numpy's lossless dtypes widen, as sd.asarray does ([scipy와 다름]: scipy keeps float32)
    assert S.csr_array(np.ones((2, 2), np.float32)).dtype is sd.float64
    assert S.coo_array((np.ones(2, np.int32), ([0, 1], [0, 1]))).dtype is sd.int64
    assert S.csr_array(sp.csr_array(np.ones((2, 2), np.float32))).dtype is sd.float64


def test_dtype_argument_converts_explicitly():
    A = S.csr_array(np.array([[1.7, 0], [np.nan, -3.2]]), dtype=sd.int64)
    assert A.dtype is sd.int64
    assert bits_equal(npy(A.data), np.array([1, 0, -3]))  # truncated, NaN -> 0 (a stored 0 stays)
    assert S.coo_array(([1, 2], ([0, 1], [0, 1])), dtype="float64").dtype is sd.float64
    assert S.csr_array(S.csr_array(D), dtype=sd.complex128).dtype is sd.complex128


@pytest.mark.parametrize(
    "call, kind, text",
    [
        (lambda: S.csr_array(D, dtype="float32"), TypeError, "csr_array(): 'dtype' must be sd.bool, sd.int64"),
        (lambda: S.csr_array(D, dtype=np.float64), TypeError, "csr_array(): 'dtype' must be"),
        (lambda: S.coo_array(np.ones((2, 2), np.uint64)), TypeError, "coo_array(): "),
        (lambda: S.csr_array(np.array([[1j]]), dtype=sd.float64), TypeError, "csr_array(): cannot convert complex128"),
        (lambda: S.csr_array([["a"]]), TypeError, "csr_array(): cannot store an object of type 'str'"),
    ],
)
def test_dtype_errors(call, kind, text):
    with pytest.raises(kind) as e:
        call()
    assert str(e.value).startswith(text)


# ---- shapes and index checks ----


def test_shape_is_inferred_from_the_largest_index():
    assert S.coo_array(([1.0], ([4], [7]))).shape == (5, 8)
    assert S.csr_array(([1.0, 2.0], [6, 1], [0, 1, 2])).shape == (2, 7)
    assert S.csc_array(([1.0, 2.0], [6, 1], [0, 1, 2])).shape == (7, 2)


@pytest.mark.parametrize(
    "call, kind, text",
    [
        (lambda: S.coo_array(([1.0], ([2], [0])), shape=(2, 2)), ValueError,
         "coo_array(): index 2 is out of bounds for axis 0 with size 2"),
        (lambda: S.csr_array(([1.0], ([0], [5])), shape=(2, 2)), ValueError,
         "csr_array(): index 5 is out of bounds for axis 1 with size 2"),
        (lambda: S.coo_array(([1.0], ([-1], [0]))), ValueError, "coo_array(): index -1 on axis 0 is negative"),
        (lambda: S.coo_array(([1.0], ([-1], [0])), shape=(2, 2)), ValueError,
         "coo_array(): index -1 is out of bounds for axis 0 with size 2"),
        (lambda: S.coo_array(([1.0], ([0.0], [0]))), TypeError,
         "coo_array(): 'coords' must hold integers, not float64"),
        (lambda: S.coo_array(([1.0], ([True], [0]))), TypeError, "coo_array(): 'coords' must hold integers, not bool"),
        (lambda: S.coo_array((np.array([]), ([], []))), ValueError,
         "coo_array(): cannot infer the shape from empty index arrays; pass 'shape'"),
        (lambda: S.coo_array(([1.0, 2.0], ([0], [0]))), ValueError,
         "coo_array(): 'data' and the index arrays must have the same length, got 2 and 1"),
        (lambda: S.csc_array(([1.0], ([0],))), ValueError, "csc_array(): a csc_array has 2 dimensions"),
        (lambda: S.csc_array(np.ones(3)), ValueError, "csc_array(): a dense input must have 2 dimensions, got 1"),
        (lambda: S.csr_array(np.ones((2, 2, 2))), ValueError,
         "csr_array(): a dense input must have 1 or 2 dimensions, got 3"),
        (lambda: S.csr_array(np.ones((2, 2)), shape=(3, 3)), ValueError,
         "csr_array(): 'shape' (3, 3) does not match the input's shape (2, 2)"),
        (lambda: S.csr_array((2, -1)), ValueError, "csr_array(): negative dimensions are not allowed"),
        (lambda: S.csr_array(D, shape=(2, 2.0)), TypeError,
         "csr_array(): 'shape' must be an int or a tuple of ints, not a tuple holding float"),
        (lambda: S.csr_array(D, shape=(1, 2, 3)), ValueError,
         "csr_array(): 'shape' must have 1 or 2 dimensions, got 3"),
        (lambda: S.csr_array(3.0), TypeError, "csr_array(): 'arg' must be a sparse array, an sd.ndarray"),
        (lambda: S.csr_array(None), TypeError, "csr_array(): 'arg' must be a sparse array"),
        (lambda: S.csr_array("abc"), TypeError, "csr_array(): 'arg' must be a sparse array"),
        (lambda: S.csr_array(D, copy=1), TypeError, "csr_array(): 'copy' must be True, False or None, not int"),
        (lambda: S.csr_array(D, 3), TypeError, "csr_array(): takes 1 positional argument but 2 were given"),
        (lambda: S.csr_array(D, format="csr"), TypeError, "csr_array(): unexpected keyword argument 'format'"),
        (lambda: S.sparray(D), TypeError, "sparray(): the base class is not constructed directly"),
    ],
)  # fmt: skip
def test_construction_errors(call, kind, text):
    with pytest.raises(kind) as e:
        call()
    assert str(e.value).startswith(text), str(e.value)


@pytest.mark.parametrize(
    "parts, text",
    [
        (([1.0], [0], [0, 1, 1, 1]), "'indptr' must have 3 entries (one more than the number of rows), got 4"),
        (([1.0], [0], [1, 1, 1]), "'indptr' must start at 0, not 1"),
        (([1.0, 2.0], [0, 1], [0, 2, 1]), "'indptr' must not decrease"),
        (([1.0], [0], [0, 1, 2]), "'indptr' must end at the number of stored entries, 1, not 2"),
        (([1.0], [5], [0, 1, 1]), "index 5 is out of bounds for axis 1 with size 3"),
        (([1.0, 2.0], [0], [0, 1, 1]), "'data' and 'indices' must have the same length, got 2 and 1"),
    ],
)
def test_compressed_parts_errors(parts, text):
    with pytest.raises(ValueError) as e:
        S.csr_array(parts, shape=(2, 3))
    assert str(e.value) == "csr_array(): " + text


# ---- copy ----


def test_copy_none_shares_the_values_and_never_the_indices():
    data = sd.asarray([1.0, 2.0, 3.0])
    indices, indptr = sd.asarray([0, 2, 1]), sd.asarray([0, 2, 3])
    A = S.csr_array((data, indices, indptr), shape=(2, 3))
    assert A.data is data  # canonical input: the values are shared
    indices[0] = 2  # the index arrays were copied: the array stays canonical
    indptr[1] = 0
    assert bits_equal(npy(A.indices), np.array([0, 2, 1])) and bits_equal(npy(A.indptr), np.array([0, 2, 3]))
    C = S.coo_array((data, (sd.asarray([0, 1, 1]), sd.asarray([0, 0, 2]))), copy=True)
    assert not np.shares_memory(npy(C.data), npy(data))


@pytest.mark.parametrize(
    "call, func, reason",
    [
        (lambda d: S.csr_array((d, ([1, 0], [0, 1])), copy=False), "csr_array", "sorting or summing duplicates"),
        (lambda d: S.csr_array(D, copy=False), "csr_array", "a dense input"),
        (lambda d: S.csr_array(sp.csr_array(D), copy=False), "csr_array", "an input from scipy"),
        (lambda d: S.coo_array(([1.0, 2.0], ([0, 1], [0, 1])), copy=False), "coo_array", None),
        (lambda d: S.dok_array((d, ([0, 1], [0, 1])), copy=False), "dok_array", "a dok_array"),
        (lambda d: S.csc_array(S.csr_array(D), copy=False), "csc_array", "converting csr to csc"),
        (lambda d: S.csr_array(S.csr_array(D), dtype=sd.int64, copy=False), "csr_array", "a dtype conversion"),
    ],
)
def test_copy_false_when_a_copy_is_needed(call, func, reason):
    with pytest.raises(ValueError) as e:
        call(sd.asarray([1.0, 2.0]))
    msg = str(e.value)
    assert msg.startswith(f"{func}(): copy=False, but ")
    if reason:
        assert msg == f"{func}(): copy=False, but {reason} requires a copy"


def test_copy_false_when_nothing_is_copied():
    data = sd.asarray([1.0, 2.0])
    assert S.coo_array((data, ([0, 1], [0, 1])), copy=False).data is data
    assert S.csr_array((data, ([0, 1], [0, 1])), copy=False).data is data  # canonical: no copy needed
    A = S.csr_array(D)
    assert S.csr_array(A, copy=False).data is A.data


def test_same_format_input_is_a_new_array_sharing_memory():
    A = S.csr_array(D)
    B = S.csr_array(A)
    assert B is not A and B.data is A.data and B.indices is A.indices
    C = S.csr_array(A, copy=True)
    assert not np.shares_memory(npy(C.data), npy(A.data))
    K = S.dok_array(D)
    L = S.dok_array(K)  # scipy shares the dict too
    L[0, 1] = 9.0
    assert K[0, 1] == 9.0
    M_ = S.dok_array(K, copy=True)
    M_[0, 1] = 1.0
    assert K[0, 1] == 9.0


# ---- attributes ----


def test_index_arrays_are_read_only_and_data_is_writable():
    A = S.csr_array(D)
    for arr in (A.indices, A.indptr):
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            arr[0] = 1
        assert not np.asarray(arr).flags.writeable
    A.data[0] = 5.0
    A.data *= 2
    assert A.toarray()[0, 0] == 10.0
    C = S.coo_array(D)
    for arr in C.coords:
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 1
    with pytest.raises(ValueError, match=r"\.data: the values must have shape \(4,\), got \(1,\)"):
        A.data = sd.asarray([1.0])
    with pytest.raises(TypeError, match=r"\.data: the values must be an sd.ndarray, not list"):
        A.data = [1.0, 2.0, 3.0, 4.0]
    A.data = sd.asarray([1, 2, 3, 4])  # a new dtype is allowed, as in scipy
    assert A.dtype is sd.int64
    for name in ("coords",):
        with pytest.raises(AttributeError):
            setattr(C, name, ())
    for name in ("indices", "indptr"):
        with pytest.raises(AttributeError):
            setattr(A, name, A.indices)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_attributes(cls):
    A = cls(D)
    assert A.shape == (3, 3) and A.ndim == 2 and A.size == 9 and A.nnz == 4 and A.dtype is sd.float64
    assert A.format == cls.format
    assert repr(A) == f"<{cls.__name__} of shape (3, 3), float64, 4 stored elements>" == str(A)
    assert repr(cls((2,), dtype=sd.int64) if cls is not S.csc_array else cls((1, 2), dtype=sd.int64)).endswith(
        "int64, 0 stored elements>"
    )
    one = cls(np.array([[0.0, 1.0]]))
    assert repr(one).endswith("1 stored element>")
    assert S.issparse(A) and isinstance(A, S.sparray)
    if cls in (S.coo_array, S.dok_array):  # CSR and CSC hold an indptr entry per line
        assert cls((10**10, 10**10)).size == 10**20


def test_has_canonical_format():
    assert S.coo_array(D).has_canonical_format
    assert not S.coo_array(([1.0, 2.0], ([1, 0], [0, 0]))).has_canonical_format
    assert S.coo_array(([1.0, 2.0], ([0, 1], [0, 0]))).has_canonical_format  # found out, not assumed
    assert not S.coo_array(([1.0, 2.0], ([0, 0], [0, 0]))).has_canonical_format  # a repeat
    assert S.csr_array(D).tocoo().has_canonical_format


def test_issparse_and_scipy_objects():
    assert not S.issparse(sp.csr_array(D)) and not S.issparse(sd.asarray(D)) and not S.issparse(D)


# ---- protocol ----


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_no_implicit_dense_conversion(cls):
    A = cls(D)
    with pytest.raises(TypeError, match=r"use A.toarray\(\)"):
        np.asarray(A)
    with pytest.raises(TypeError):
        sd.asarray(A)
    with pytest.raises(TypeError, match="length of a sparse array is ambiguous"):
        len(A)
    with pytest.raises(TypeError, match="truth value of a sparse array is ambiguous"):
        bool(A)
    with pytest.raises(TypeError, match="not iterable"):
        iter(A)


def test_numpy_scalars_and_indices_are_ints():
    A = S.coo_array(([1.0], ([np.int64(1)], [np.int32(2)])), shape=(np.int64(3), 4))
    assert A.shape == (3, 4)
    assert isinstance(A.shape[0], int)


def test_large_shape_without_entries():
    A = S.coo_array((10**12, 10**12))
    assert A.shape == (10**12, 10**12) and A.nnz == 0
    B = S.coo_array(([1.0], ([10**12 - 1], [5])), shape=(10**12, 10**12))
    assert B.nnz == 1 and i64(B.coords[0])[0] == 10**12 - 1
