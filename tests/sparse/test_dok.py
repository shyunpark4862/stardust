"""dok_array, the format for building one entry at a time (docs/api/sparse.md §2,
docs/design/sparse.md §2.1, §3.1): `A[i, j] = v` and `A[i, j]` by the array's rules for one
element, a zero removes the entry (scipy), no dict methods, conversions independent of the order
the entries went in, and a hash map that keeps working through growth and removals."""

import numpy as np
import pytest
import scipy.sparse as sp

import stardust as sd

from .._bits import bits_equal
from .conftest import LIVE, npy

S = sd.sparse


def test_set_and_get():
    A = S.dok_array((3, 4))
    A[0, 1] = 2.5
    A[2, 3] = 1  # an int into float64: allowed (no information lost)
    A[-1, 0] = 7.0  # negative counts from the end
    assert A.nnz == 3 and A[0, 1] == 2.5 and A[2, 0] == 7.0 and A[2, 3] == 1.0
    assert type(A[2, 3]) is float and A[1, 1] == 0.0 and type(A[1, 1]) is float
    A[np.int64(1), np.int32(2)] = 4.0  # numpy ints are ints
    assert A[1, 2] == 4.0
    assert bits_equal(npy(A.toarray()), np.array([[0, 2.5, 0, 0], [0, 0, 4, 0], [7, 0, 0, 1.0]]))


@pytest.mark.parametrize(
    "dtype, value, stored",
    [
        (sd.bool, True, True),
        (sd.int64, 2**62, 2**62),
        (sd.int64, False, None),
        (sd.float64, np.nan, "nan"),
        (sd.complex128, 1 + 2j, 1 + 2j),
        (sd.complex128, 3, 3 + 0j),
    ],
)
def test_values_by_the_array_rules(dtype, value, stored):
    A = S.dok_array((2, 2), dtype=dtype)
    A[0, 0] = value
    if stored is None:
        assert A.nnz == 0
    elif stored == "nan":
        assert A.nnz == 1 and np.isnan(A[0, 0])
    else:
        assert A.nnz == 1 and A[0, 0] == stored and type(A[0, 0]) is type(stored)


def test_zero_removes_the_entry():
    A = S.dok_array((2, 2))
    A[0, 0] = 1.0
    A[1, 1] = 2.0
    A[0, 0] = 0.0
    assert A.nnz == 1
    A[1, 1] = -0.0  # -0.0 == 0: removed too (scipy: `if x`)
    assert A.nnz == 0
    A[1, 0] = 0  # removing an absent key is nothing
    assert A.nnz == 0


@pytest.mark.parametrize(
    "key, value, kind, text",
    [
        ((2, 0), 1.0, IndexError, "index 2 is out of bounds for axis 0 with size 2"),
        ((0, -4), 1.0, IndexError, "index -4 is out of bounds for axis 1 with size 3"),
        ((0, 2**70), 1.0, IndexError, "is out of bounds for axis 1 with size 3"),
        # other keys are the general indexing of SP3, with the array's messages
        ((0.0, 1), 1.0, IndexError, "only integers, slices (`:`), ellipsis (`...`), newaxis (`None`) and integer"),
        ((True, 1), 1.0, IndexError, "a bool is not a valid index; use an int, or a boolean array"),
        ((0, 1, 2), 1.0, IndexError, "too many indices for array: array is 2-dimensional, but 3 were indexed"),
        ((0, 1), 1j, TypeError, "cannot implicitly convert complex128 to float64"),
        ((0, 1), "a", TypeError, "only a bool, int, float or complex can be assigned to a single element, not str"),
        ((0, 1), [1.0], TypeError, "only a bool, int, float or complex can be assigned to a single element, not list"),
    ],
)
def test_errors_are_pythons_protocol_form(key, value, kind, text):
    A = S.dok_array((2, 3))
    with pytest.raises(kind) as e:
        A[key] = value
    assert text in str(e.value)
    if kind is IndexError:
        with pytest.raises(kind):
            A[key]


def test_int_dtype_takes_no_float():
    A = S.dok_array((2, 2), dtype=sd.int64)
    with pytest.raises(TypeError, match="cannot implicitly convert float64 to int64"):
        A[0, 0] = 1.5


def test_one_dimensional():
    A = S.dok_array((5,))
    A[3] = 2.0
    A[(0,)] = 1.0
    A[-1] = 4.0
    assert A[3] == 2.0 and A[4] == 4.0 and A.nnz == 3
    assert bits_equal(npy(A.toarray()), np.array([1.0, 0, 0, 2, 4]))
    assert bits_equal(npy(A.tocsr().indices), np.array([0, 3, 4]))
    with pytest.raises(IndexError, match="too many indices for array: array is 1-dimensional, but 2 were indexed"):
        A[0, 0] = 1.0


def test_no_dict_methods():
    A = S.dok_array((2, 2))
    for name in ("keys", "values", "items", "get", "pop", "update", "setdefault", "clear", "popitem", "conjtransp"):
        assert not hasattr(A, name), name
    with pytest.raises(TypeError, match="length of a sparse array is ambiguous"):
        len(A)


def test_conversion_does_not_depend_on_insertion_order():
    g = np.random.Generator(np.random.PCG64(4))
    keys = [(int(i), int(j)) for i, j in g.integers(0, 50, (400, 2))]
    vals = g.standard_normal(len(keys))
    A, B = S.dok_array((50, 50)), S.dok_array((50, 50))
    for (i, j), v in zip(keys, vals):
        A[i, j] = float(v)
    last = {}
    for (i, j), v in zip(keys, vals):
        last[i, j] = float(v)
    for i, j in reversed(list(last)):
        B[i, j] = last[i, j]
    for f in ("tocoo", "tocsr", "tocsc"):
        a, b = getattr(A, f)(), getattr(B, f)()
        assert bits_equal(npy(a.toarray()), npy(b.toarray()))
        assert all(bits_equal(npy(x), npy(y)) for x, y in zip(a.nonzero(), b.nonzero()))
    C = A.tocoo()
    assert C.has_canonical_format  # row-major ([scipy와 다름]: scipy keeps insertion order)


def test_growth_and_removal():
    A = S.dok_array((1000, 1000), dtype=sd.int64)
    for k in range(5000):
        A[k % 1000, (7 * k) % 1000] = k + 1
    for k in range(0, 5000, 2):
        A[k % 1000, (7 * k) % 1000] = 0
    for k in range(0, 5000, 4):
        A[k % 1000, (7 * k) % 1000] = -k - 1
    want = np.zeros((1000, 1000), dtype=np.int64)
    for k in range(5000):
        want[k % 1000, (7 * k) % 1000] = k + 1
    for k in range(0, 5000, 2):
        want[k % 1000, (7 * k) % 1000] = 0
    for k in range(0, 5000, 4):
        want[k % 1000, (7 * k) % 1000] = -k - 1
    assert A.nnz == int(np.count_nonzero(want))
    assert bits_equal(npy(A.toarray()), want)


@LIVE
def test_conversions_keep_stored_zeros_like_scipy():
    c = sp.coo_array((np.array([0.0, 1.0, 0.0]), (np.array([0, 0, 1]), np.array([0, 1, 1]))), shape=(2, 2))
    A = S.coo_array(c).todok()
    assert A.nnz == c.todok().nnz == 3 and A[0, 0] == 0.0


def test_copy_transpose_resize_astype():
    A = S.dok_array(np.array([[1.0, 0, 2], [0, 3, 0]]))
    B = A.copy()
    B[0, 0] = 9.0
    assert A[0, 0] == 1.0
    T = A.T
    assert T.format == "dok" and T.shape == (3, 2) and T[2, 0] == 2.0
    A.resize((1, 2))
    assert A.shape == (1, 2) and A.nnz == 1
    with pytest.raises(IndexError):
        A[1, 0]
    K = S.dok_array(np.array([[1.9, -0.4]])).astype(sd.int64)
    assert K[0, 0] == 1 and K.nnz == 2 and K[0, 1] == 0
