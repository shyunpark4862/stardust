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


# ---- a shared map, the conversions' sort and the GIL ----


def test_resizing_a_shared_map_copies_it():
    """dok_array(D) shares D's map (element writes are seen by both); resizing one copies the map
    first (SP121), so the other keeps its shape and keys and its conversions stay in bounds."""
    D = S.dok_array((2, 2))
    D[0, 0] = 1.0
    D2 = S.dok_array(D)
    D2[1, 1] = 5.0
    assert D[1, 1] == 5.0  # shared until a resize
    D2[1, 1] = 0.0
    D.resize((1000, 1000))
    D[999, 999] = 7.0
    assert D2.shape == (2, 2) and D2.nnz == 1
    with pytest.raises(IndexError, match="index 999 is out of bounds for axis 0 with size 2"):
        D2[999, 999]
    C = D2.tocsr()
    assert npy(C.indices).tolist() == [0] and npy(C.indptr).tolist() == [0, 1, 1]
    assert D2.sum() == 1.0 and D2.max() == 1.0
    assert npy(sd.ones(2) @ C).tolist() == [1.0, 0.0]
    assert npy(D2.sum(axis=0)).tolist() == [1.0, 0.0]
    assert D.shape == (1000, 1000) and D.nnz == 2 and D[999, 999] == 7.0
    D[0, 0] = 3.0
    assert D2[0, 0] == 1.0  # no longer shared
    D2.resize((1, 1))
    assert D2.toarray().tolist() == [[1.0]] and D[0, 0] == 3.0
    # 1-D
    v = S.dok_array((3,))
    v[2] = 4.0
    w = S.dok_array(v)
    v.resize((2,))
    assert w.shape == (3,) and w[2] == 4.0 and npy(w.tocsr().indices).tolist() == [2] and v.nnz == 0


def _random_dok(n, nnz, seed, shape=None):
    g = np.random.default_rng(seed)
    shape = shape or (n, n)
    flat = g.choice(shape[0] * shape[1], nnz, replace=False)
    r, c = flat // shape[1], flat % shape[1]
    A = S.coo_array((sd.asarray(g.standard_normal(nnz)), (sd.asarray(r), sd.asarray(c))), shape=shape)
    return A, A.todok()


@pytest.mark.usefixtures("restore_threads")
@pytest.mark.parametrize("nnz", [0, 7, 40_000, 300_000])
def test_conversions_sort_by_counting_with_the_same_bits(nnz):
    """DOK -> CSR and COO take COO -> CSR's counting sort (SPR-13): the same arrays as converting
    the COO entries, for 1 and 10 threads (the pool's gather is cut by slot blocks)."""
    A, D = _random_dok(3000, nnz, nnz)
    want = A.tocsr()
    for t in (1, 10):
        sd.set_num_threads(t)
        got = D.tocsr()
        assert bits_equal(npy(got.indptr), npy(want.indptr)) and bits_equal(npy(got.indices), npy(want.indices))
        assert bits_equal(npy(got.data), npy(want.data))
        coo = D.tocoo()
        assert coo.has_canonical_format
        assert bits_equal(npy(coo.coords[0]), npy(want.tocoo().coords[0]))
        assert bits_equal(npy(coo.coords[1]), npy(want.indices)) and bits_equal(npy(coo.data), npy(want.data))
        assert bits_equal(npy(D.tocsc().toarray()), npy(want.toarray()))


def test_conversions_of_a_huge_shape_sort_the_keys():
    """Few entries in a huge shape: to_coo sorts the keys (no indptr of 2^62 rows), row-major."""
    D = S.dok_array((2**62, 2**62))
    D[2**61, 5] = 1.0
    D[3, 2**60] = 2.0
    D[3, 1] = 3.0
    c = D.tocoo()
    assert npy(c.coords[0]).tolist() == [3, 3, 2**61] and npy(c.coords[1]).tolist() == [1, 2**60, 5]
    assert npy(c.data).tolist() == [3.0, 2.0, 1.0]
    v = S.dok_array((10,))
    v[7] = 2.0
    v[3] = 1.0
    assert npy(v.tocsr().indices).tolist() == [3, 7] and npy(v.tocoo().coords[0]).tolist() == [3, 7]


@pytest.mark.serial
@pytest.mark.usefixtures("restore_threads")
def test_building_and_reading_a_map_release_the_gil():
    """todok and DOK -> CSR / COO touch no Python object: from the parallel threshold they run
    without the GIL on one thread too (API.md §3), so another Python thread keeps running."""
    import threading
    import time

    A, D = _random_dok(200_000, 1_500_000, 3, shape=(200_000, 200_000))
    C = A.tocsr()
    sd.set_num_threads(1)
    calls = {"csr.todok": C.todok, "coo.todok": A.todok, "dok.tocsr": D.tocsr, "dok.tocoo": D.tocoo}
    for name, call in calls.items():
        call()
        best = 1.0
        for _ in range(4):
            stop, longest = threading.Event(), [0.0]

            def watch(stop=stop, longest=longest):
                last = time.perf_counter()
                while not stop.is_set():
                    now = time.perf_counter()
                    longest[0] = max(longest[0], now - last)
                    last = now

            th = threading.Thread(target=watch)
            th.start()
            time.sleep(0.005)
            longest[0] = 0.0
            t0 = time.perf_counter()
            call()
            took = time.perf_counter() - t0
            time.sleep(0.002)
            stop.set()
            th.join()
            best = min(best, longest[0] / took)
            if best < 0.25:
                break
        assert best < 0.25, f"{name} held the GIL ({best:.2f} of the call)"
