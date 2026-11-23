"""The builders of SP3 (docs/api/sparse.md §4, docs/design/sparse.md §3.4): eye_array,
diags_array, random_array, kron, kronsum, hstack, vstack, block_array, block_diag, tril, triu,
find.

Every value is an input value or the array's product of two, so toarray() is the dense result bit
for bit (sd.eye, sd.tril, sd.triu, the dense blocks; sd.linalg.kron up to the sign of a zero: an
implicit zero is a strong zero). Against scipy 1.18.1
(live): the stored entries, the dtype and the format -- scipy's, except where scipy gives DIA or
BSR (eye_array, diags_array, kron), which are CSR here (decision SP51), and int64 diagonals give
int64 (SP52; scipy's default is float64 with a FutureWarning). random_array draws with an
sd.random.Generator (SP49): the same state gives the same array on every thread count."""

import numpy as np
import pytest
import scipy.sparse as sp

import stardust as sd

from .._bits import bits_equal, value_equal
from . import matrices as M
from .conftest import LIVE, coo, npy

S = sd.sparse
FORMATS = ("coo", "csr", "csc", "dok")


def entries(A):
    c = A.tocoo()
    coords = [np.asarray(x).tolist() for x in c.coords]
    vals = np.asarray(c.data)
    return sorted(zip(*coords, [v.tobytes() for v in vals]))


def same_as_scipy(a, b, fmt=None):
    """The same stored entries (bits), dtype, shape and format (fmt: the format sd gives). A bool
    product is int64 in sd (the array's multiply; scipy keeps bool)."""
    assert a.shape == b.shape
    if b.dtype == np.bool_ and a.dtype == sd.int64:
        b = b.astype(np.int64)
    assert str(a.dtype) == b.dtype.name, (a.dtype, b.dtype)
    assert a.format == (fmt or b.format), (a.format, b.format)
    assert entries(a) == entries(b)


def mats():
    r = np.random.default_rng(1)

    def rnd(m, n, d=0.4, dt=np.float64):
        D = r.standard_normal((m, n)) * (r.random((m, n)) < d)
        if dt in (np.int64, bool):
            D = np.round(D * 5)
            D[0, 0] = 1  # never empty (scipy's kron of an empty operand is float64)
        if dt == np.complex128:
            D = D + 1j * r.standard_normal((m, n)) * (D != 0)
        return D.astype(dt)

    return [rnd(3, 4), rnd(2, 2), rnd(4, 3, dt=np.int64), rnd(1, 5), rnd(3, 3, dt=np.complex128), rnd(2, 3, dt=bool)]


def both(D, fmt):
    return S.csr_array(sd.asarray(D)).asformat(fmt), sp.csr_array(D).asformat(fmt)


# ---- kron, kronsum ----


@LIVE
@pytest.mark.parametrize("f1", FORMATS)
@pytest.mark.parametrize("f2", FORMATS)
def test_kron_and_kronsum_store_what_scipy_stores(f1, f2):
    ms = mats()
    for i, j in [(0, 1), (1, 0), (2, 4), (3, 5), (5, 5), (4, 2)]:
        A, As = both(ms[i], f1)
        B, Bs = both(ms[j], f2)
        same_as_scipy(S.kron(A, B), sp.kron(As, Bs, format="csr"))  # scipy's default is BSR
        if A.dtype == B.dtype == sd.bool:
            assert S.kron(A, B).dtype == sd.int64  # the array's multiply of bools
        # the same products; the dense kron's 0 * (-x) is -0.0 where sd stores nothing (a strong zero)
        assert value_equal(npy(S.kron(A, B).toarray()), npy(sd.linalg.kron(A.toarray(), B.toarray())))
    for i in (1, 4, 5):
        B, Bs = both(ms[i if i != 5 else 1], f2)
        A, As = both(ms[4], f1)
        same_as_scipy(S.kronsum(A, B), sp.kronsum(As, Bs))


@pytest.mark.parametrize("f1", FORMATS)
def test_kronsum_is_the_dense_kronsum(f1):
    # values placed, not multiplied by the identity (SP111): bit for bit sd.linalg.kronsum but for
    # the zeros sparse does not store -- complex inf and -0.0 parts included
    g = np.random.default_rng(11)
    inf = np.inf
    A = np.array(
        [[complex(0, inf), complex(1, -0.0), 0], [0, complex(2, -0.0), complex(-inf, 0)], [3, 0, complex(0, -0.0) + 1]]
    )
    B = np.array([[1, complex(0, -inf)], [complex(-0.0, -0.0) + 2j, -3]])
    cases = [(A, B), (g.standard_normal((4, 4)) * (g.random((4, 4)) < 0.5), g.standard_normal((3, 3)))]
    cases.append((np.array([[np.nan, 1], [0, -inf]]), np.array([[inf, 0], [2, np.nan]])))
    for a, b in cases:
        sa = S.csr_array(sd.asarray(a)).asformat(f1)
        sb = S.csr_array(sd.asarray(b))
        K = S.kronsum(sa, sb)
        D = npy(sd.linalg.kronsum(sd.asarray(a), sd.asarray(b)))
        c = K.tocoo()
        r, j, v = (npy(x) for x in (c.coords[0], c.coords[1], c.data))
        assert np.array_equal(v.view(np.uint8), D[r, j].view(np.uint8))  # every stored value, its bits
        rest = np.ones(D.shape, bool)
        rest[r, j] = False
        assert not np.any(D[rest])  # what is not stored is a zero of the dense result


def test_kron_of_an_empty_operand_keeps_the_dtype():
    # scipy 1.18.1 gives float64 when an operand stores nothing (it builds an empty default array)
    E = S.csr_array((2, 2), dtype=sd.bool)
    assert S.kron(E, E).dtype == sd.int64 and S.kron(E, S.eye_array(2, dtype="complex128")).dtype == sd.complex128


def test_kron_keeps_stored_zeros_and_takes_1d_as_rows():
    a = S.coo_array((sd.asarray([0.0, 2.0]), (sd.asarray([0, 2]),)), shape=(3,))
    b = S.csr_array(sd.asarray([1.0, 3.0]))
    k = S.kron(a, b)
    assert k.shape == (6,) and k.format == "csr" and k.nnz == 4
    assert npy(k.toarray()).tolist() == [0, 0, 0, 0, 2, 6]
    M2 = S.csr_array(sd.asarray(np.eye(2)))
    assert S.kron(a, M2).shape == (2, 6) and S.kron(M2, a).shape == (2, 6)


# ---- stacking ----


@LIVE
@pytest.mark.parametrize("f1", FORMATS)
@pytest.mark.parametrize("f2", FORMATS)
def test_stacks_store_what_scipy_stores(f1, f2):
    ms = mats()
    A, As = both(ms[0], f1)
    A2, A2s = both(ms[0], f2)
    B, Bs = both(ms[1], f2)
    C, Cs = both(ms[2], f2)
    same_as_scipy(S.hstack([A, A2]), sp.hstack([As, A2s]))
    same_as_scipy(S.vstack([A.T, C]), sp.vstack([As.T, Cs]))
    same_as_scipy(S.block_array([[A, None], [None, B]]), sp.block_array([[As, None], [None, Bs]]))
    same_as_scipy(S.block_array([[A, A2], [A2, A]]), sp.block_array([[As, A2s], [A2s, As]]))
    same_as_scipy(S.block_diag([A, B, C]), sp.block_diag([As, Bs, Cs]))
    for fmt in FORMATS:
        same_as_scipy(S.vstack([A, A2], format=fmt), sp.vstack([As, A2s], format=fmt))
    same_as_scipy(S.hstack([A, A2], dtype="complex128"), sp.hstack([As, A2s], dtype=np.complex128))


@LIVE
def test_stacks_of_1d_blocks_are_rows_and_dense_blocks_count_their_nonzeros():
    a, b = sd.asarray([1.0, 0, 2]), sd.asarray([0.0, 3])
    same_as_scipy(S.hstack([S.coo_array(a), S.csr_array(b)]), sp.hstack([sp.coo_array(npy(a)), sp.csr_array(npy(b))]))
    same_as_scipy(S.vstack([S.coo_array(a), S.coo_array(a)]), sp.vstack([sp.coo_array(npy(a))] * 2))
    D = sd.asarray(np.array([[1.0, 0], [0, 2]]))
    got = S.hstack([S.csr_array(D), D])
    same_as_scipy(got, sp.hstack([sp.csr_array(npy(D)), npy(D)]))


def test_stack_errors():
    A = S.csr_array(sd.asarray(np.eye(2)))
    with pytest.raises(ValueError, match=r"vstack\(\): blocks\[:,0\] has incompatible column dimensions"):
        S.vstack([A, S.csr_array((2, 3))])
    with pytest.raises(ValueError, match=r"hstack\(\): blocks\[0,:\] has incompatible row dimensions"):
        S.hstack([A, S.csr_array((3, 3))])
    with pytest.raises(
        TypeError, match=r"hstack\(\): blocks\[1\] must be a sparse array, an sd.ndarray or None, not list"
    ):
        S.hstack([A, [[1, 2]]])
    with pytest.raises(TypeError, match=r"vstack\(\): 'blocks' must be a list or a tuple, not csr_array"):
        S.vstack(A)
    with pytest.raises(ValueError, match=r"block_array\(\): some block must be given"):
        S.block_array([[None]])
    with pytest.raises(ValueError, match=r"block_diag\(\): 'arrays' must hold at least one array"):
        S.block_diag([])
    with pytest.raises(ValueError, match=r"'format' must be 'coo', 'csr', 'csc', 'dok' or None, not 'bsr'"):
        S.hstack([A], format="bsr")
    with pytest.raises(TypeError, match=r"hstack\(\): unexpected keyword argument 'axis'"):
        S.hstack([A], axis=0)


# ---- tril, triu, find ----


@LIVE
@pytest.mark.parametrize("mid", ["coo/dups", "coo/complex", "rect/20x60", "rect/60x20", "empty/0x5", "coo/nan_inf"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_tril_triu_find(mid, fmt):
    m = M.get(mid)
    A = coo(m).asformat(fmt)
    B = sp.coo_array((m.data, (m.row, m.col)), shape=m.shape).asformat(fmt)
    for k in (0, 1, -2, 70, -70):
        for f, g, h in ((S.tril, sp.tril, sd.tril), (S.triu, sp.triu, sd.triu)):
            got = f(A, k=k)
            same_as_scipy(got, g(B, k=k))
            assert bits_equal(npy(got.toarray()), npy(h(A.toarray(), k=k)))
            if fmt == "coo":  # repeats and stored order kept (scipy's)
                want = g(B, k=k)
                assert [npy(c).tolist() for c in got.coords] == [c.tolist() for c in want.coords]
    r, c, v = S.find(A)
    rs, cs, vs = sp.find(B)
    assert npy(r).tolist() == rs.tolist() and npy(c).tolist() == cs.tolist() and bits_equal(npy(v), vs)
    assert all(x._flags()["writeable"] for x in (r, c, v))


def test_tril_of_1d_is_an_error():
    with pytest.raises(ValueError, match=r"tril\(\): 'A' must have 2 dimensions, got 1"):
        S.tril(S.csr_array(sd.asarray([1.0, 2])))
    with pytest.raises(TypeError, match=r"triu\(\): 'k' must be an int, not float"):
        S.triu(S.csr_array(sd.asarray(np.eye(2))), k=1.0)
    with pytest.raises(TypeError, match=r"tril\(\): 'A' must be a sparse array or an sd.ndarray, not numpy.ndarray"):
        S.tril(np.eye(2))


# ---- eye_array, diags_array ----


@LIVE
def test_eye_array_is_sd_eye_and_stores_what_scipy_stores():
    for m, n, k in [(3, 3, 0), (3, 5, 1), (5, 3, -2), (2, 3, 3), (0, 0, 0), (4, 4, -3), (1, 6, 2)]:
        for dtype in (None, "int64", "bool", "complex128"):
            for fmt in (None, "csr", "csc", "coo", "dok"):
                a = S.eye_array(m, n, k=k, dtype=dtype, format=fmt)
                b = sp.eye_array(m, n, k=k, dtype=np.dtype(dtype or "float64"), format=fmt or "csr")
                same_as_scipy(a, b, fmt or "csr")
                assert bits_equal(npy(a.toarray()), npy(sd.eye(m, n, k=k, dtype=dtype)))
    assert S.eye_array(2, 3, k=5).nnz == 0  # sd.eye's empty diagonal (scipy: ValueError)


@LIVE
def test_diags_array_stores_what_scipy_stores():
    cases = [
        (sd.asarray([1.0, 2, 3]), 0, None),
        ([sd.asarray([1, 2]), sd.asarray([3])], [0, 1], None),
        ([sd.asarray([1.0, 0, 2]), 5.0], [-1, 1], (4, 4)),
        (sd.asarray([True, False]), 0, None),
        ([sd.asarray([1, 2, 9]), sd.asarray([3, 4])], [0, -1], (3, 3)),
        ([sd.asarray([1 + 2j, 3])], [2], (2, 4)),
        ([sd.asarray([1.0, 2])], [2], (2, 2)),
    ]
    for diags, offs, shape in cases:
        a = S.diags_array(diags, offsets=offs, shape=shape)
        npd = [npy(d) if isinstance(d, sd.ndarray) else d for d in diags] if isinstance(diags, list) else npy(diags)
        b = sp.diags_array(npd, offsets=offs, shape=shape, dtype=None, format="csr")
        same_as_scipy(a, b)
        c = S.diags_array(diags, offsets=offs, shape=shape, dtype="complex128", format="coo")
        same_as_scipy(c, sp.diags_array(npd, offsets=offs, shape=shape, dtype=np.complex128, format="coo"))
    assert S.diags_array(sd.asarray([1, 2])).dtype == sd.int64  # SP52


def test_diags_array_errors():
    with pytest.raises(ValueError, match=r"diags_array\(\): the offsets repeat a value"):
        S.diags_array([sd.asarray([1.0]), sd.asarray([2.0])], offsets=[0, 0])
    with pytest.raises(ValueError, match=r"diags_array\(\): offset 3 \(diagonal 0\) is outside"):
        S.diags_array([sd.asarray([1.0])], offsets=[3], shape=(2, 2))
    with pytest.raises(ValueError, match=r"diags_array\(\): diagonal 0 has 2 values"):
        S.diags_array([sd.asarray([1.0, 2])], offsets=[0], shape=(3, 3))
    with pytest.raises(ValueError, match=r"diags_array\(\): 2 diagonals but 1 offsets"):
        S.diags_array([sd.asarray([1.0]), sd.asarray([2.0])], offsets=[0])
    with pytest.raises(TypeError, match=r"diags_array\(\): with one offset, 'diagonals' must be one diagonal"):
        S.diags_array([1.0, 2.0])
    with pytest.raises(TypeError, match=r"diags_array\(\): diagonal 0 must be an sd.ndarray or a number, not list"):
        S.diags_array([[1.0, 2.0]], offsets=[0])


# ---- random_array ----


def test_random_array_needs_an_rng_and_is_reproducible(restore_threads):
    with pytest.raises(TypeError, match=r"^random_array\(\): missing 1 required keyword-only argument: 'rng'$"):
        S.random_array((3, 4))
    with pytest.raises(TypeError, match=r"random_array\(\): 'rng' must be an sd.random.Generator .*not NoneType"):
        S.random_array((3, 4), rng=None)
    with pytest.raises(TypeError, match=r"random_array\(\): 'rng' must be an sd.random.Generator .*not int"):
        S.random_array((3, 4), rng=0)
    with pytest.raises(ValueError, match=r"random_array\(\): 'density' must be in \[0, 1\], got 1.5"):
        S.random_array((3, 4), density=1.5, rng=sd.random.default_rng(0))
    base = None
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        A = S.random_array((300, 400), density=0.2, rng=sd.random.default_rng(42))
        enc = (npy(A.coords[0]).tobytes(), npy(A.coords[1]).tobytes(), npy(A.data).tobytes())
        if base is None:
            base = enc
        assert enc == base
    assert A.format == "coo" and A.nnz == round(0.2 * 300 * 400)
    flat = npy(A.coords[0]) * 400 + npy(A.coords[1])
    assert len(set(flat.tolist())) == A.nnz  # no repeats
    v = npy(A.data)
    assert v.min() >= 0 and v.max() < 1


def test_random_array_dtypes_and_sampler():
    g = sd.random.default_rng(3)
    for dtype in ("float64", "int64", "complex128", "bool"):
        A = S.random_array((20, 30), density=0.3, dtype=dtype, rng=g, format="csr")
        assert str(A.dtype) == dtype and A.nnz == 180 and A.format == "csr"
    B = S.random_array((10,), density=0.5, rng=g, data_sampler=lambda n: sd.ones(n, dtype=sd.int64))
    assert B.shape == (10,) and B.dtype == sd.int64 and B.nnz == 5
    with pytest.raises(ValueError, match=r"data_sampler\(5\) must give 5 values"):
        S.random_array((10,), density=0.5, rng=g, data_sampler=lambda n: sd.ones(3))
    huge = S.random_array((2**40, 2**40), density=2**-78, rng=g)  # more positions than an int64 counts
    assert huge.nnz == 4 and len({tuple(p) for p in zip(*(npy(c).tolist() for c in huge.coords))}) == 4


# ---- the format argument ----


@pytest.mark.parametrize(
    "call",
    [lambda f: S.eye_array(3, format=f), lambda f: S.kron(S.eye_array(2), S.eye_array(2), format=f),
     lambda f: S.tril(S.eye_array(2), format=f), lambda f: S.diags_array(sd.asarray([1.0]), format=f),
     lambda f: S.random_array((2, 2), rng=sd.random.default_rng(0), format=f)],
)  # fmt: skip
def test_format_names(call):
    for bad in ("dia", "bsr", "lil"):
        with pytest.raises(ValueError, match=rf"'format' must be 'coo', 'csr', 'csc', 'dok' or None, not '{bad}'"):
            call(bad)
    with pytest.raises(TypeError, match="'format' must be"):
        call(1)
    assert call(None).format in ("csr", "coo")


# ---- out of the int64 range and too large to allocate ----


def _one(shape, at):
    return S.coo_array((sd.asarray([1.0]), tuple(sd.asarray([i]) for i in at)), shape=shape)


def test_result_shapes_past_int64_are_an_overflow_error():
    """A built shape past int64 is refused before anything is built (SPR-2): no negative index."""
    A = S.csr_array(_one((1, 2**32), (0, 2**32 - 1)))
    with pytest.raises(OverflowError, match=r"^kron\(\): the result's shape is out of range$"):
        S.kron(A, A)
    v = S.coo_array((sd.asarray([1.0]), (sd.asarray([2**32 - 1]),)), shape=(2**32,))
    with pytest.raises(OverflowError, match=r"^kron\(\): the result's shape is out of range$"):
        S.kron(v, v)
    # rows as well as columns, and before the inputs' own conversion (a COO of 2**62 rows has no CSR)
    for shape_a, shape_b in (((2**62, 1), (4, 1)), ((1, 2**62), (1, 4))):
        with pytest.raises(OverflowError, match=r"^kron\(\): the result's shape is out of range$"):
            S.kron(S.coo_array(shape_a), S.coo_array(shape_b))
    H = _one((1, 2**62 + 1), (0, 2**62))
    V = _one((2**62 + 1, 1), (2**62, 0))
    for f, arg in ((S.hstack, [H, H]), (S.vstack, [V, V]), (S.block_diag, [V, V]), (S.block_array, [[H, H]])):
        with pytest.raises(OverflowError, match=rf"^{f.__name__}\(\): the result's shape is out of range$"):
            f(arg, format="coo")
    with pytest.raises(OverflowError, match=r"^hstack\(\): the result's shape is out of range$"):
        S.hstack([H.tocsr(), H.tocsr()])
    big = S.coo_array((2**32, 2**32))
    with pytest.raises(OverflowError, match=r"^kronsum\(\): the result's shape is out of range$"):
        S.kronsum(big, big)  # (unchecked: a 32 GB arange)
    # just inside: the largest index stays in range
    K = S.kron(_one((1, 2**31), (0, 2**31 - 1)), _one((1, 2**31), (0, 2**31 - 1)))
    assert K.shape == (1, 2**62) and npy(K.indices).tolist() == [2**62 - 1]


def test_int_arguments_past_int64_are_an_overflow_error():
    """k, offsets, n_rows, n_cols outside int64: the dense twins' OverflowError (SPR-6)."""
    A = S.csr_array(sd.asarray([[1.0, 0, 2], [0, 3, 0]]))
    for fmt in ("csr", "csc", "coo", "dok"):
        B = A.asformat(fmt)
        with pytest.raises(OverflowError, match=r"^diagonal\(\): 'k' is out of range$"):
            B.diagonal(k=2**70)
        with pytest.raises(OverflowError, match=r"^trace\(\): 'k' is out of range$"):
            B.trace(k=-(2**70))
        with pytest.raises(OverflowError, match=r"^setdiag\(\): 'k' is out of range$"):
            B.copy().setdiag(1.0, k=2**70)
    for f in (S.tril, S.triu):
        with pytest.raises(OverflowError, match=rf"^{f.__name__}\(\): 'k' is out of range$"):
            f(A, k=2**63)
    assert S.tril(A, k=2**63 - 1).nnz == 3 and S.triu(A, k=-(2**63)).nnz == 3
    with pytest.raises(OverflowError, match=r"^diags_array\(\): 'offsets' is out of range$"):
        S.diags_array(sd.ones(2), offsets=2**70)
    with pytest.raises(OverflowError, match=r"^diags_array\(\): 'offsets' is out of range$"):
        S.diags_array([sd.ones(2)], offsets=[-(2**70)])
    with pytest.raises(OverflowError, match=r"^eye_array\(\): 'n_rows' is out of range$"):
        S.eye_array(2**70)
    with pytest.raises(OverflowError, match=r"^eye_array\(\): 'n_cols' is out of range$"):
        S.eye_array(2, 2**64)
    with pytest.raises(OverflowError, match=r"^eye_array\(\): 'k' is out of range$"):
        S.eye_array(2, k=2**70)
    assert S.eye_array(2, k=2**63 - 1).nnz == 0 and S.eye_array(2, k=-(2**63)).nnz == 0
    for f in (lambda: sd.eye(2**70), lambda: sd.tril(sd.ones((2, 2)), k=2**70)):  # the dense twins' form
        with pytest.raises(OverflowError, match=r"\(\): '(n_rows|k)' is out of range$"):
            f()


def test_allocation_failures_name_the_function_called():
    """An array kernel's allocation failure carries the name the user called (SPR-7)."""
    with pytest.raises(MemoryError, match=r"^csr_array\(\): cannot allocate \d+ bytes: "):
        S.csr_array((2**40, 3))
    with pytest.raises(MemoryError, match=r"^diags_array\(\): cannot allocate \d+ bytes: "):
        S.diags_array(1.0, offsets=0, shape=(2**40, 2**40))
    with pytest.raises(MemoryError, match=r"^eye_array\(\): cannot allocate \d+ bytes: "):
        S.eye_array(2**40)
    with pytest.raises(MemoryError, match=r"^kron\(\): cannot allocate \d+ bytes: "):
        S.kron(_one((2**40, 1), (0, 0)), S.eye_array(1))
    tall = _one((2**62 + 1, 1), (2**62, 0))
    with pytest.raises(ValueError, match=r"^tocsr\(\): array is too big$"):
        tall.tocsr()
    with pytest.raises(ValueError, match=r"^tocsc\(\): array is too big$"):
        S.coo_array((2**63 - 1, 2**63 - 1)).tocsc()  # an indptr of 2^63 entries
    with pytest.raises(ValueError, match=r"^csr_array\(\): array is too big$"):
        S.csr_array(tall)
    with pytest.raises(ValueError, match=r"^eye_array\(\): array is too big$"):
        S.eye_array(2**63 - 1)
    with pytest.raises(ValueError, match=r"^csr_array\(\): array is too big$"):
        S.csr_array((2**63 - 1, 1))
    with pytest.raises(ValueError, match=r"^resize\(\): array is too big$"):
        S.csr_array((2, 2)).resize((2**63 - 1, 2))
