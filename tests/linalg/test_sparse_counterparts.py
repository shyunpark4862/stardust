"""The dense counterparts of sd.sparse's builders and checks (docs/api/linalg.md §2, §4, §6; the
rule of API.md §7): kronsum, block_diag, bandwidth, kron of 1-D operands and lu_solve's trans.
Each against numpy / scipy, bit for bit where the work is copies or one rounding, and against
the sd.sparse function it stands for (its toarray()), bit for bit."""

import numpy as np
import pytest
import scipy.linalg as sl
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .._tier import examples
from .conftest import DTYPES, LAYOUTS, as_sd, batch_shapes, compute_dtype, npy, random_values, residual_ok, with_layout

SETTINGS = settings(max_examples=examples(100), deadline=None)
LA = sd.linalg


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == np.ascontiguousarray(b).tobytes()


def kronsum_ref(a, b):
    """kron(I_n, a) + kron(b, I_m) entry by entry: a on the diagonal blocks, b[i, j] on the
    diagonal of block (i, j), the one sum a[p, p] + b[i, i] where both are (numpy's add)."""
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    m, n = a.shape[-1], b.shape[-1]
    ct = np.result_type(a, b, np.float64)
    a = np.broadcast_to(a, (*batch, m, m)).astype(ct)
    b = np.broadcast_to(b, (*batch, n, n)).astype(ct)
    out = np.zeros((*batch, m * n, m * n), ct)
    idx = np.arange(m)
    for i in range(n):
        for j in range(n):
            if i == j:
                out[..., i * m : (i + 1) * m, i * m : (i + 1) * m] = a
                out[..., i * m + idx, i * m + idx] = a[..., idx, idx] + b[..., i, i][..., None]
            else:
                out[..., i * m + idx, j * m + idx] = b[..., i, j][..., None]
    return out


# ---------------------------------------------------------------- kronsum --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    drop=st.integers(0, 2),
    mn=st.tuples(st.integers(0, 4), st.integers(0, 4)),
    dtypes=st.tuples(st.sampled_from(DTYPES), st.sampled_from(DTYPES)),
    layouts=st.tuples(st.sampled_from(LAYOUTS), st.sampled_from(LAYOUTS)),
)
def test_kronsum(seed, batch, drop, mn, dtypes, layouts):
    rng = np.random.default_rng(seed)
    m, n = mn
    b2 = tuple(1 if i % 2 else s for i, s in enumerate(batch[drop:]))
    a = with_layout(random_values(rng, (*batch, m, m), dtypes[0]), layouts[0])
    b = with_layout(random_values(rng, (*b2, n, n), dtypes[1]), layouts[1])
    r = npy(LA.kronsum(as_sd(a), as_sd(b)))
    assert r.dtype == compute_dtype(*dtypes)
    assert same_bits(r, kronsum_ref(a, b))
    if r.ndim == 2 and r.size:
        assert r.flags.f_contiguous


def test_kronsum_values_stay_in_place():
    a = np.array([[1.0, np.inf], [np.nan, -0.0]])
    b = np.array([[-0.0, 2.0], [np.inf, 3.0]])
    r = npy(LA.kronsum(as_sd(a), as_sd(b)))
    assert same_bits(r, kronsum_ref(a, b))
    # no product with the identity: inf * 0 makes no NaN off the diagonal (np.kron(eye, a) would)
    assert np.isnan(r).sum() == 2 and np.isinf(r).sum() == 4
    assert np.signbit(r[0, 1]) is np.False_ and np.signbit(r[1, 1])  # +0 off the pattern, -0.0 kept
    with pytest.raises(ValueError, match="kronsum\\(\\): 'x1' must be square"):
        LA.kronsum(as_sd(np.ones((2, 3))), as_sd(np.ones((2, 2))))
    with pytest.raises(ValueError, match="'x2' must have at least 2 dimensions"):
        LA.kronsum(as_sd(np.ones((2, 2))), as_sd(np.ones(2)))
    with pytest.raises(TypeError, match="for a sparse array use sd.sparse.kronsum"):
        LA.kronsum(sd.sparse.csr_array(as_sd(np.eye(2))), as_sd(np.eye(2)))


@pytest.mark.parametrize("dt", ["float64", "complex128"])
def test_kronsum_is_sparse_kronsums_toarray(dt):
    rng = np.random.default_rng(1)
    for m, n in ((1, 1), (3, 4), (5, 2), (0, 3)):
        a = random_values(rng, (m, m), dt)
        b = random_values(rng, (n, n), dt)
        a[rng.random(a.shape) < 0.3] = 0  # a pattern for the sparse side
        b[rng.random(b.shape) < 0.3] = 0
        want = sd.sparse.kronsum(as_sd(a), as_sd(b)).toarray()
        assert same_bits(LA.kronsum(as_sd(a), as_sd(b)), want)


# ------------------------------------------------------------- block_diag --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    shapes=st.lists(st.tuples(st.integers(0, 3), st.integers(0, 3), st.booleans()), min_size=1, max_size=4),
    dtypes=st.lists(st.sampled_from(DTYPES), min_size=4, max_size=4),
    layouts=st.lists(st.sampled_from(LAYOUTS[:4]), min_size=4, max_size=4),
)
def test_block_diag_is_scipys(seed, shapes, dtypes, layouts):
    rng = np.random.default_rng(seed)
    blocks = []
    for (m, n, vec), dt, lo in zip(shapes, dtypes, layouts):
        x = random_values(rng, (n,) if vec else (m, n), dt)
        blocks.append(x if vec else with_layout(x, lo))
    r = npy(LA.block_diag([as_sd(x) for x in blocks]))
    want = sl.block_diag(*blocks)
    assert same_bits(r, want)


def test_block_diag_batches_broadcast_as_scipys():
    rng = np.random.default_rng(2)
    xs = [
        rng.standard_normal((3, 2, 2)),
        rng.standard_normal((1, 1)),
        rng.standard_normal(3),
        rng.standard_normal((2, 1, 1, 2)),
    ]
    r = npy(LA.block_diag([as_sd(x) for x in xs]))
    assert r.shape == (2, 3, 5, 8)
    assert same_bits(r, sl.block_diag(*xs))
    with pytest.raises(ValueError, match="block_diag\\(\\): batch shapes do not broadcast: \\(2,\\) \\(3,\\)"):
        LA.block_diag([as_sd(np.ones((2, 1, 1))), as_sd(np.ones((3, 1, 1)))])


def test_block_diag_rules():
    x = as_sd(np.ones((2, 2), np.int64))
    assert npy(LA.block_diag([x, as_sd(np.ones((1, 1), np.bool_))])).dtype == np.int64  # moved, not computed
    assert npy(LA.block_diag((x,))).tolist() == [[1, 1], [1, 1]]  # a tuple too
    with pytest.raises(ValueError, match="'arrays' must hold at least one array"):
        LA.block_diag([])
    with pytest.raises(TypeError, match="'arrays' must be a list or tuple of sd.ndarray, not ndarray"):
        LA.block_diag(x)
    with pytest.raises(TypeError, match="'arrays\\[1\\]' must be an sd.ndarray, not numpy.ndarray"):
        LA.block_diag([x, np.ones((2, 2))])
    with pytest.raises(TypeError, match="for a sparse array use sd.sparse.block_diag"):
        LA.block_diag([x, sd.sparse.csr_array(x)])
    with pytest.raises(TypeError):
        LA.block_diag(x, x)  # one list, not scipy's *arrs


@pytest.mark.parametrize("dt", DTYPES)
def test_block_diag_is_sparse_block_diags_toarray(dt):
    rng = np.random.default_rng(3)
    xs = [random_values(rng, s, dt) for s in ((2, 3), (4,), (0, 2), (3, 3))]
    for x in xs:
        if dt in ("float64", "complex128"):
            x[rng.random(x.shape) < 0.3] = 0
    want = sd.sparse.block_diag([as_sd(x) for x in xs]).toarray()
    assert same_bits(LA.block_diag([as_sd(x) for x in xs]), want)


# -------------------------------------------------------------- bandwidth --


def banded(rng, shape, lo, up, dt):
    x = random_values(rng, shape, dt)
    if dt == "bool":
        x = x | True
    return np.triu(np.tril(x, up), -lo)


def band_ref(x):
    """(lower, upper) of every matrix of x from its nonzero positions."""
    batch = x.shape[:-2]
    lo, up = np.zeros(batch, np.int64), np.zeros(batch, np.int64)
    for idx in np.ndindex(*batch):
        i, j = np.nonzero(x[idx] != 0)
        lo[idx] = max(0, (i - j).max(initial=0))
        up[idx] = max(0, (j - i).max(initial=0))
    return lo, up


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    mn=st.tuples(st.integers(0, 6), st.integers(0, 6)),
    band=st.tuples(st.integers(0, 6), st.integers(0, 6)),
    dt=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_bandwidth(seed, batch, mn, band, dt, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(banded(rng, (*batch, *mn), *band, dt), layout)
    lower, upper = LA.bandwidth(as_sd(x))
    want_lo, want_up = band_ref(x)
    if x.ndim == 2:
        assert type(lower) is int and type(upper) is int
        assert (lower, upper) == (int(want_lo), int(want_up))
    else:
        assert npy(lower).dtype == np.int64 and npy(lower).shape == batch
        assert np.array_equal(npy(lower), want_lo) and np.array_equal(npy(upper), want_up)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_bandwidth_is_scipys(layout):
    rng = np.random.default_rng(9)
    for shape, lo, up in (((7, 7), 2, 1), ((4, 6, 6), 0, 5), ((3, 5, 8), 4, 2), ((8, 3), 6, 0)):
        x = with_layout(banded(rng, shape, lo, up, "float64"), layout)
        r, want = LA.bandwidth(as_sd(x)), sl.bandwidth(x)
        assert np.array_equal(npy(r.lower), want[0]) and np.array_equal(npy(r.upper), want[1])


def test_bandwidth_rules():
    r = LA.bandwidth(as_sd(np.triu(np.ones((4, 4)))))
    assert r == (0, 3) and (r.lower, r.upper) == (0, 3) and type(r).__name__ == "BandwidthResult"
    assert LA.bandwidth(as_sd(np.array([[0.0, 0.0], [np.nan, 0.0]]))) == (1, 0)  # NaN is nonzero
    assert LA.bandwidth(as_sd(np.array([[1.0, -0.0], [-0.0, 1.0]]))) == (0, 0)  # -0.0 is zero
    assert LA.bandwidth(as_sd(np.array([[1.0, 1e-300j], [0.0, 1.0]]))) == (0, 1)  # either part counts
    assert LA.bandwidth(as_sd(np.zeros((0, 4)))) == (0, 0)
    with pytest.raises(ValueError, match="bandwidth\\(\\): 'x' must have at least 2 dimensions, got 1"):
        LA.bandwidth(as_sd(np.ones(3)))
    with pytest.raises(TypeError, match="for a sparse array use sd.sparse.spbandwidth"):
        LA.bandwidth(sd.sparse.csr_array(as_sd(np.eye(2))))


@pytest.mark.parametrize("dt", DTYPES)
def test_bandwidth_is_spbandwidth_and_the_triangular_check(dt):
    rng = np.random.default_rng(4)
    for shape, lo, up in (((5, 5), 0, 2), ((5, 5), 3, 0), ((6, 3), 1, 1), ((3, 6), 0, 0), ((4, 4), 3, 3)):
        x = banded(rng, shape, lo, up, dt)
        A = sd.sparse.csr_array(as_sd(x))
        r = LA.bandwidth(as_sd(x))
        assert r == sd.sparse.spbandwidth(A)
        assert (r.upper == 0, r.lower == 0) == sd.sparse.is_sptriangular(A)


# ------------------------------------------------------------ kron of 1-D --


@pytest.mark.parametrize("dts", [("float64", "float64"), ("int64", "complex128"), ("bool", "bool")])
def test_kron_of_vectors_is_npkron(dts):
    rng = np.random.default_rng(5)
    x, y, m = random_values(rng, (3,), dts[0]), random_values(rng, (4,), dts[1]), random_values(rng, (2, 2), dts[1])
    ct = compute_dtype(*dts)
    assert same_bits(LA.kron(as_sd(x), as_sd(y)), np.kron(x.astype(ct), y.astype(ct)))  # (12,)
    assert same_bits(LA.kron(as_sd(x), as_sd(m)), np.kron(x.astype(ct), m.astype(ct)))  # (2, 6)
    assert same_bits(LA.kron(as_sd(m), as_sd(y)), np.kron(m.astype(ct), y.astype(ct)))  # (2, 8)
    # a 1-D operand has no batch axes: it meets every matrix of a batch
    b = random_values(rng, (5, 2, 2), dts[1])
    r = npy(LA.kron(as_sd(x), as_sd(b)))
    assert r.shape == (5, 2, 6)
    assert all(same_bits(r[i], np.kron(x.astype(ct), b[i].astype(ct))) for i in range(5))
    assert same_bits(LA.kron(as_sd(x[::-2]), as_sd(y)), np.kron(x[::-2].astype(ct), y.astype(ct)))


def test_kron_of_vectors_is_sparse_krons_toarray():
    rng = np.random.default_rng(6)
    x, y = rng.standard_normal(4), rng.standard_normal(3)
    x[1] = 0
    assert same_bits(LA.kron(as_sd(x), as_sd(y)), sd.sparse.kron(as_sd(x), as_sd(y)).toarray())


# ------------------------------------------------------- lu_solve's trans --


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("trans", ["N", "T", "H"])
def test_lu_solve_trans(dt, trans):
    from .conftest import well_conditioned

    rng = np.random.default_rng(7)
    a = well_conditioned(rng, (3,), 5, dt)
    b = random_values(rng, (3, 5, 2), dt)
    x = npy(LA.lu_solve(*LA.lu(as_sd(a)), as_sd(b), trans=trans))
    op = {"N": a, "T": np.swapaxes(a, -1, -2), "H": np.swapaxes(a, -1, -2).conj()}[trans].astype(x.dtype)
    assert residual_ok(op, x, b.astype(x.dtype))
    if trans == "N":
        assert same_bits(x, LA.solve(as_sd(a), as_sd(b)))
    # the same as splu(A).solve(b, trans=) up to rounding (one matrix: splu takes no batch)
    want = npy(sd.sparse.splu(sd.sparse.csc_array(as_sd(a[0]))).solve(as_sd(b[0]), trans=trans))
    assert np.allclose(x[0], want, rtol=1e-12, atol=1e-12)


def test_lu_solve_trans_options():
    a = as_sd(np.array([[2.0, 1.0], [1.0, 3.0]]))
    lu = LA.lu(a)
    b = as_sd(np.ones(2))
    # real factors: 'H' is 'T'
    assert same_bits(LA.lu_solve(*lu, b, trans="H"), LA.lu_solve(*lu, b, trans="T"))
    # a complex b with real factors: real and imaginary parts as the real solves
    zc = npy(LA.lu_solve(*lu, as_sd(np.array([1 + 2j, 3 - 1j])), trans="T"))
    assert np.allclose(zc, npy(LA.lu_solve(*lu, as_sd(np.array([1.0, 3.0])), trans="T"))
                       + 1j * npy(LA.lu_solve(*lu, as_sd(np.array([2.0, -1.0])), trans="T")))  # fmt: skip
    with pytest.raises(ValueError, match="lu_solve\\(\\): 'trans' must be 'N', 'T' or 'H', not 'C'"):
        LA.lu_solve(*lu, b, trans="C")
    with pytest.raises(TypeError, match="lu_solve\\(\\): 'trans' must be 'N', 'T' or 'H', not int"):
        LA.lu_solve(*lu, b, trans=1)
    with pytest.raises(TypeError, match="'trans' must be 'N', 'T' or 'H', not NoneType"):
        LA.lu_solve(*lu, b, trans=None)
    with pytest.raises(LA.LinAlgError):  # a zero on u's diagonal: checked for every trans
        LA.lu_solve(lu[0], as_sd(np.zeros((2, 2))), lu[2], b, trans="T")


# ------------------------------------------------------ threads and batch --


def test_same_bits_for_any_thread_count():
    rng = np.random.default_rng(8)
    a, b = as_sd(rng.standard_normal((3000, 3, 3))), as_sd(rng.standard_normal((3000, 4, 4)))
    lu = LA.lu(a)
    rhs = as_sd(rng.standard_normal((3000, 3, 2)))
    calls = {
        "kronsum": lambda: LA.kronsum(a, b),
        "block_diag": lambda: LA.block_diag([a, b, as_sd(np.ones(5))]),
        "bandwidth": lambda: LA.bandwidth(as_sd(np.triu(npy(a)))).upper,
        "kron 1-D": lambda: LA.kron(as_sd(np.arange(3.0)), b),
        "lu_solve T": lambda: LA.lu_solve(*lu, rhs, trans="T"),
    }
    before = sd.get_num_threads()
    try:
        out = {}
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            for name, f in calls.items():
                r = np.asarray(f()).tobytes()
                assert out.setdefault(name, r) == r, (name, t)
    finally:
        sd.set_num_threads(before)


# ------------------------------------------- the sparse pairs that are methods --


@pytest.mark.parametrize("name", ["trace", "diagonal"])
def test_trace_and_diagonal_of_a_sparse_array_name_the_method(name):
    S = sd.sparse.csr_array(as_sd(np.array([[1.0, 2.0], [3.0, 4.0]])))
    with pytest.raises(TypeError) as e:
        getattr(LA, name)(S)
    assert str(e.value) == f"{name}(): 'x' must be an sd.ndarray, not csr_array; for a sparse array use A.{name}()"
    # and the method is there, with the dense function's value
    assert same_bits(np.asarray(getattr(S, name)()), np.asarray(getattr(LA, name)(S.toarray())))
