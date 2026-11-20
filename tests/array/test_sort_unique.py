"""sort, argsort and unique_*."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust as sd

from .conftest import NP_DTYPE, assert_same

REAL = [sd.bool, sd.int64, sd.float64]


def sample(dtype, shape=(4, 37), seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(-5, 6, shape)  # many ties
    if dtype is sd.bool:
        return n > 0
    if dtype is sd.int64:
        return n
    f = n * 0.5
    f[rng.random(shape) < 0.1] = np.nan
    f[rng.random(shape) < 0.05] = -0.0
    return f


def reference_argsort(lane, descending):
    """Stable, NaN the largest value (last ascending, first descending), ties in original order."""

    def key(i):
        v = lane[i]
        nan = isinstance(v, float) and math.isnan(v)
        if descending:
            return (not nan, 0 if nan else -v)
        return (nan, 0 if nan else v)

    return sorted(range(len(lane)), key=key)


@pytest.mark.parametrize("dtype", REAL, ids=str)
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("descending", [False, True])
def test_stable_argsort_and_sort(dtype, axis, descending):
    n = sample(dtype)
    x = sd.asarray(n)
    got = np.asarray(sd.argsort(x, axis=axis, descending=descending))
    moved = np.moveaxis(n, axis, -1)
    want = np.array([reference_argsort(lane.tolist(), descending) for lane in moved.reshape(-1, moved.shape[-1])])
    want = np.moveaxis(want.reshape(moved.shape), -1, axis)
    assert np.array_equal(got, want)
    assert_same(sd.sort(x, axis=axis, descending=descending), np.take_along_axis(n, want, axis=axis))
    if not descending:  # numpy's stable sort is the same ordering (-0.0 and 0.0 tie and keep their order)
        assert_same(sd.sort(x, axis=axis), np.sort(n, axis=axis, stable=True))
        assert np.array_equal(got, np.argsort(n, axis=axis, stable=True))


@pytest.mark.parametrize("dtype", REAL, ids=str)
@pytest.mark.parametrize("descending", [False, True])
def test_unstable_sort_gives_sorted_values(dtype, descending):
    n = sample(dtype, (3, 2000), seed=1)
    x = sd.asarray(n)
    stable = np.asarray(sd.sort(x, descending=descending))
    # Equal values, but tied -0.0/+0.0 may come out in either order: compare as numbers.
    assert np.array_equal(np.asarray(sd.sort(x, descending=descending, stable=False)), stable, equal_nan=True)
    idx = np.asarray(sd.argsort(x, descending=descending, stable=False))
    assert np.array_equal(np.take_along_axis(n, idx, axis=-1), stable, equal_nan=True)
    assert all(sorted(r) == list(range(n.shape[1])) for r in idx.tolist())  # a permutation


@pytest.mark.parametrize("pattern", ["sorted", "reversed", "equal", "sawtooth", "random"])
@pytest.mark.parametrize("stable", [True, False])
def test_adversarial_inputs(pattern, stable):
    n = {
        "sorted": np.arange(5000.0),
        "reversed": np.arange(5000.0)[::-1].copy(),
        "equal": np.ones(5000),
        "sawtooth": np.tile(np.arange(50.0), 100),
        "random": np.random.default_rng(2).standard_normal(5000),
    }[pattern]
    assert_same(sd.sort(sd.asarray(n), stable=stable), np.sort(n))


def test_layouts_and_rules():
    n = np.random.default_rng(3).standard_normal((6, 7))
    for view in (n.T, n[::-1, ::2], np.asfortranarray(n)):
        assert_same(sd.sort(sd.asarray(view), axis=0), np.sort(view, axis=0))
    with pytest.raises(TypeError):
        sd.sort(sd.asarray([1j]))  # docs/api/array.md §1.6
    with pytest.raises(TypeError):
        sd.sort(sd.asarray([1.0]), axis=None)  # an int axis only
    with pytest.raises(ValueError):
        sd.sort(sd.asarray([1.0]), axis=1)
    with pytest.raises(TypeError):
        sd.sort(sd.asarray([1.0]), -1)  # keyword-only
    assert sd.sort(sd.zeros((3, 0))).shape == (3, 0)
    assert sd.argsort(sd.zeros(0)).dtype is sd.int64


def test_descending_keeps_tie_order_and_puts_nan_first():
    # NaN is the largest value (docs/api/array.md §8.2), as in the frame's sort_values.
    x = sd.asarray([1.0, math.nan, 3.0, 1.0, 3.0, math.nan])
    assert sd.argsort(x, descending=True).tolist() == [1, 5, 2, 4, 0, 3]
    assert sd.argsort(x).tolist() == [0, 3, 2, 4, 1, 5]
    got = sd.sort(x, descending=True).tolist()
    assert all(math.isnan(v) for v in got[:2]) and got[2:] == [3.0, 3.0, 1.0, 1.0]
    for stable in (True, False):
        idx = sd.argsort(x, descending=True, stable=stable).tolist()
        assert sorted(idx[:2]) == [1, 5]  # the NaNs come first either way
    inf = sd.asarray([math.inf, math.nan, -math.inf])
    assert sd.argsort(inf, descending=True).tolist() == [1, 0, 2]  # NaN is above +inf
    assert sd.argsort(inf).tolist() == [2, 0, 1]


def test_descending_matches_the_frame():
    # One rule for the two areas (DESIGN.md §9.1 #9).
    from stardust import frame as F

    v = [2.0, math.nan, -0.0, 0.0, 2.0, math.nan, -1.0]
    s = F.Series(v)
    for desc in (False, True):
        got = sd.argsort(sd.asarray(v), descending=desc).tolist()
        want = s.sort_values(descending=desc).index.tolist()
        assert got == want


# ------------------------------------------------------------------ unique


def _is_nan(v):
    if isinstance(v, complex):
        return math.isnan(v.real) or math.isnan(v.imag)
    return isinstance(v, float) and math.isnan(v)


def reference_unique(flat):
    """Sorted distinct values; all NaNs one value, the last; first occurrences."""

    def key(i):
        v = flat[i]
        if _is_nan(v):
            return (True, 0, 0)
        if isinstance(v, complex):
            return (False, v.real, v.imag)
        return (False, v, 0)

    order = sorted(range(len(flat)), key=key)
    groups = []
    for i in order:
        v = flat[i]
        if groups and (flat[groups[-1][0]] == v or (_is_nan(v) and _is_nan(flat[groups[-1][0]]))):
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


@pytest.mark.parametrize("dtype", [sd.bool, sd.int64, sd.float64, sd.complex128], ids=str)
def test_unique_all(dtype):
    n = sample(sd.float64 if dtype is sd.complex128 else dtype, (5, 7))
    if dtype is sd.complex128:
        n = n + 1j * sample(sd.float64, (5, 7), seed=4)
    x = sd.asarray(n)
    r = sd.unique_all(x)
    assert r._fields == ("values", "indices", "inverse_indices", "counts")
    flat = n.ravel().tolist()
    groups = reference_unique(flat)
    assert_same(r.values, np.array([flat[g[0]] for g in groups], dtype=NP_DTYPE[dtype]))
    assert r.indices.tolist() == [g[0] for g in groups]
    assert r.counts.tolist() == [len(g) for g in groups]
    assert r.inverse_indices.shape == n.shape  # Array API: the shape of x
    rebuilt = np.asarray(r.values)[np.asarray(r.inverse_indices)]
    assert np.array_equal(rebuilt, n, equal_nan=True)
    assert_same(sd.unique_values(x), np.asarray(r.values))
    assert sd.unique_counts(x)._fields == ("values", "counts")
    assert sd.unique_inverse(x)._fields == ("values", "inverse_indices")


def test_unique_nan_and_signed_zero():
    # All NaNs are one value, sorted last; -0.0 and 0.0 are one value. The
    # value shown is the first occurrence (docs/api/array.md §8.3).
    x = sd.asarray([math.nan, 0.0, -0.0, math.nan, 1.0])
    r = sd.unique_all(x)
    v = r.values.tolist()
    assert len(v) == 3 and v[:2] == [0.0, 1.0] and math.copysign(1, v[0]) == 1 and math.isnan(v[2])
    assert r.counts.tolist() == [2, 1, 2]
    assert r.indices.tolist() == [1, 4, 0]
    assert r.inverse_indices.tolist() == [2, 0, 0, 2, 1]
    u = sd.unique_values(sd.asarray([-0.0, 0.0, math.nan]))
    assert math.copysign(1, u.tolist()[0]) == -1 and len(u.tolist()) == 2
    assert sd.unique_values(sd.zeros(0)).shape == (0,)
    assert sd.unique_values(sd.asarray([math.nan] * 3)).shape == (1,)


def test_unique_shows_the_first_nan_bits_and_matches_the_frame():
    a = np.array([np.nan, 1.0, np.nan, 1.0])
    a.view(np.int64)[0] |= 7  # a NaN with a payload, first
    a.view(np.int64)[2] ^= np.int64(-(2**63))  # a negative NaN
    for fn in (
        sd.unique_values,
        lambda x: sd.unique_all(x).values,
        lambda x: sd.unique_counts(x).values,
        lambda x: sd.unique_inverse(x).values,
    ):
        got = np.asarray(fn(sd.asarray(a)))
        assert got.view(np.int64).tolist() == [a.view(np.int64)[1], a.view(np.int64)[0]]
    from stardust import frame as F

    v = [np.nan, 0.0, -0.0, 2.0, np.nan, -1.0]
    want = F.Series(v).unique_values().tolist()
    got = sd.unique_values(sd.asarray(v)).tolist()
    assert len(got) == len(want) and all(
        (math.isnan(g) and math.isnan(w)) or (g == w and math.copysign(1, g) == math.copysign(1, w))
        for g, w in zip(got, want)
    )


def test_unique_complex_nan_is_one_value():
    # Any complex number with a NaN part is the NaN, one value after the
    # others (numpy 2 collapses them too; it shows its smallest NaN, we the first).
    nan = math.nan
    c = sd.asarray([complex(nan, 2), complex(1, nan), 1 + 0j, complex(nan, nan), 1j, 1 + 0j])
    r = sd.unique_all(c)
    assert r.values.tolist()[:2] == [1j, 1 + 0j] and repr(r.values.tolist()[2]) == repr(complex(nan, 2))
    assert r.indices.tolist() == [4, 2, 0] and r.counts.tolist() == [1, 2, 3]
    assert r.inverse_indices.tolist() == [2, 2, 1, 2, 0, 1]
    _, w_inverse, w_counts = np.unique(np.asarray(c), return_inverse=True, return_counts=True, equal_nan=True)
    assert r.counts.tolist() == w_counts.tolist()
    assert r.inverse_indices.tolist() == w_inverse.tolist()


@given(
    hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=1, max_dims=2, max_side=8),
        elements=st.sampled_from([np.nan, -np.nan, 0.0, -0.0, 1.5, -2.0, np.inf, -np.inf]),
    )
)
def test_unique_matches_numpy_equal_nan_on_floats(n):
    # numpy 2's np.unique defaults to equal_nan=True (np.unique_all does not:
    # it follows the Array API): the same groups, counts and first indices.
    # Which of -0.0 / 0.0 numpy shows depends on its sort, so values are
    # compared as numbers.
    r = sd.unique_all(sd.asarray(n))
    values, index, inverse, counts = np.unique(
        n, return_index=True, return_inverse=True, return_counts=True, equal_nan=True
    )
    assert np.array_equal(np.asarray(r.values), values, equal_nan=True)
    assert r.counts.tolist() == counts.tolist()
    assert r.indices.tolist() == index.tolist()
    assert np.array_equal(np.asarray(r.inverse_indices), inverse.reshape(n.shape))
    assert np.array_equal(np.asarray(sd.unique_values(sd.asarray(n))), values, equal_nan=True)


@given(hnp.arrays(np.int64, hnp.array_shapes(min_dims=1, max_dims=3, max_side=6), elements=st.integers(-4, 4)))
def test_unique_matches_numpy_on_integers(n):
    r = sd.unique_all(sd.asarray(n))
    w = np.unique_all(n)
    assert r.values.tolist() == w.values.tolist()
    assert r.counts.tolist() == w.counts.tolist()
    assert r.indices.tolist() == w.indices.tolist()
    assert np.array_equal(np.asarray(r.inverse_indices), w.inverse_indices.reshape(n.shape))


# ---------------------------------------------------- the NEON sorting path


@pytest.mark.parametrize("n", [31, 32, 33, 47, 48, 64, 100, 1000, 4097])
@pytest.mark.parametrize("kind", ["int", "float", "float-ties", "extremes"])
def test_simd_sort_sizes_and_values(n, kind):
    # Sizes around the 16-element blocks and the padding; values that equal the
    # padding (inf, INT64_MAX) and many ties.
    rng = np.random.default_rng(n)
    v = {
        "int": rng.integers(-50, 50, n),
        "float": rng.standard_normal(n),
        "float-ties": rng.integers(-3, 3, n) * 0.5 + 0.25,
        "extremes": np.where(rng.random(n) < 0.2, 2**63 - 1, rng.integers(-(2**63), 2**63 - 1, n)),
    }[kind]
    if kind == "float":
        v[rng.random(n) < 0.1] = np.inf
        v[rng.random(n) < 0.05] = np.nan
    for stable in (True, False):
        assert_same(sd.sort(sd.asarray(v), stable=stable), np.sort(v, stable=True))
        got = np.asarray(sd.sort(sd.asarray(v), descending=True, stable=stable))
        assert np.array_equal(got, np.sort(v)[::-1], equal_nan=True)  # NaN first: the reverse of ascending
        if got.dtype.kind == "f":
            k = int(np.isnan(v).sum())
            assert np.isnan(got[:k]).all() and not np.isnan(got[k:]).any()
    assert_same(sd.unique_values(sd.asarray(v)), np.asarray(sd.unique_all(sd.asarray(v)).values))


def test_stable_float_sort_with_both_zero_signs_keeps_order():
    # -0.0 == +0.0: here the order of equal values is visible, so the stable
    # sort must not take the (unstable) NEON path.
    v = np.array([0.0, -0.0] * 40 + [1.0, -1.0] * 10)
    got = np.asarray(sd.sort(sd.asarray(v)))
    assert np.array_equal(got.view(np.int64), np.sort(v, stable=True).view(np.int64))
    u = sd.unique_values(sd.asarray(v))
    assert u.tolist() == [-1.0, 0.0, 1.0] and math.copysign(1, u.tolist()[1]) == 1  # first occurrence: +0.0


@given(st.lists(st.integers(-(2**63), 2**63 - 1), min_size=0, max_size=300))
def test_simd_sort_random_int64(values):
    n = np.array(values, dtype=np.int64)
    assert_same(sd.sort(sd.asarray(n)), np.sort(n))


def test_many_lanes_short_and_long():
    # Short lanes reuse per-thread scratch, long ones allocate
    # (docs/design/array.md §8.7): 4096 Keyed pairs (64 KB) is the last that fits.
    rng = np.random.default_rng(3)
    for shape in [(20_000, 8), (3, 4096), (3, 4097), (2, 50_000)]:
        n = rng.integers(-5, 5, shape).astype(np.float64)
        n[::7, ::3] = np.nan
        x = sd.asarray(n)
        assert_same(sd.sort(x, axis=1), np.sort(n, axis=1, kind="stable"))
        assert_same(sd.argsort(x, axis=1), np.argsort(n, axis=1, kind="stable"))
        assert_same(sd.sort(x, axis=1, descending=True), np.sort(n, axis=1)[:, ::-1])  # NaN first


@pytest.mark.parametrize("n", [127, 128, 1023, 1024, 5000, 70_000])
@pytest.mark.parametrize("kind", ["wide", "narrow", "extremes", "equal", "one-digit"])
def test_int64_radix_sort_matches_stable_numpy(n, kind):
    # int64 lanes from 128 (pairs) / 1024 (values) elements are radix
    # sorted, relative to the smallest key (docs/design/array.md §8.4).
    rng = np.random.default_rng(n)
    lo, hi = {
        "wide": (-(2**63), 2**63 - 1),
        "narrow": (-1000, 1000),
        "extremes": (-(2**63), 2**63 - 1),
        "equal": (7, 8),
        "one-digit": (-100, 100),
    }[kind]
    a = rng.integers(lo, hi, n, endpoint=True)
    if kind == "extremes":
        a[::3] = -(2**63)
        a[1::5] = 2**63 - 1
    x = sd.asarray(a)
    assert_same(sd.sort(x), np.sort(a, kind="stable"))
    assert_same(sd.argsort(x), np.argsort(a, kind="stable"))
    assert_same(sd.sort(x, descending=True), np.sort(a)[::-1])
    want_desc = np.argsort(-a.astype(object), kind="stable").astype(np.int64)  # ties keep their order
    assert_same(sd.argsort(x, descending=True), want_desc)
    two = sd.asarray(np.stack([a, a[::-1]]))
    assert_same(sd.argsort(two, axis=1), np.argsort(np.stack([a, a[::-1]]), axis=1, kind="stable"))


@pytest.mark.parametrize("n", [2 * 65536 - 1, 2 * 65536, 3 * 65536 + 17, 600_001])
def test_long_lane_parallel_sort_is_the_stable_result(n):
    # Pieces of 65536 sorted on the pool, then merged (docs/design/array.md §8.5).
    rng = np.random.default_rng(n)
    f = rng.integers(-50, 50, n).astype(np.float64) * 0.5  # many ties
    f[::101] = np.nan
    f[::7] = 0.0
    f[3::7] = -0.0
    i = rng.integers(-(2**40), 2**40, n)
    for a in (f, i):
        x = sd.asarray(a)
        assert_same(sd.sort(x), np.sort(a, kind="stable"))
        assert_same(sd.argsort(x), np.argsort(a, kind="stable"))
        if a is i:
            assert_same(sd.argsort(x, descending=True), np.argsort(-a, kind="stable"))
        assert np.array_equal(np.asarray(sd.sort(x, stable=False)), np.sort(a), equal_nan=True)
    want = np.unique_counts(i[: n // 2] % 1000)
    got = sd.unique_counts(sd.asarray(i[: n // 2] % 1000))
    assert_same(got.values, want.values)
    assert_same(got.counts, want.counts)


@pytest.mark.parametrize("func", [sd.sort, sd.argsort])
def test_options_are_checked(func):
    # DESIGN.md §10.3 (the frame's message format) and §10.1 #13 (numpy integers as axis)
    x = sd.asarray([[3, 1], [2, 4]])
    name = func.__name__
    for option in ("descending", "stable"):
        for bad, shown in ((1, "int"), (None, "NoneType"), (np.True_, "numpy.bool")):
            with pytest.raises(TypeError, match=rf"^{name}\(\): '{option}' must be True or False, not {shown}$"):
                func(x, **{option: bad})
    with pytest.raises(TypeError, match=rf"^{name}\(\): 'axis' must be an int, not NoneType$"):
        func(x, axis=None)
    assert func(x, axis=np.int64(0)).tolist() == func(x, axis=0).tolist()


def test_unique_counts_matches_unique_all():
    rng = np.random.default_rng(3)
    for a in [
        np.array([0.0, -0.0, 1, np.nan, -0.0, np.nan, 1]),
        rng.integers(0, 1000, 300_000),
        rng.standard_normal(5000).round(1),
        np.array([1 + 1j, np.nan, 1 + 1j]),
        np.array([], float),
    ]:
        s = sd.asarray(a)
        c, r = sd.unique_counts(s), sd.unique_all(s)
        assert np.asarray(c.values).tobytes() == np.asarray(r.values).tobytes()
        assert_same(c.counts, np.asarray(r.counts))
        _, ni, ninv, nc = np.unique(a, return_index=True, return_inverse=True, return_counts=True, equal_nan=True)
        assert_same(c.counts, nc)
        assert_same(r.indices, ni)
        assert np.array_equal(np.asarray(r.inverse_indices).ravel(), ninv.ravel())


def test_float_sort_keeps_the_stable_order_of_signed_zeros():
    rng = np.random.default_rng(6)
    for n in (100, 1023, 1024, 5000, 300_000):
        a = rng.choice([0.0, -0.0, 1.0, -1.0, np.nan, np.inf, -np.inf, 2.5], n)
        for desc in (False, True):
            s = sd.asarray(a)
            order = np.asarray(sd.argsort(s, descending=desc))  # the stable order
            assert np.asarray(sd.sort(s, descending=desc)).tobytes() == a[order].tobytes()
            assert np.asarray(sd.sort(s, descending=desc, stable=False)).tobytes() == a[order].tobytes()
        assert (
            np.asarray(sd.unique_values(sd.asarray(a))).tobytes()
            == np.asarray(sd.unique_all(sd.asarray(a)).values).tobytes()
        )


def test_result_object_arrays_are_read_only():
    x = sd.asarray([3, 1, 3, 2])
    for r in (sd.unique_all(x), sd.unique_counts(x), sd.unique_inverse(x)):
        for a in r:
            with pytest.raises(ValueError, match="^assignment destination is read-only$"):
                a[0] = 0
            with pytest.raises(ValueError, match="read-only"):
                a[:1][0] = 0  # views of a field stay read-only
            assert np.asarray(a).flags.writeable is False
    edges = sd.asarray([0.0, 1.0, 2.0, 3.0])
    h = sd.histogram(x, bins=edges)
    for a in h:
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 0
    edges[0] = -1.0  # the caller's own array stays writable
    assert h.bin_edges[0] == -1.0
    v = sd.unique_values(x)  # a plain array result is not a result object
    v[0] = 9
