"""The Array API 2024 functions beyond the core ones, plus bincount and histogram, against
numpy.

Exact unless noted: the elementwise functions are numpy's scalar formulas,
the shape and search functions move or compare values only, and histogram
replays numpy's block sums. tensordot goes through matmul (BLAS), so it is
compared with a tolerance.
"""

import gc
import warnings

import numpy as np
import pytest

import stardust as sd

from .conftest import assert_same

RNG = np.random.default_rng(0)
X3 = RNG.standard_normal((3, 4, 5))
SPECIAL = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e308, -1e308, 5e-324, -5e-324, 2.5, -7.25])


# --------------------------------------------------------------- elementwise


@pytest.mark.parametrize("name", ["copysign", "hypot", "logaddexp", "nextafter"])
def test_binary_elementwise_bits(name):
    a = np.concatenate([RNG.standard_normal(500) * 30, SPECIAL])
    b = np.concatenate([RNG.standard_normal(500) * 30, SPECIAL[::-1]])
    aa, bb = np.meshgrid(SPECIAL, SPECIAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, y in ((a, b), (aa.ravel(), bb.ravel())):
            assert_same(getattr(sd, name)(sd.asarray(x), sd.asarray(y)), getattr(np, name)(x, y))
        assert_same(getattr(sd, name)(sd.asarray([1, 2, 3]), 2), getattr(np, name)(np.array([1.0, 2, 3]), 2.0))
    assert type(getattr(sd, name)(1.0, 2.0)) is float
    with pytest.raises(TypeError):
        getattr(sd, name)(sd.asarray([1j]), 1.0)


def test_reciprocal_and_signbit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = np.concatenate([RNG.standard_normal(500) * 30, SPECIAL])
        assert_same(sd.reciprocal(sd.asarray(x)), np.reciprocal(x))
        z = RNG.standard_normal(2000) * 10 + 1j * RNG.standard_normal(2000) * 10
        z = np.concatenate(
            [z, [0j, complex(np.inf, 1), complex(1, np.inf), complex(np.nan, 0), complex(1e-310, 1e300)]]
        )
        assert_same(sd.reciprocal(sd.asarray(z)), np.reciprocal(z))
    assert_same(sd.reciprocal(sd.asarray([2, 4])), np.array([0.5, 0.25]))  # integers in float64 (Array API)
    assert_same(sd.signbit(sd.asarray(x)), np.signbit(x))
    assert_same(sd.signbit(sd.asarray([-1, 0, 3])), np.array([True, False, False]))
    with pytest.raises(TypeError):
        sd.signbit(sd.asarray([1j]))


# ----------------------------------------------------------------- creation


@pytest.mark.parametrize("k", range(-6, 7))
def test_tril_triu(k):
    m = RNG.standard_normal((2, 4, 5))
    assert_same(sd.tril(sd.asarray(m), k=k), np.tril(m, k))
    assert_same(sd.triu(sd.asarray(m), k=k), np.triu(m, k))
    b = RNG.random((3, 3)) > 0.5
    assert_same(sd.triu(sd.asarray(b), k=k), np.triu(b, k))


def test_tril_triu_extreme_k():
    m = np.arange(12.0).reshape(3, 4)
    for k in (2**63 - 1, -(2**63) + 10, 5, -4):
        assert_same(sd.tril(sd.asarray(m), k=k), np.tril(m, k))
        assert_same(sd.triu(sd.asarray(m), k=k), np.triu(m, k))


def test_tril_errors_and_copy():
    with pytest.raises(ValueError, match=r"tril\(\): 'x' must have at least 2 dimensions"):
        sd.tril(sd.asarray([1.0, 2.0]))
    x = sd.asarray(np.ones((2, 2)))
    t = sd.tril(x)
    t[0, 0] = 5.0
    assert x[0, 0] == 1.0


def test_meshgrid():
    a, b, c = np.array([1, 2, 3]), np.array([4.0, 5.0]), np.array([True, False, True, True])
    for indexing in ("xy", "ij"):
        got = sd.meshgrid(sd.asarray(a), sd.asarray(b), sd.asarray(c), indexing=indexing)
        want = np.meshgrid(a, b, c, indexing=indexing)
        assert isinstance(got, tuple) and len(got) == 3
        for g, w in zip(got, want):
            assert_same(g, w)
    assert_same(sd.meshgrid(sd.asarray(a))[0], np.meshgrid(a)[0])
    assert sd.meshgrid() == ()
    with pytest.raises(ValueError, match="'indexing' must be 'xy' or 'ij'"):
        sd.meshgrid(sd.asarray(a), indexing="xx")
    with pytest.raises(ValueError, match="every array must be 1-D"):
        sd.meshgrid(sd.zeros((2, 2)))


# ------------------------------------------------------------- manipulation


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2)])
def test_flip_is_a_view(axis):
    a = sd.asarray(X3)
    f = sd.flip(a, axis=axis)
    assert_same(f, np.flip(X3, axis))
    assert f.base is not None
    f[0, 0, 0] = 123.0
    assert 123.0 in np.asarray(a)


@pytest.mark.parametrize(
    "shift,axis", [(7, None), (-3, None), (2, 0), (-1, (1, 2)), ((1, -2), (0, 2)), (100, 1), (0, 1)]
)
def test_roll(shift, axis):
    assert_same(sd.roll(sd.asarray(X3), shift, axis=axis), np.roll(X3, shift, axis))
    assert_same(sd.roll(sd.asarray(X3), shift=shift, axis=axis), np.roll(X3, shift, axis))


def test_roll_x_is_positional_only():
    assert sd.roll.__nb_signature__[0][0] == "def roll(x, /, shift, *, axis=None)"
    with pytest.raises(TypeError):
        sd.roll(x=sd.asarray([1.0]), shift=1)


def test_roll_errors():
    with pytest.raises(TypeError, match="'shift' must be an int when 'axis' is None"):
        sd.roll(sd.asarray(X3), (1, 2))
    with pytest.raises(ValueError, match="same length"):
        sd.roll(sd.asarray(X3), (1, 2), axis=(0,))
    with pytest.raises(TypeError, match=r"^roll\(\): 'shift' must be an int or a tuple or list of ints, not float$"):
        sd.roll(sd.asarray(X3), 1.5, axis=0)


def test_roll_takes_lists_like_axis():
    # shift pairs with axis, which takes lists (docs/api/array.md §1.3, §4.4)
    assert_same(sd.roll(sd.asarray(X3), [1, -2], axis=[0, 2]), np.roll(X3, (1, -2), (0, 2)))


@pytest.mark.parametrize("axis", [None, 0, (0, 1)])
def test_roll_result_owns_its_memory(axis):
    # a new array (docs/api/array.md §4.4, §10.1), also when axis=None goes through the flattened array
    r = sd.roll(sd.asarray(X3), 1, axis=axis)
    assert r.base is None
    assert_same(r, np.roll(X3, 1, axis))
    assert r._flags()["c_contiguous"]


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda: sd.roll(sd.asarray(X3), 2**70), "shift"),
        (lambda: sd.roll(sd.asarray(X3), (2**70,), axis=(0,)), "shift"),
        (lambda: sd.repeat(sd.asarray(X3), 2**70), "repeats"),
        (lambda: sd.repeat(sd.asarray(X3), -(2**70)), "repeats"),
        (lambda: sd.tile(sd.asarray(X3), 2**70), "repetitions"),
        (lambda: sd.tile(sd.asarray(X3), (1, 2**70)), "repetitions"),
    ],
)
def test_int_arguments_outside_int64_overflow(call, name):
    # API.md §2, docs/api/array.md §1.3: like shape and axis
    with pytest.raises(OverflowError, match=rf"^\w+\(\): '{name}' is out of range$"):
        call()


def test_int_only_axis():
    # unstack and repeat take an int axis only (docs/api/array.md §1.3)
    for f in (lambda ax: sd.unstack(sd.asarray(X3), axis=ax), lambda ax: sd.repeat(sd.asarray(X3), 2, axis=ax)):
        for bad in [(0,), [0]]:
            with pytest.raises(TypeError, match=r"'axis' must be an int, not (tuple|list)$"):
                f(bad)


def test_repeat():
    assert_same(sd.repeat(sd.asarray(X3), 2), np.repeat(X3, 2))
    assert_same(sd.repeat(sd.asarray(X3), 3, axis=1), np.repeat(X3, 3, axis=1))
    assert_same(sd.repeat(sd.asarray(X3), sd.asarray([2, 0, 1, 3]), axis=1), np.repeat(X3, [2, 0, 1, 3], axis=1))
    assert_same(sd.repeat(sd.asarray(X3), sd.asarray([2]), axis=2), np.repeat(X3, [2], axis=2))
    with pytest.raises(ValueError, match="non-negative"):
        sd.repeat(sd.asarray([1, 2]), -1)
    with pytest.raises(ValueError, match="'repeats' has length 3 but the axis has length 2"):
        sd.repeat(sd.asarray([1, 2]), sd.asarray([1, 2, 3]))
    with pytest.raises(ValueError, match="array is too big"):
        sd.repeat(sd.asarray([1.0, 2.0, 3.0]), 2**62)
    with pytest.raises(TypeError, match="'repeats' must be an int or an int64 array"):
        sd.repeat(sd.asarray([1, 2]), 1.5)
    assert_same(sd.repeat(sd.asarray([1, 2]), np.int64(2)), np.repeat([1, 2], 2))


@pytest.mark.parametrize("reps", [(2,), 2, (2, 1, 3), (1, 1, 1), (2, 1, 1, 3), (0, 2), (3, 2, 1, 2, 1)])
def test_tile(reps):
    assert_same(sd.tile(sd.asarray(X3), reps), np.tile(X3, reps))
    assert_same(sd.tile(sd.asarray([True, False]), reps), np.tile([True, False], reps))


def test_tile_repetitions_are_like_shape():
    # an int or a tuple of ints; a list is a TypeError, as for shape (docs/api/array.md §1.3, §4.4)
    with pytest.raises(TypeError, match=r"^tile\(\): 'repetitions' must be an int or a tuple of ints, not list$"):
        sd.tile(sd.asarray(X3), [2])
    with pytest.raises(
        TypeError, match=r"^tile\(\): 'repetitions' must be an int or a tuple of ints, not a tuple holding float$"
    ):
        sd.tile(sd.asarray(X3), (2, 1.0))
    with pytest.raises(ValueError, match=r"^tile\(\): 'repetitions' must be non-negative$"):
        sd.tile(sd.asarray(X3), (2, -1))
    assert_same(sd.tile(sd.asarray(X3), (np.int64(2),)), np.tile(X3, 2))


def test_tile_many_dimensions():
    # 30-D: room for two interleaved axes per copy, so five repeated axes take three copies.
    x = np.arange(6).reshape((1,) * 28 + (2, 3))
    reps = (1,) * 25 + (2,) * 5
    assert_same(sd.tile(sd.asarray(x), reps), np.tile(x, reps))
    # 32-D: no room at all, one block copy per repetition.
    y = np.arange(4).reshape((1,) * 30 + (2, 2))
    reps = (3,) + (1,) * 29 + (2, 1)
    assert_same(sd.tile(sd.asarray(y), reps), np.tile(y, reps))


def test_unstack():
    for axis in (0, 1, -1):
        got = sd.unstack(sd.asarray(X3), axis=axis)
        assert isinstance(got, tuple) and len(got) == X3.shape[axis]
        for i, g in enumerate(got):
            assert_same(g, np.take(X3, i, axis=axis))
            assert g.base is not None
    assert sd.unstack(sd.asarray([1, 2, 3])) == (1, 2, 3)  # no 0-d arrays: Python scalars


def test_diff():
    assert_same(sd.diff(sd.asarray(X3)), np.diff(X3))
    assert_same(sd.diff(sd.asarray(X3), axis=0, n=2), np.diff(X3, axis=0, n=2))
    assert_same(sd.diff(sd.asarray(X3), n=9), np.diff(X3, n=9))
    assert_same(
        sd.diff(sd.asarray(X3), prepend=0.5, append=sd.asarray(np.ones((3, 4, 2)))),
        np.diff(X3, prepend=0.5, append=np.ones((3, 4, 2))),
    )
    assert_same(sd.diff(sd.asarray([1, 4, 9, 16])), np.diff([1, 4, 9, 16]))
    assert_same(sd.diff(sd.asarray([True, False, False, True])), np.array([-1, 0, 1]))  # subtract rule: int64
    x = sd.asarray([1.0, 2.0])
    assert sd.diff(x, n=0) is x
    with pytest.raises(ValueError, match="'n' must be non-negative"):
        sd.diff(x, n=-1)


def test_diff_edge_errors_name_the_argument():
    x2 = sd.ones((2, 3))
    with pytest.raises(
        TypeError,
        match=r"^diff\(\): 'prepend' must be an sd.ndarray or a bool, int, float or complex, "
        r"not list$",
    ):
        sd.diff(sd.asarray([1.0, 2.0]), prepend=[0])
    with pytest.raises(
        TypeError,
        match=r"^diff\(\): 'append' must be an sd.ndarray or a bool, int, float or complex, "
        r"not str$",
    ):
        sd.diff(sd.asarray([1.0, 2.0]), append="a")
    with pytest.raises(ValueError, match=r"^diff\(\): 'prepend' must have as many dimensions as 'x' \(2\), not 1$"):
        sd.diff(x2, axis=1, prepend=sd.asarray([0.0]))
    with pytest.raises(
        ValueError,
        match=r"^diff\(\): 'append' must have the shape of 'x' \(2, 3\) except along axis 1, "
        r"not \(3, 1\)$",
    ):
        sd.diff(x2, axis=1, append=sd.ones((3, 1)))
    assert_same(
        sd.diff(x2, axis=1, prepend=sd.zeros((2, 1)), append=np.int64(4)),
        np.diff(np.ones((2, 3)), axis=1, prepend=np.zeros((2, 1)), append=4),
    )


# ---------------------------------------------------------- indexing, search


def test_take():
    idx = np.array([[0, 3], [-1, 2]])
    assert_same(sd.take(sd.asarray(X3), sd.asarray(idx), axis=1), np.take(X3, idx, axis=1))
    assert_same(sd.take(sd.asarray(X3), 2, axis=2), np.take(X3, 2, axis=2))
    assert sd.take(sd.asarray([5, 6, 7]), 1) == 6
    assert_same(sd.take(sd.asarray([5, 6, 7]), sd.asarray([2, 0])), np.array([7, 5]))
    with pytest.raises(IndexError):
        sd.take(sd.asarray([5, 6, 7]), 3)
    # take names itself like take_along_axis (API.md §2.1)
    with pytest.raises(IndexError, match=r"^take\(\): index 5 is out of bounds for axis 0 with size 3$"):
        sd.take(sd.asarray([5, 6, 7]), sd.asarray([0, 5]))
    with pytest.raises(ValueError, match="'axis' is required"):
        sd.take(sd.asarray(X3), 0)
    # one message for the argument, whatever is wrong (API.md §2.1)
    for bad, shown in [(sd.asarray([True, False]), "bool"), ([0], "list"), (sd.asarray([0.0]), "float64")]:
        with pytest.raises(TypeError, match=rf"^take\(\): 'indices' must be an int or an int64 array, not {shown}$"):
            sd.take(sd.asarray([5, 6]), bad)
    for bad, shown in [(sd.asarray([[True]]), "bool"), ([0], "list"), (0, "int")]:
        with pytest.raises(TypeError, match=rf"^take_along_axis\(\): 'indices' must be an int64 array, not {shown}$"):
            sd.take_along_axis(sd.asarray([[5, 6]]), bad, axis=1)


def test_take_along_axis():
    for axis in (0, 1, -1):
        ix = np.argsort(X3, axis=axis)
        assert_same(
            sd.take_along_axis(sd.asarray(X3), sd.asarray(ix), axis=axis), np.take_along_axis(X3, ix, axis=axis)
        )
    ix = np.argsort(X3, axis=1)[:, :2]
    assert_same(sd.take_along_axis(sd.asarray(X3), sd.asarray(ix), axis=1), np.take_along_axis(X3, ix, axis=1))
    with pytest.raises(ValueError, match="as many dimensions"):
        sd.take_along_axis(sd.asarray(X3), sd.asarray([0]), axis=0)


@pytest.mark.parametrize("shape", [(3_000_000,), (500, 3000), (40, 50, 600), (5, 1, 7)])
def test_nonzero_large_and_strided(shape):
    rng = np.random.default_rng(14)
    for p in (0.3, 0.0, 1.0):
        z = (rng.random(shape) < p).astype(float)
        for v in (z, z.T if z.ndim > 1 else z[::-1]):
            for g, w in zip(sd.nonzero(sd.asarray(v)), np.nonzero(v)):
                assert_same(g, w)


def test_take_along_axis_broadcast_and_bounds():
    rng = np.random.default_rng(15)
    x = rng.standard_normal((4, 5, 6))
    for axis in (0, 1, 2):
        others = [d for d in range(3) if d != axis]
        shapes = [list(x.shape)]
        for length, ones in ((3, []), (2, others[:1]), (1, others)):  # other lengths along the axis, 1s elsewhere
            sh = list(x.shape)
            sh[axis] = length
            for d in ones:
                sh[d] = 1
            shapes.append(sh)
        for ishape in shapes:
            n = x.shape[axis]
            idx = rng.integers(-n, n, ishape)
            assert_same(
                sd.take_along_axis(sd.asarray(x), sd.asarray(idx), axis=axis), np.take_along_axis(x, idx, axis=axis)
            )
    assert_same(
        sd.take_along_axis(sd.asarray(x).mT, sd.asarray(np.zeros((4, 6, 2), int)), axis=2),
        np.take_along_axis(x.transpose(0, 2, 1), np.zeros((4, 6, 2), int), axis=2),
    )
    with pytest.raises(IndexError, match="index 6 is out of bounds for axis 2 with size 6"):
        sd.take_along_axis(sd.asarray(x), sd.asarray(np.full((4, 5, 1), 6)), axis=2)
    with pytest.raises(IndexError, match=r"^take_along_axis\(\): shapes could not be broadcast together"):
        sd.take_along_axis(sd.asarray(x), sd.asarray(np.zeros((3, 5, 1), int)), axis=2)


def test_nonzero_and_count_nonzero():
    z = RNG.integers(0, 3, (3, 4, 5)).astype(float)
    z[0, 0, 0] = np.nan  # NaN is nonzero
    got = sd.nonzero(sd.asarray(z))
    assert isinstance(got, tuple) and len(got) == 3
    for g, w in zip(got, np.nonzero(z)):
        assert_same(g, w)
    for g, w in zip(sd.nonzero(sd.asarray([0j, 1j, 2])), np.nonzero(np.array([0j, 1j, 2]))):
        assert_same(g, w)
    assert_same(sd.nonzero(sd.zeros((0,)))[0], np.zeros(0, np.int64))
    assert sd.count_nonzero(sd.asarray(z)) == np.count_nonzero(z)
    assert_same(sd.count_nonzero(sd.asarray(z), axis=(0, 2)), np.count_nonzero(z, axis=(0, 2)))
    assert_same(sd.count_nonzero(sd.asarray(z), axis=1, keepdims=True), np.count_nonzero(z, axis=1, keepdims=True))


def test_searchsorted():
    s = np.sort(np.concatenate([RNG.standard_normal(100), [np.nan, 0.5, 0.5, -np.inf]]))
    v = np.concatenate([RNG.standard_normal(50), [np.nan, 0.5, -np.inf, np.inf]]).reshape(6, 9)
    for side in ("left", "right"):
        assert_same(sd.searchsorted(sd.asarray(s), sd.asarray(v), side=side), np.searchsorted(s, v, side=side))
    assert sd.searchsorted(sd.asarray(s), 0.5) == np.searchsorted(s, 0.5)
    assert sd.searchsorted(sd.asarray([1, 2, 3]), 2.5) == 2
    u = RNG.standard_normal(30)
    so = np.argsort(u)
    assert_same(sd.searchsorted(sd.asarray(u), sd.asarray(v), sorter=sd.asarray(so)), np.searchsorted(u, v, sorter=so))
    big = np.sort(RNG.standard_normal(10_000))
    keys = RNG.standard_normal(200_000)
    assert_same(sd.searchsorted(sd.asarray(big), sd.asarray(keys)), np.searchsorted(big, keys))
    with pytest.raises(ValueError, match="'x1' must be 1-D"):
        sd.searchsorted(sd.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError, match="'side' must be 'left' or 'right'"):
        sd.searchsorted(sd.asarray([1.0]), 1.0, side="middle")
    with pytest.raises(TypeError, match="complex128"):
        sd.searchsorted(sd.asarray([1.0]), 1j)


def test_searchsorted_arguments():
    s = sd.asarray([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match=r"^searchsorted\(\): 'side' must be 'left' or 'right', not NoneType$"):
        sd.searchsorted(s, 1.0, side=None)
    # sorter is an array of positions; an int would sort nothing
    for bad, shown in [(1, "int"), (1.5, "float"), ([0, 1, 2], "list")]:
        with pytest.raises(TypeError, match=rf"^searchsorted\(\): 'sorter' must be an int64 array, not {shown}$"):
            sd.searchsorted(s, 1.0, sorter=bad)
    with pytest.raises(TypeError, match=r"^searchsorted\(\): 'sorter' must be an int64 array, not float64$"):
        sd.searchsorted(s, 1.0, sorter=sd.asarray([0.0, 1.0, 2.0]))
    with pytest.raises(IndexError, match=r"^searchsorted\(\): index 9 is out of bounds for axis 0 with size 3$"):
        sd.searchsorted(s, 1.0, sorter=sd.asarray([0, 1, 9]))
    with pytest.raises(
        TypeError,
        match=r"^searchsorted\(\): 'x2' must be an sd.ndarray or a bool, int, float or "
        r"complex, not NoneType$",
    ):
        sd.searchsorted(s, None)
    # An int outside int64 (docs/api/array.md §1.5): an error only where it must become an int64.
    with pytest.raises(OverflowError, match=r"^searchsorted\(\): "):
        sd.searchsorted(sd.asarray([1, 2]), 2**70)
    assert sd.searchsorted(s, 2**70) == 3
    assert sd.searchsorted(s, -(2**70)) == 0
    assert sd.searchsorted(sd.asarray([1, 2]), True) == 0


# ---------------------------------------------------------------- dtypes


def test_dtype_functions():
    assert sd.can_cast(sd.int64, sd.float64) and not sd.can_cast(sd.float64, sd.int64)
    assert sd.can_cast(sd.asarray([True]), "complex128")
    assert sd.result_type(sd.asarray([1]), 1.5) is sd.float64
    assert sd.result_type(sd.bool, True) is sd.bool
    assert sd.result_type(sd.asarray([1.0]), 1j, sd.int64) is sd.complex128
    with pytest.raises(ValueError):
        sd.result_type()
    with pytest.raises(TypeError, match="sd.string"):
        sd.can_cast(sd.string, sd.float64)
    kinds = {
        "bool": [sd.bool],
        "signed integer": [sd.int64],
        "unsigned integer": [],
        "integral": [sd.int64],
        "real floating": [sd.float64],
        "complex floating": [sd.complex128],
        "numeric": [sd.int64, sd.float64, sd.complex128],
    }
    for kind, members in kinds.items():
        for d in (sd.bool, sd.int64, sd.float64, sd.complex128):
            assert sd.isdtype(d, kind) == (d in members)
    assert sd.isdtype(sd.float64, ("bool", sd.float64)) and not sd.isdtype(sd.int64, sd.float64)
    with pytest.raises(ValueError, match="'kind' must be"):
        sd.isdtype(sd.int64, "integer")
    # Names work as in every dtype argument (API.md §1); messages are the shared ones.
    assert sd.isdtype("float64", "real floating") and sd.isdtype(sd.int64, ("bool", "int64"))
    with pytest.raises(
        TypeError,
        match=r"isdtype\(\): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128 "
        r"\(or their names\), not sd.string",
    ):
        sd.isdtype(sd.string, "numeric")
    with pytest.raises(TypeError, match=r"can_cast\(\): 'from_' must be sd.bool, .* or an sd.ndarray, not sd.string"):
        sd.can_cast(sd.string, sd.float64)
    # The accepted values are exactly the ones each takes, whatever was wrong.
    finfo_msg = (
        r"^finfo\(\): 'type' must be sd.float64 or sd.complex128 \(or their names\) "
        r"or an sd.ndarray of those dtypes, not "
    )
    iinfo_msg = r"^iinfo\(\): 'type' must be sd.int64 \(or its name\) or an sd.ndarray of int64, not "
    for arg, shown in [
        (sd.int64, "sd.int64"),
        ("int64", "'int64'"),
        (sd.ones(1, dtype=sd.int64), "an int64 array"),
        (1, "int"),
        (sd.string, "sd.string"),
    ]:
        with pytest.raises(TypeError, match=finfo_msg + shown + "$"):
            sd.finfo(arg)
    for arg, shown in [(sd.float64, "sd.float64"), (sd.ones(1), "a float64 array"), (1, "int")]:
        with pytest.raises(TypeError, match=iinfo_msg + shown + "$"):
            sd.iinfo(arg)
    f, nf = sd.finfo(sd.float64), np.finfo(np.float64)
    assert (f.bits, f.eps, f.max, f.min, f.smallest_normal, f.dtype) == (
        64,
        nf.eps,
        nf.max,
        nf.min,
        nf.smallest_normal,
        sd.float64,
    )
    assert sd.finfo(sd.asarray([1j])).dtype is sd.float64
    i = sd.iinfo(sd.int64)
    assert (i.bits, i.max, i.min, i.dtype) == (64, 2**63 - 1, -(2**63), sd.int64)
    with pytest.raises(TypeError):
        sd.finfo(sd.int64)
    with pytest.raises(TypeError):
        sd.iinfo(sd.float64)


# ----------------------------------------------------------- tensordot, vecdot


@pytest.mark.parametrize(
    "axes,shape_b", [(2, (4, 5, 6)), (1, (5, 2)), (0, (2,)), (([1, 0], [0, 2]), (4, 2, 3)), (([2], [1]), (2, 5, 7))]
)
def test_tensordot(axes, shape_b):
    b = RNG.standard_normal(shape_b)
    got = sd.tensordot(sd.asarray(X3), sd.asarray(b), axes=axes)
    np.testing.assert_allclose(np.asarray(got), np.tensordot(X3, b, axes), rtol=1e-13, atol=1e-14)
    i = RNG.integers(-3, 4, (3, 4))
    assert_same(sd.tensordot(sd.asarray(i), sd.asarray(i.T.copy()), axes=1), np.tensordot(i, i.T, 1))
    assert sd.tensordot(sd.asarray([1.0, 2.0]), sd.asarray([3.0, 4.0]), axes=1) == 11.0


def test_tensordot_errors():
    with pytest.raises(ValueError, match="'axes' has a repeated axis"):
        sd.tensordot(sd.asarray(X3), sd.asarray(X3), axes=([0, 0], [0, 1]))
    with pytest.raises(
        ValueError,
        match=r"^tensordot\(\): shape mismatch: axis 0 of 'x1' has length 3, "
        r"axis 1 of 'x2' has 4$",
    ):
        sd.tensordot(sd.asarray(X3), sd.asarray(X3), axes=([0], [1]))
    # every wrong form of axes: one message naming the wrong value's type, as for axis=(1.5,)
    for bad, shown in [
        (1.5, "float"),
        (([1.0], [0]), "float"),
        (([1], 0), "int"),
        (([1], [0], [2]), "tuple"),
        (None, "NoneType"),
    ]:
        with pytest.raises(
            TypeError,
            match=rf"^tensordot\(\): 'axes' must be an int or a pair of int sequences, "
            rf"not {shown}$",
        ):
            sd.tensordot(sd.asarray(X3), sd.asarray(X3), axes=bad)
    with pytest.raises(OverflowError, match=r"^tensordot\(\): 'axes' is out of range$"):
        sd.tensordot(sd.asarray(X3), sd.asarray(X3), axes=([2**70], [0]))
    with pytest.raises(ValueError, match="larger than"):
        sd.tensordot(sd.asarray([1.0]), sd.asarray([1.0]), axes=2)


def test_vecdot_keeps_the_array_dtype_rules():
    # D-22: the top-level vecdot is sum(conj(x1) * x2, axis): int stays int (wrapping), as
    # matmul and tensordot; sd.linalg.vecdot computes in float64.
    x, y = sd.asarray([1, 2]), sd.asarray([3, 4])
    r = sd.vecdot(x, y)
    assert r == 11 and type(r) is int
    assert sd.linalg.vecdot(x, y) == 11.0 and type(sd.linalg.vecdot(x, y)) is float
    assert sd.vecdot(sd.asarray([True, True]), sd.asarray([True, False])) == 1
    assert sd.vecdot(sd.asarray([2**62, 2**62]), sd.asarray([4, 4])) == 0  # wraps, as sum does
    a = np.arange(24).reshape(2, 3, 4)
    for axis in (-1, 0, 1):
        assert_same(sd.vecdot(sd.asarray(a), sd.asarray(a), axis=axis), np.sum(a * a, axis=axis))
    assert_same(sd.vecdot(sd.asarray(a), sd.asarray(a[0, 0]), axis=-1), np.sum(a * a[0, 0], axis=-1))
    # float64 and complex128: the linalg computation, the same bits
    rng = np.random.default_rng(4)
    f = sd.asarray(rng.standard_normal((50, 7)))
    assert_same(sd.vecdot(f, f), sd.linalg.vecdot(f, f))
    c = sd.asarray(rng.standard_normal(9) + 1j * rng.standard_normal(9))
    assert sd.vecdot(c, c) == sd.linalg.vecdot(c, c)
    assert sd.vecdot(x, sd.asarray([1.5, 2.0])) == 5.5
    # the shape rules and messages are linalg's
    with pytest.raises(
        ValueError, match=r"^vecdot\(\): 'x1' and 'x2' must have the same size along the axis, got 2 and 3$"
    ):
        sd.vecdot(x, sd.asarray([1, 2, 3]))
    with pytest.raises(ValueError, match=r"^vecdot\(\): batch shapes do not broadcast: \(2,\) \(4,\)$"):
        sd.vecdot(sd.ones((2, 3), dtype=sd.int64), sd.ones((4, 3), dtype=sd.int64))
    with pytest.raises(ValueError, match=r"^vecdot\(\): axis 3 is out of bounds"):
        sd.vecdot(x, y, axis=3)
    with pytest.raises(TypeError, match=r"^vecdot\(\): 'x2' must be an sd.ndarray, not list$"):
        sd.vecdot(x, [1, 2])


# ------------------------------------------------------------------ dlpack


def test_from_dlpack_shares_numpy_memory():
    x = np.arange(12.0).reshape(3, 4)[:, ::-1]
    a = sd.from_dlpack(x)
    assert_same(a, x)
    x[0, 0] = 99.0
    assert a[0, 0] == 99.0
    copied = sd.from_dlpack(x, copy=True)  # numpy makes the copy (DLPack's is-copied flag): no second one
    x[0, 1] = -5.0
    assert copied[0, 1] == 2.0
    for v in (np.array([1 + 2j]), np.array([True, False]), np.array([3, -4])):
        assert_same(sd.from_dlpack(v), v)
    ro = np.arange(3.0)
    ro.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        sd.from_dlpack(ro)[0] = 1.0


def test_dlpack_export_to_numpy():
    x = sd.asarray(np.arange(6).reshape(2, 3))
    n = np.from_dlpack(x)
    assert np.array_equal(n, np.arange(6).reshape(2, 3)) and n.flags.writeable
    n[0, 0] = 42
    assert x[0, 0] == 42  # shared
    b = np.from_dlpack(sd.broadcast_to(sd.asarray([1.0, 2.0]), (3, 2)))
    assert not b.flags.writeable  # read-only travels with the versioned format
    assert x.__dlpack_device__() == (1, 0)
    with pytest.raises(BufferError):
        sd.broadcast_to(sd.asarray([1.0]), (2,)).__dlpack__()  # legacy capsule cannot say read-only
    rt = sd.from_dlpack(sd.asarray([[1.0, 2.0], [3.0, 4.0]]).mT)
    assert_same(rt, np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_dlpack_errors_and_lifetime():
    with pytest.raises(TypeError, match="float32"):
        sd.from_dlpack(np.zeros(3, np.float32))
    with pytest.raises(TypeError, match="DLPack protocol"):
        sd.from_dlpack([1.0, 2.0])
    with pytest.raises(ValueError, match="0-dimensional"):
        sd.from_dlpack(np.array(1.0))

    class OnTheGpu:  # a producer on another device (docs/api/array.md §3.5)
        def __dlpack_device__(self):
            return (2, 0)

        def __dlpack__(self, **kw):
            raise AssertionError("not reached")

    # BufferError, as the Array API asks (API.md §2), named like every error of a function (§2.1)
    with pytest.raises(BufferError, match=r"^from_dlpack\(\): only CPU arrays can be imported$"):
        sd.from_dlpack(OnTheGpu())
    cap = sd.asarray([1.0, 2.0]).__dlpack__()
    del cap  # an unconsumed capsule frees its tensor
    for _ in range(3):
        big = np.ones(1_000_000)
        s = sd.from_dlpack(big)
        del big
        assert float(sd.sum(s)) == 1e6
        del s
    gc.collect()


# --------------------------------------------------------- bincount, histogram


def test_bincount():
    x = RNG.integers(0, 10, 1000)
    w = RNG.random(1000)
    assert_same(sd.bincount(sd.asarray(x)), np.bincount(x))
    assert_same(
        sd.bincount(sd.asarray(x), weights=sd.asarray(w), minlength=20), np.bincount(x, weights=w, minlength=20)
    )
    assert_same(sd.bincount(sd.zeros((0,), dtype=sd.int64), minlength=3), np.bincount(np.zeros(0, int), minlength=3))
    assert_same(sd.bincount(sd.asarray([True, False, True])), np.array([1, 2]))
    with pytest.raises(ValueError, match="non-negative"):
        sd.bincount(sd.asarray([1, -1]))
    with pytest.raises(TypeError, match="bool or int64"):
        sd.bincount(sd.asarray([1.0]))
    # weights: any real dtype (bool, int64, float64), summed as float64; complex is a TypeError
    # (docs/api/array.md §7.12)
    xi = sd.asarray([0, 1, 1])
    for w in ([2, 3, 4], [True, False, True]):
        assert_same(sd.bincount(xi, weights=sd.asarray(w)), np.bincount([0, 1, 1], weights=w))
    with pytest.raises(TypeError, match=r"^bincount\(\): 'weights' must be real, not complex128$"):
        sd.bincount(xi, weights=sd.asarray([1j, 2j, 3j]))


HIST_X = RNG.standard_normal(150_000)  # more than one 65536-element block
HIST_W = RNG.random(150_000)


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"bins": 37},
        {"bins": 5, "range": (-1, 2)},
        {"weights": HIST_W},
        {"weights": HIST_W, "bins": 13, "range": (-3, 0.5)},
        {"weights": RNG.integers(-5, 6, 150_000), "bins": 9},
        {"density": True},
        {"density": True, "weights": HIST_W},
        {"bins": np.array([-2, -0.5, 0, 0, 1, 3.0])},
        {"bins": np.array([-2, -0.5, 0, 1, 1.0])},
        {"bins": np.array([-3, 0, 3])},
        {"bins": np.array([-3, -1, 0, 2.0]), "weights": HIST_W},
    ],
    ids=lambda kw: ",".join(kw) or "default",
)
def test_histogram_bits(kw):
    skw = {k: sd.asarray(v) if isinstance(v, np.ndarray) else v for k, v in kw.items()}
    got = sd.histogram(sd.asarray(HIST_X), **skw)
    hist, edges = np.histogram(HIST_X, **kw)
    assert got.hist is got[0] and got.bin_edges is got[1]
    if "weights" in kw and isinstance(kw.get("bins"), np.ndarray) and hist.dtype.kind == "f":
        # given edges: each bin's own weights summed per block (numpy: differences of
        # cumulative sums), so the last bits differ, the result is closer
        assert np.asarray(got.hist).dtype == hist.dtype
        assert np.allclose(np.asarray(got.hist), hist, rtol=1e-12, atol=1e-9)
    else:
        assert_same(got.hist, hist)
    assert_same(got.bin_edges, edges)


def test_histogram_empty_bins_array():
    for edges in (np.array([], float), np.array([1.0])):
        for kw in ({}, {"weights": np.ones(3)}):
            got = sd.histogram(
                sd.asarray([1.0, 2.0, 3.0]), bins=sd.asarray(edges), **{k: sd.asarray(v) for k, v in kw.items()}
            )
            hist, e = np.histogram([1.0, 2.0, 3.0], bins=edges, **kw)
            assert_same(got.hist, hist)
            assert_same(got.bin_edges, e)


def test_histogram_edges_cases():
    xi = RNG.integers(0, 10, 1000)
    for g, w in zip(sd.histogram(sd.asarray(xi)), np.histogram(xi)):
        assert_same(g, w)
    for data in (np.ones(5), np.zeros(0)):
        for g, w in zip(sd.histogram(sd.asarray(data)), np.histogram(data)):
            assert_same(g, w)
    xn = HIST_X[:1000].copy()
    xn[5] = np.nan
    with pytest.raises(ValueError, match="range is not finite"):
        sd.histogram(sd.asarray(xn))
    assert_same(sd.histogram(sd.asarray(xn), range=(-1, 1)).hist, np.histogram(xn, range=(-1, 1))[0])
    with pytest.raises(ValueError, match="'bins' must increase monotonically"):
        sd.histogram(sd.asarray(xn), bins=sd.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="'bins' must be positive"):
        sd.histogram(sd.asarray(xn), bins=0)
    with pytest.raises(TypeError, match="'bins' must be an int or a 1-D array, not str"):
        sd.histogram(sd.asarray(xn), bins="auto")
    with pytest.raises(ValueError, match="'weights' must have the same shape"):
        sd.histogram(sd.asarray([1.0, 2.0]), weights=sd.asarray([1.0]))
    with pytest.raises(TypeError, match="complex128"):
        sd.histogram(sd.asarray([1j]))
    with pytest.raises(TypeError, match=r"'range' must hold two ints or floats, not bool"):
        sd.histogram(sd.asarray([1.0]), range=(0, True))


# ---- sizes that overflow int64 -----------------------------------------------


def test_tile_product_overflow_is_too_big():
    for x, reps in [(sd.ones(4), (2**62,)), (sd.ones(2), (2**63 - 1,)), (sd.ones((4, 3)), (2**62, 1))]:
        with pytest.raises(ValueError, match=r"^tile\(\): array is too big$"):
            sd.tile(x, reps)


def test_bincount_and_histogram_at_int64_max():
    with pytest.raises(ValueError, match=r"^bincount\(\): array is too big$"):
        sd.bincount(sd.asarray([2**63 - 1]))
    with pytest.raises(ValueError, match=r"^histogram\(\): array is too big$"):  # not linspace's 'num'
        sd.histogram(sd.asarray([1.0]), bins=2**63 - 1)


# ---- DLPack ----------------------------------------------------------------------


class _Producer:
    def __init__(self, device):
        self.device = device

    def __dlpack__(self, **kw):
        return np.ones(3).__dlpack__(**kw)

    def __dlpack_device__(self):
        return self.device


def test_dlpack_argument_types():
    x = sd.ones(2)
    cases = [
        (
            lambda: sd.from_dlpack(np.ones(3), device=("cpu", 0)),
            r"^from_dlpack\(\): 'device' must be None or the CPU device \(1, 0\), not a tuple holding str$",
        ),
        (
            lambda: sd.from_dlpack(np.ones(3), device="cpu"),
            r"^from_dlpack\(\): 'device' must be None or the CPU device \(1, 0\), not str$",
        ),
        (
            lambda: x.__dlpack__(dl_device=("cpu", 0)),
            r"^__dlpack__\(\): 'dl_device' must be None or a \(device_type, device_id\) tuple of ints, "
            r"not a tuple holding str$",
        ),
        (
            lambda: x.__dlpack__(max_version=("a", "b")),
            r"^__dlpack__\(\): 'max_version' must be None or a \(major, minor\) tuple of ints, "
            r"not a tuple holding str$",
        ),
        (lambda: x.__dlpack__(max_version=(1.0, 0)), r"not a tuple holding float$"),
        (lambda: x.__dlpack__(max_version=(True, 0)), r"not a tuple holding bool$"),  # bool is no int (API.md §2)
        (lambda: x.__dlpack__(max_version=[1, 0]), r"tuple of ints, not list$"),
        (
            lambda: sd.from_dlpack(_Producer(("x", 0))),
            r"^from_dlpack\(\): __dlpack_device__ must return a \(device_type, device_id\) tuple of ints, "
            r"not a tuple holding str$",
        ),
    ]
    for f, msg in cases:
        with pytest.raises(TypeError, match=msg):
            f()
    # right types, wrong values: as before
    with pytest.raises(ValueError, match=r"'device' must be None or the CPU device \(1, 0\)$"):
        sd.from_dlpack(np.ones(3), device=(2, 0))
    with pytest.raises(BufferError):
        x.__dlpack__(dl_device=(2, 0))
    with pytest.raises(BufferError):
        sd.from_dlpack(_Producer((2, 0)))
    assert_same(sd.from_dlpack(np.ones(3), device=(1, 0)), np.ones(3))
    assert_same(sd.from_dlpack(np.ones(3), device=(np.int64(1), 0)), np.ones(3))
    assert type(x.__dlpack__(max_version=(2**70, 0))).__name__ == "PyCapsule"


def test_dlpack_bool_check_contiguous_and_strided():
    b = np.random.default_rng(1).random((300, 400)) < 0.5
    for v in [b, b[:, ::3], b[::-1], b.T]:
        assert_same(sd.from_dlpack(v), v)
    raw = np.frombuffer(bytes([0, 1, 2, 1] * 100_000), dtype=np.uint8).view(bool).copy()
    assert_same(sd.from_dlpack(raw), raw != 0)  # normalizing copy
    assert_same(sd.from_dlpack(raw.reshape(200, 2000)[:, ::2]), raw.reshape(200, 2000)[:, ::2] != 0)
    with pytest.raises(ValueError, match="holds bytes other than 0 and 1"):
        sd.from_dlpack(raw, copy=False)


def test_diff_n_zero_is_x_itself():
    x = sd.asarray([1.0, 2.0])
    assert sd.diff(x, n=0, prepend=0.0) is x
    assert sd.diff(x, n=0, append=sd.asarray([9.0])) is x


def test_roll_shift_sequence_with_one_axis():
    a = np.arange(10.0).reshape(2, 5)
    for shift, axis in [((1, 2), 0), ((1, 2), 1), ((1, -1, 7), -1), ([2, 3], 1), ((), 1), ((2**62, 2**62), 1)]:
        assert_same(sd.roll(sd.asarray(a), shift, axis=axis), np.roll(a, shift, axis))
    assert_same(sd.roll(sd.arange(5.0), (1, 2), axis=0), np.roll(np.arange(5.0), 3))
    with pytest.raises(ValueError, match="repeated axis"):
        sd.roll(sd.asarray(a), 1, axis=(0, 0))


def test_searchsorted_without_copies():
    rng = np.random.default_rng(5)
    a = np.sort(rng.standard_normal(10_000))
    keys = rng.standard_normal(1000)
    for x1, x2 in [(a, keys), (a[::2], keys), (a, keys[::3]), (a.astype(np.int64), keys)]:
        assert_same(sd.searchsorted(sd.asarray(x1), sd.asarray(x2)), np.searchsorted(x1, x2))
        assert_same(sd.searchsorted(sd.asarray(x1), sd.asarray(x2), side="right"), np.searchsorted(x1, x2, "right"))
    assert sd.searchsorted(sd.asarray(a), 0.0) == np.searchsorted(a, 0.0)


def test_weighted_histogram_with_edges_sums_each_bin():
    import math

    rng = np.random.default_rng(8)
    x = rng.standard_normal(200_000)
    w = rng.standard_normal(200_000) * 10.0 ** rng.integers(-3, 3, 200_000)
    edges = np.sort(rng.standard_normal(41))
    got = np.asarray(sd.histogram(sd.asarray(x), bins=sd.asarray(edges), weights=sd.asarray(w)).hist)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == edges[-1]] = len(edges) - 2
    for b in range(len(edges) - 1):
        exact = math.fsum(w[idx == b])
        assert abs(got[b] - exact) <= 1e-13 * math.fsum(np.abs(w[idx == b])) + 1e-300
    wi = rng.integers(-(2**40), 2**40, 200_000)  # int64 weights: exact
    assert_same(
        sd.histogram(sd.asarray(x), bins=sd.asarray(edges), weights=sd.asarray(wi)).hist,
        np.histogram(x, bins=edges, weights=wi)[0],
    )
    old = sd.get_num_threads()
    try:
        res = []
        for n in (1, 10):
            sd.set_num_threads(n)
            res.append(np.asarray(sd.histogram(sd.asarray(x), bins=sd.asarray(edges), weights=sd.asarray(w)).hist))
        assert np.array_equal(res[0], res[1])
    finally:
        sd.set_num_threads(old)


def test_weighted_bincount_in_fixed_ranges():
    rng = np.random.default_rng(9)
    v = rng.integers(0, 500, 1_000_000)
    w = rng.standard_normal(1_000_000)
    old = sd.get_num_threads()
    try:
        res = []
        for n in (1, 3, 10):
            sd.set_num_threads(n)
            res.append(np.asarray(sd.bincount(sd.asarray(v), weights=sd.asarray(w))))
        assert all(np.array_equal(r, res[0]) for r in res)
    finally:
        sd.set_num_threads(old)
    assert np.allclose(res[0], np.bincount(v, w), rtol=1e-12, atol=1e-10)
    assert_same(sd.bincount(sd.asarray([0, 2, 2]), weights=sd.asarray([0.5, 1.0, 2.0])), np.array([0.5, 0.0, 3.0]))


def test_searchsorted_sorter_positions_are_in_range():
    # Positions in [0, n): a negative one is out of range, as one past the end (one rule).
    x1 = sd.asarray([3.0, 1.0, 2.0])
    for bad, shown in [([-2, -1, 0], -2), ([1, 2, -3], -3), ([0, 1, 9], 9)]:
        with pytest.raises(
            IndexError, match=rf"^searchsorted\(\): index {shown} is out of bounds for axis 0 with size 3$"
        ):
            sd.searchsorted(x1, 2.5, sorter=sd.asarray(bad))
    assert sd.searchsorted(x1, 2.5, sorter=sd.asarray([1, 2, 0])) == 2
