"""concat and stack."""

import numpy as np
import pytest

import stardust as sd

from .conftest import NP_DTYPE, assert_same, run_guarded


def arrays(*shapes, dtype=np.float64):
    ns = [np.arange(int(np.prod(s))).astype(dtype).reshape(s) + 100 * k for k, s in enumerate(shapes)]
    return ns, [sd.asarray(n) for n in ns]


@pytest.mark.parametrize(
    "shapes,axis",
    [
        ([(2, 3), (4, 3)], 0),
        ([(2, 3), (2, 1), (2, 0)], 1),
        ([(2, 3), (2, 2)], -1),
        ([(5,)], 0),
        ([(2, 2, 2), (2, 3, 2)], 1),
    ],
)
def test_concat(shapes, axis):
    ns, xs = arrays(*shapes)
    assert_same(sd.concat(xs, axis=axis), np.concatenate(ns, axis=axis))
    assert_same(sd.concat(tuple(xs), axis=axis), np.concatenate(ns, axis=axis))


def test_concat_axis_none_flattens():
    ns, xs = arrays((2, 3), (4,), (1, 2, 2))
    xs[0] = xs[0].T  # layout must not matter: C index order
    ns[0] = ns[0].T
    assert_same(sd.concat(xs, axis=None), np.concatenate(ns, axis=None))


def test_concat_promotes():
    got = sd.concat([sd.asarray([True]), sd.asarray([2]), sd.asarray([0.5])])
    assert_same(got, np.array([1.0, 2.0, 0.5]))
    got = sd.concat([sd.asarray([1j]), sd.asarray([2])])
    assert got.dtype is sd.complex128


def test_concat_from_strided_views():
    n = np.arange(24.0).reshape(4, 6)
    a = sd.asarray(n)
    assert_same(sd.concat([a[::2, ::-1], a.T[:2, :2]], axis=1), np.concatenate([n[::2, ::-1], n.T[:2, :2]], axis=1))


@pytest.mark.parametrize(
    "bad,exc",
    [
        ([], ValueError),
        ([sd.zeros((2, 3)), sd.zeros((2, 4))], ValueError),
        ([sd.zeros((2, 3)), sd.zeros(3)], ValueError),
        ([sd.zeros(2), [1, 2]], TypeError),
        (sd.zeros(2), TypeError),
        ([np.zeros(2)], TypeError),
    ],
)
def test_concat_errors(bad, exc):
    with pytest.raises(exc):
        sd.concat(bad)


def test_concat_axis_out_of_bounds():
    with pytest.raises(ValueError):
        sd.concat([sd.zeros(2)], axis=1)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
def test_stack(axis, dtype):
    ns, xs = arrays((2, 3), (2, 3), (2, 3), dtype=NP_DTYPE[dtype])
    assert_same(sd.stack(xs, axis=axis), np.stack(ns, axis=axis))


def test_stack_errors():
    with pytest.raises(ValueError):
        sd.stack([sd.zeros(2), sd.zeros(3)])
    with pytest.raises(ValueError):
        sd.stack([sd.zeros(2)], axis=2)
    with pytest.raises(ValueError):
        sd.stack([])
    with pytest.raises(TypeError):
        sd.stack([sd.zeros(2)], axis=None)


def test_results_are_fresh_c_contiguous():
    a = sd.zeros((2, 2))
    for r in (sd.concat([a, a]), sd.stack([a, a], axis=2)):
        assert r.base is None and r._flags()["c_contiguous"]


# ---- sizes that overflow int64 -----------------------------------------------


def test_concat_of_huge_broadcast_views_is_too_big():
    # Unchecked, the length sum wraps to 0 and the copies write past the allocation (SIGSEGV).
    rc, out = run_guarded(
        """
        import stardust as sd
        b = sd.broadcast_to(sd.ones(1, dtype=sd.bool), (2**62,))
        for f in [lambda: sd.concat([b] * 4), lambda: sd.concat([b] * 4, axis=None), lambda: sd.concat([b, b]),
                  lambda: sd.diff(b, prepend=b)]:
            try:
                f()
                print("returned")
            except ValueError as e:
                print(e)
        """
    )
    assert rc == 0, out
    assert out.split("\n")[:4] == [
        "concat(): array is too big",
        "concat(): array is too big",
        "concat(): array is too big",
        "diff(): array is too big",
    ]
