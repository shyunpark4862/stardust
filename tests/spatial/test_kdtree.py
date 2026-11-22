"""KDTree's build, attributes, pickling and errors (docs/api/spatial.md §4, docs/design/spatial.md §2.4, §3.3;
decisions SPT50-SPT62): the tree's invariants (median splits on the widest axis under (coordinate, index),
boxes the range of their points, node counts from n and leafsize alone, leaves in index order), the same
tree bytes for 1, 3 and 10 threads, the data always copied, and every error with its message."""

import copy
import math
import pickle
import re

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import points as P

S = sd.spatial


def raises(exc, msg):
    return pytest.raises(exc, match="^" + re.escape(msg) + "$")


def node_count(c, leafsize):
    return 1 if c <= leafsize else 1 + node_count(c // 2, leafsize) + node_count(c - c // 2, leafsize)


def state(t):
    return [np.asarray(a) for a in t._tree._state()]


def check_invariants(t, x):
    x = np.asarray(x, dtype=np.float64)
    idx, starts, ends, rights, boxes = state(t)
    n, m, L = t.n, t.m, t.leafsize
    assert sorted(idx.tolist()) == list(range(n))
    assert t.size == len(starts) == node_count(n, L)
    assert starts[0] == 0 and ends[0] == n

    def walk(node, depth=0):
        s, e, r = int(starts[node]), int(ends[node]), int(rights[node])
        lo, hi = boxes[node, :m], boxes[node, m:]
        pts = x[idx[s:e]]
        if e > s:
            assert np.array_equal(lo, pts.min(axis=0)) and np.array_equal(hi, pts.max(axis=0)), node
        if r < 0:
            assert e - s <= L, node
            assert np.all(np.diff(idx[s:e]) > 0), node  # a leaf in index order (SPT50)
            return node + 1
        assert e - s > L, node
        left = node + 1
        mid = s + (e - s) // 2
        assert (starts[left], ends[left], starts[r], ends[r]) == (s, mid, mid, e), node
        dim = int(np.argmax(hi - lo)) if m else None  # the widest axis, the first on a tie
        keys_l = sorted((x[i, dim] if m else 0.0, i) for i in idx[s:mid])
        keys_r = sorted((x[i, dim] if m else 0.0, i) for i in idx[mid:e])
        assert keys_l[-1] < keys_r[0], node  # the median under (coordinate, index)
        after = walk(left, depth + 1)
        assert after == r, node  # preorder numbering
        return walk(r, depth + 1)

    assert walk(0) == t.size


@pytest.mark.parametrize(
    ("n", "m", "leafsize"),
    [(0, 3, 10), (1, 2, 10), (10, 3, 10), (11, 3, 10), (200, 3, 1), (513, 4, 7), (1000, 1, 10), (300, 0, 5),
     (70000, 3, 3000)],  # the last: the pool's selection at the top (SPT79)
)  # fmt: skip
def test_build_invariants(n, m, leafsize):
    x = np.random.default_rng(n + 7 * m).standard_normal((n, m))
    check_invariants(S.KDTree(x, leafsize=leafsize), x)


def test_build_invariants_with_many_equal_coordinates():
    g = np.random.default_rng(3)
    x = g.integers(0, 3, (700, 3)).astype(float)  # every coordinate repeated about 230 times
    x[:100] = 1.0  # 100 identical points
    t = S.KDTree(x, leafsize=4)
    check_invariants(t, x)
    assert t.size == node_count(700, 4)  # split even with no width (SPT50; scipy stops there)
    s = P.get("dup/3")
    check_invariants(S.KDTree(s.x), s.x)


def test_build_invariants_of_the_pool_selection_with_ties():
    # SPT79: the top nodes are split by a sampled band and nth_element on it; many equal coordinates put
    # the band's ends among ties, broken by the index
    g = np.random.default_rng(3)
    x = g.integers(0, 4, (90000, 2)).astype(np.float64)
    check_invariants(S.KDTree(x, leafsize=5000), x)
    x = np.zeros((70000, 2))
    x[::7, 1] = 1.0
    check_invariants(S.KDTree(x, leafsize=5000), x)


def test_attributes():
    x = np.random.default_rng(1).standard_normal((50, 3))
    t = S.KDTree(x, leafsize=4)
    assert (t.n, t.m, t.leafsize, t.size) == (50, 3, 4, node_count(50, 4))
    assert all(type(v) is int for v in (t.n, t.m, t.leafsize, t.size))
    assert bits_equal(t.data, x) and t.data.dtype == sd.float64
    assert bits_equal(t.mins, x.min(axis=0)) and bits_equal(t.maxes, x.max(axis=0))
    for a in (t.data, t.mins, t.maxes):
        assert not a._flags()["writeable"]
        with pytest.raises(ValueError):
            a[0] = 1.0
    assert t.data is t.data  # read without a copy
    assert repr(t) == "KDTree(n=50, m=3, leafsize=4)"
    with pytest.raises(AttributeError):
        t.n = 3
    with pytest.raises(AttributeError):
        t.extra = 1
    for name in ("tree", "indices", "boxsize"):  # scipy's, not here
        assert not hasattr(t, name)


def test_empty_trees():
    t = S.KDTree(np.zeros((0, 3)))
    assert (t.n, t.m, t.size) == (0, 3, 1)
    assert bits_equal(t.mins, np.zeros(3)) and bits_equal(t.maxes, np.zeros(3))  # scipy's zeros
    z = S.KDTree(np.zeros((4, 0)))
    assert (z.n, z.m) == (4, 0) and z.mins.shape == (0,)


def test_data_is_always_copied():
    x = np.random.default_rng(2).standard_normal((40, 2))
    keep = x.copy()
    for src in (x, sd.asarray(x)):
        t = S.KDTree(src)
        d0, i0 = t.query(np.zeros(2), k=5)
        if isinstance(src, np.ndarray):
            src[:] = 0.0
        x[:] = keep
        assert bits_equal(t.data, keep)
        d1, i1 = t.query(np.zeros(2), k=5)
        assert bits_equal(d0, d1) and bits_equal(i0, i1)


def test_inputs():
    x = np.random.default_rng(4).standard_normal((30, 3))
    want = S.KDTree(x).query(x[:4], k=3)
    for src in (x.tolist(), sd.asarray(x), sd.asarray(x.T).T, x.astype(np.float32)):
        got = S.KDTree(src).query(x[:4], k=3)
        if not isinstance(src, np.ndarray) or src.dtype == np.float64:
            assert bits_equal(got, want)
    ints = np.arange(12).reshape(4, 3)
    assert S.KDTree(ints).data.dtype == sd.float64
    assert S.KDTree(ints > 5).data.dtype == sd.float64
    df = sd.frame.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 1.0, 0.0]})
    assert bits_equal(S.KDTree(df).data, np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]))


def test_pickle_and_copy_rebuild_the_same_tree():
    x = np.random.default_rng(5).standard_normal((2000, 3))
    t = S.KDTree(x, leafsize=6)
    for u in (pickle.loads(pickle.dumps(t)), copy.copy(t), copy.deepcopy(t)):
        assert type(u) is S.KDTree and u.leafsize == 6
        assert bits_equal(u.data, t.data)
        assert all(bits_equal(a, b) for a, b in zip(state(u), state(t)))
        assert bits_equal(u.query(x[:50], k=4), t.query(x[:50], k=4))
    e = pickle.loads(pickle.dumps(S.KDTree(np.zeros((0, 2)))))
    assert (e.n, e.m) == (0, 2)


@pytest.mark.usefixtures("restore_threads")
def test_the_same_tree_for_any_thread_count():
    # large enough for the parallel build: the top on the calling thread, subtrees on the pool, the boxes
    # of the top nodes by chunks
    g = np.random.default_rng(6)
    x = g.standard_normal((120_000, 3))
    x[::7] = np.round(x[::7], 1)  # equal coordinates for the (coordinate, index) order
    trees = {}
    for k in (1, 3, 10):
        sd.set_num_threads(k)
        trees[k] = state(S.KDTree(x, leafsize=8))
    for k in (3, 10):
        assert all(bits_equal(a, b) for a, b in zip(trees[1], trees[k])), k
    sd.set_num_threads(10)
    check_invariants(S.KDTree(x[:30_000]), x[:30_000])


# ---- errors ----


def test_constructor_errors():
    x = np.zeros((4, 3))
    with raises(TypeError, "KDTree(): takes 1 positional argument but 2 were given"):
        S.KDTree(x, 5)  # SPT15: leafsize is keyword only
    with raises(TypeError, "KDTree(): unexpected keyword argument 'balanced_tree'"):
        S.KDTree(x, balanced_tree=False)
    with raises(TypeError, "KDTree(): unexpected keyword argument 'copy_data'"):
        S.KDTree(x, copy_data=True)
    with raises(TypeError, "KDTree(): unexpected keyword argument 'boxsize'"):
        S.KDTree(x, boxsize=1.0)
    with raises(TypeError, "KDTree(): 'leafsize' must be an int, not float"):
        S.KDTree(x, leafsize=2.5)  # scipy truncates to 2
    with raises(TypeError, "KDTree(): 'leafsize' must be an int, not bool"):
        S.KDTree(x, leafsize=True)
    with raises(ValueError, "KDTree(): 'leafsize' must be at least 1, not 0"):
        S.KDTree(x, leafsize=0)
    with raises(OverflowError, "KDTree(): 'leafsize' is out of range"):
        S.KDTree(x, leafsize=2**64)
    with raises(ValueError, "KDTree(): 'data' must have 2 dimensions, not 1"):
        S.KDTree(np.zeros(3))
    with raises(ValueError, "KDTree(): 'data' must have 2 dimensions, not 3"):
        S.KDTree(np.zeros((2, 2, 2)))
    with raises(ValueError, "KDTree(): 'data' must be finite"):
        S.KDTree([[math.nan, 0.0], [1.0, 1.0]])
    with raises(ValueError, "KDTree(): 'data' must be finite"):
        S.KDTree([[0.0, 0.0], [1.0, -math.inf]])
    with raises(TypeError, "KDTree(): 'data' must be real, not a complex128 array"):
        S.KDTree(np.ones((3, 2), complex))
    with raises(TypeError, "KDTree(): 'data' must be an sd.ndarray, a sequence of numbers, a numpy array or a "
                           "frame DataFrame, not str"):  # fmt: skip
        S.KDTree("abc")
    assert S.KDTree(x, leafsize=np.int64(3)).leafsize == 3
