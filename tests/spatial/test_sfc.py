"""The orders of csrc/spatial/sfc.hpp (docs/design/spatial_geometry.md §3.3, §3.4; SPT8): the
lexicographic order with its representatives, the Hilbert keys, and the BRIO insertion order."""

import itertools

import numpy as np
import pytest

import stardust as sd

from . import geompoints as gp

C = sd._core.spatial
SETS = [i for i in gp.IDS if "large" not in gp.get(i).tags]


@pytest.mark.parametrize("id_", SETS)
def test_lex_order_and_representatives(id_):
    x = gp.get(id_).x
    order, rep = (np.asarray(a) for a in C._lex_order(sd.asarray(x)))
    keyed = sorted(range(len(x)), key=lambda i: (tuple(v + 0.0 for v in x[i]), i))  # + 0.0: -0.0 is 0.0
    assert order.tolist() == keyed
    first = {}
    want = [first.setdefault(tuple(v + 0.0 for v in p), i) for i, p in enumerate(x.tolist())]
    assert rep.tolist() == want


def test_negative_zero_is_zero():
    x = np.array([[0.0, 1.0], [-0.0, 1.0], [-0.0, -0.0], [0.0, 0.0]])
    order, rep = (np.asarray(a) for a in C._lex_order(sd.asarray(x)))
    assert order.tolist() == [2, 3, 0, 1] and rep.tolist() == [0, 0, 2, 2]


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_hilbert_curve(d, bits):
    """Every cell once, and consecutive cells share a face."""
    cells = np.array(list(itertools.product(range(2**bits), repeat=d)), dtype=np.int64)
    keys = np.asarray(C._hilbert_keys(sd.asarray(cells), bits))
    assert sorted(keys.tolist()) == list(range(2 ** (d * bits)))
    path = cells[np.argsort(keys)]
    assert (np.abs(np.diff(path, axis=0)).sum(axis=1) == 1).all()


def test_hilbert_keys_at_full_resolution():
    g = np.random.default_rng(2)
    for d in (2, 3):
        cells = g.integers(0, 2**21, (2000, d))
        keys = np.asarray(C._hilbert_keys(sd.asarray(cells), 21))
        assert len(set(keys.tolist())) == len({tuple(c) for c in cells.tolist()})
        assert keys.min() >= 0 and keys.max() < 2 ** (21 * d)
        # a cell and its neighbour at a coarser level share the high bits of the key: locality
        coarse = np.asarray(C._hilbert_keys(sd.asarray(cells >> 5), 16))
        assert np.array_equal(keys >> (5 * d), coarse)


@pytest.mark.parametrize("id_", SETS)
def test_brio_order(id_):
    """A permutation of the representatives; a function of the coordinates (shuffling the rows gives the
    same points in the same order); rounds that double."""
    x = gp.get(id_).x
    order = np.asarray(C._brio_order(sd.asarray(x)))
    _, rep = (np.asarray(a) for a in C._lex_order(sd.asarray(x)))
    assert sorted(order.tolist()) == sorted(set(rep.tolist()))
    perm = np.random.default_rng(4).permutation(len(x))
    again = np.asarray(C._brio_order(sd.asarray(np.ascontiguousarray(x[perm]))))
    assert np.array_equal(x[perm][again] + 0.0, x[order] + 0.0)


def test_brio_rounds_and_locality():
    x = gp.get("uniform2/20000").x
    order = np.asarray(C._brio_order(sd.asarray(x)))
    assert len(order) == 20000
    first = C._predicate_constants()["brio_first_round"]
    # the last round holds about half the points and is one Hilbert sweep: consecutive points are close
    tail = x[order[-9000:]]
    step = np.linalg.norm(np.diff(tail, axis=0), axis=1)
    assert np.median(step) < 4.0 / np.sqrt(10000)
    head = x[order[: first // 2]]
    assert np.median(np.linalg.norm(np.diff(head, axis=0), axis=1)) > 10 * np.median(step)


def test_wrong_calls():
    with pytest.raises(TypeError):
        C._lex_order(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        C._lex_order(sd.asarray(np.zeros((3, 4))))
    with pytest.raises(ValueError):
        C._hilbert_keys(sd.asarray(np.array([[4, 0]])), 2)
    assert np.asarray(C._brio_order(sd.asarray(np.zeros((0, 2))))).shape == (0,)
