"""reverse_cuthill_mckee and structural_rank (docs/api/graph.md §5, docs/design/graph.md §2.5, §3.6, §4;
decision GR52; milestone GR4): the permutation of the definition (tests/graph/definitions.py) on every
test graph and input form and on random graphs; scipy 1.18.1's algorithm is the same one with numpy's
unstable argsort for the seeds, so the two agree whenever that sort keeps the ties in index order (always
below 17 nodes) and the pattern of A + A^T has no cancelled entry; the structural rank is the largest
matching's size."""

import numpy as np
import pytest
import scipy.sparse as S
import scipy.sparse.csgraph as CS

import stardust as sd

from . import definitions as D
from . import graphs as G
from .conftest import LIVE
from .forms import csr, forms, square_ids

GR = sd.graph


def pairs_of(g):
    return {(i, j) for i, j, _ in g.edges()}


def scipy_seeds(degree):
    """scipy's seed order: numpy's default (unstable) argsort of the int32 degrees."""
    return np.argsort(np.asarray(degree, dtype=np.int32)).tolist()


@pytest.mark.parametrize("gid", square_ids())
def test_rcm_is_the_definition_for_every_form(gid):
    g = G.get(gid)
    pairs = pairs_of(g)
    for mode in (False, True):
        want = D.reverse_cuthill_mckee(g.n, pairs, mode)
        for form, A in forms(g).items():
            got = GR.reverse_cuthill_mckee(A, symmetric_mode=mode)
            assert isinstance(got, sd.ndarray) and got.dtype == sd.int64 and got.shape == (g.n,), form
            assert np.asarray(got).tolist() == want, (form, mode)


def _random(n, d, seed, loops=False):
    r = np.random.Generator(np.random.PCG64([seed, n, d]))
    rows, cols = r.integers(0, n, n * d), r.integers(0, n, n * d)
    keep = loops | (rows != cols)
    a = S.coo_array((r.uniform(0.5, 2, int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n)).tocsr()
    a.sum_duplicates()
    a.sort_indices()
    return a


@pytest.mark.parametrize("n", [1, 5, 16, 17, 40, 300, 2000])
def test_rcm_random_graphs(n):
    for seed in range(6):
        for d in (1, 2, 4):
            a = _random(n, d, seed, loops=seed % 2 == 1)
            A = sd.sparse.csr_array(a)
            pairs = set(zip(*a.nonzero()))
            for mode in (False, True):
                got = np.asarray(GR.reverse_cuthill_mckee(A, symmetric_mode=mode)).tolist()
                assert got == D.reverse_cuthill_mckee(n, pairs, mode), (seed, d, mode)
                assert sorted(got) == list(range(n))


@LIVE
@pytest.mark.parametrize("n", [5, 12, 16, 17, 40, 300])
def test_scipy_is_the_same_algorithm_with_its_own_seed_ties(n):
    """scipy's permutation is the definition with numpy's argsort for the seeds (the only tie scipy leaves
    to an unstable sort); below 17 nodes numpy's sort keeps ties in index order, so sd and scipy agree."""
    differs = 0
    for seed in range(8):
        for d in (1, 2, 3):
            a = _random(n, d, seed, loops=seed % 3 == 0)
            pairs = set(zip(*a.nonzero()))
            for mode in (False, True):
                b = a if mode else S.csr_array(a + a.T)
                want = CS.reverse_cuthill_mckee(b, symmetric_mode=mode).tolist()
                assert want == D.reverse_cuthill_mckee(n, pairs, mode, scipy_seeds), (seed, d, mode)
                ours = np.asarray(GR.reverse_cuthill_mckee(sd.sparse.csr_array(b), symmetric_mode=mode)).tolist()
                if n < 17:
                    assert ours == want, (seed, d, mode)
                differs += ours != want
    if n >= 40:
        assert differs > 0  # [scipy와 다름]: the seed ties of the unstable sort


@LIVE
def test_scipy_rcm_on_the_test_graphs():
    """The recorded behaviour holds for the live scipy too: the test graphs below 17 nodes without stored
    zeros or cancelling pairs give scipy's permutation."""
    checked = 0
    for gid in square_ids():
        g = G.get(gid)
        if g.n == 0 or g.n >= 17 or g.data.dtype == np.bool_:
            continue
        a = g.scipy().copy() if g.matrix is None else S.csr_array(np.where(np.isfinite(g.matrix), g.matrix, 0.0))
        a.eliminate_zeros()
        if (a + a.T).nnz != len({(i, j) for i, j in zip(*a.nonzero())} | {(j, i) for i, j in zip(*a.nonzero())}):
            continue
        A = sd.sparse.csr_array(a.toarray())
        if A.nnz != a.nnz:  # a stored NaN or inf
            continue
        for mode in (False, True):
            assert np.asarray(GR.reverse_cuthill_mckee(A, symmetric_mode=mode)).tolist() == (
                CS.reverse_cuthill_mckee(a, symmetric_mode=mode).tolist()
            ), (gid, mode)
        checked += 1
    assert checked >= 15


@LIVE
def test_rcm_reads_the_pattern_where_scipy_sums():
    """[scipy와 다름]: scipy forms A + A^T numerically, so a stored 0 one way and two weights that cancel
    are no edge; sd reads the pattern (a structure function, docs/design/graph.md §2.1)."""
    a = S.csr_array((np.array([1.0, -1.0, 0.0, 1.0, 1.0]), ([0, 1, 2, 2, 3], [1, 0, 3, 4, 4])), shape=(5, 5))
    ours = np.asarray(GR.reverse_cuthill_mckee(sd.sparse.csr_array(a))).tolist()
    assert ours == D.reverse_cuthill_mckee(5, {(0, 1), (1, 0), (2, 3), (2, 4), (3, 4)}, False) == [4, 3, 2, 1, 0]
    assert CS.reverse_cuthill_mckee(a).tolist() == [3, 4, 2, 1, 0]  # 0 - 1 and 2 - 3 vanish in scipy's sum
    # with symmetric_mode the stored entries are read by both
    assert np.asarray(GR.reverse_cuthill_mckee(sd.sparse.csr_array(a), symmetric_mode=True)).tolist() == (
        CS.reverse_cuthill_mckee(a, symmetric_mode=True).tolist()
    )


def test_rcm_self_loops_count_twice_and_ties():
    # a path 0 - 1 - 2 - 3 with a loop at 0: degrees 3, 2, 2, 1; the seed is 3
    a = sd.sparse.csr_array(np.array([[1.0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]))
    assert np.asarray(GR.reverse_cuthill_mckee(a)).tolist() == [0, 1, 2, 3]
    # star 0 with leaves 1..4: the leaves tie (degree 1) and the seed is the smallest, 1
    star = np.zeros((5, 5))
    star[0, 1:] = 1
    assert np.asarray(GR.reverse_cuthill_mckee(star)).tolist() == [4, 3, 2, 0, 1]
    # components follow their seeds' order; isolated nodes (degree 0) come first, reversed last
    two = np.zeros((6, 6))
    two[1, 2] = two[4, 5] = two[5, 3] = 1
    assert np.asarray(GR.reverse_cuthill_mckee(two)).tolist() == D.reverse_cuthill_mckee(
        6, {(1, 2), (4, 5), (5, 3)}, False
    )
    assert np.asarray(GR.reverse_cuthill_mckee(two)).tolist()[-1] == 0


def test_rcm_edges_and_errors():
    assert np.asarray(GR.reverse_cuthill_mckee(sd.sparse.csr_array((0, 0)))).tolist() == []
    assert GR.reverse_cuthill_mckee(np.zeros((0, 0))).dtype == sd.int64
    assert np.asarray(GR.reverse_cuthill_mckee([[0.0]])).tolist() == [0]
    # a stored NaN is an edge; a dense NaN is not (the graph rule)
    nan = sd.sparse.csr_array((sd.asarray([np.nan]), sd.asarray([2]), sd.asarray([0, 1, 1, 1])), shape=(3, 3))
    assert np.asarray(GR.reverse_cuthill_mckee(nan)).tolist() == [2, 0, 1]
    assert np.asarray(GR.reverse_cuthill_mckee([[0, 0, np.nan], [0, 0, 0], [0, 0, 0]])).tolist() == [2, 1, 0]
    with pytest.raises(ValueError, match=r"reverse_cuthill_mckee\(\): 'graph' must be a square matrix"):
        GR.reverse_cuthill_mckee(np.zeros((2, 3)))
    with pytest.raises(TypeError, match="'symmetric_mode' must be True or False"):
        GR.reverse_cuthill_mckee(np.eye(2), symmetric_mode=1)
    with pytest.raises(TypeError, match="positional"):
        GR.reverse_cuthill_mckee(np.eye(2), True)
    with pytest.raises(TypeError, match=r"sd.sparse.csr_array\(A\)"):
        GR.reverse_cuthill_mckee(S.eye_array(3, format="csr"))


@pytest.mark.parametrize("gid", [i for i in G.ids() if "cost" not in G.get(i).tags])
def test_structural_rank_is_the_matching_size(gid):
    g = G.get(gid)
    rows, cols = g.shape
    want = D.matching_size(rows, cols, pairs_of(g))
    A = csr(g) if g.square else sd.sparse.csr_array((g.data, g.indices, g.indptr), shape=g.shape)
    for B in (A, A.tocsc(), A.tocoo()):
        got = GR.structural_rank(B)
        assert type(got) is int and got == want, gid


@LIVE
def test_scipy_structural_rank():
    for gid in G.ids():
        g = G.get(gid)
        if "cost" in g.tags or g.matrix is not None:
            continue
        a = g.scipy()
        if 0 in a.shape:
            continue
        A = sd.sparse.csr_array((g.data, g.indices, g.indptr), shape=g.shape)
        assert GR.structural_rank(A) == int(CS.structural_rank(a)), gid


def test_structural_rank_shapes_and_forms():
    assert GR.structural_rank(np.zeros((0, 3))) == 0
    assert GR.structural_rank(sd.sparse.csr_array((4, 0))) == 0
    assert GR.structural_rank(np.eye(3)[:, ::-1]) == 3  # a strided view
    assert GR.structural_rank([[1, 1, 0], [0, 0, 0]]) == 1
    assert GR.structural_rank(np.ones((3, 5), dtype=bool)) == 3
    zeros = sd.sparse.coo_array((sd.zeros(2), (sd.asarray([0, 1]), sd.asarray([1, 0]))), shape=(2, 2))
    assert GR.structural_rank(zeros) == 2  # a stored 0 is an entry
    with pytest.raises(ValueError, match=r"structural_rank\(\): 'graph' must have 2 dimensions"):
        GR.structural_rank(np.zeros(3))
    with pytest.raises(TypeError, match="must be real"):
        GR.structural_rank(np.eye(2) * 1j)
