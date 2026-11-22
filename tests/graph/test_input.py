"""The input rules every sd.graph function shares (docs/api/graph.md §1.2, docs/design/graph.md §2.1, §3.1;
decisions GR7, GR8, GR21, GR22; milestone GR1): the four sparse formats and the dense forms give the same
graph (stored entries against the dense rule: 0, NaN and +-inf are no edge), dtypes, the TypeErrors (complex,
scipy.sparse, DataFrame, other objects) and ValueErrors (dimensions, squareness) with their messages, node
arguments, flags and options, and the call shape (options are keyword only)."""

import numpy as np
import pytest
import scipy.sparse as S

import stardust as sd

from . import graphs as G
from .forms import csr, forms

GR = sd.graph
ALL = (
    lambda A: GR.connected_components(A),
    lambda A: GR.breadth_first_order(A, 0),
    lambda A: GR.depth_first_order(A, 0),
    lambda A: GR.breadth_first_tree(A, 0),
    lambda A: GR.depth_first_tree(A, 0),
    lambda A: GR.topological_sort(A),
    lambda A: GR.csgraph_to_dense(A),
)
NAMES = (
    "connected_components",
    "breadth_first_order",
    "depth_first_order",
    "breadth_first_tree",
    "depth_first_tree",
    "topological_sort",
    "csgraph_to_dense",
)


def test_dense_rule_and_stored_entries():
    # dense: 0, NaN, +-inf are no edge; sparse: every stored entry is one (a stored 0 and NaN too)
    M = np.array([[0, np.nan, np.inf, 2.0], [-np.inf, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert np.asarray(GR.breadth_first_order(M, 0).node_array).tolist() == [0, 3]
    A = sd.sparse.coo_array((np.array([np.nan, 0.0, 2.0]), (np.array([0, 0, 0]), np.array([1, 2, 3]))), shape=(4, 4))
    assert np.asarray(GR.breadth_first_order(A, 0).node_array).tolist() == [0, 1, 2, 3]


def test_every_form_gives_the_same_graph():
    g = G.get("er/int60")
    want = np.asarray(GR.depth_first_order(csr(g), 5).node_array).tolist()
    for form, x in forms(g).items():
        assert np.asarray(GR.depth_first_order(x, 5).node_array).tolist() == want, form


def test_a_non_contiguous_dense_matrix():
    D = G.get("dag/30").dense()
    F = sd.asarray(np.asfortranarray(D))
    assert not F._flags()["c_contiguous"]
    assert np.asarray(GR.topological_sort(F)).tolist() == np.asarray(GR.topological_sort(D)).tolist()


def test_coo_duplicates_are_summed():
    # COO duplicates add when it becomes CSR: 1 + (-1) = 0 is a stored zero, an edge
    A = sd.sparse.coo_array((np.array([1.0, -1.0]), (np.array([0, 0]), np.array([1, 1]))), shape=(2, 2))
    T = GR.breadth_first_tree(A, 0)
    assert T.nnz == 1 and T.data[0] == 0.0


def test_dtypes():
    B = np.array([[False, True], [False, False]])
    I = np.array([[0, 3], [0, 0]])
    for x in (B, I, sd.asarray(B), sd.sparse.csr_array(B), sd.sparse.csr_array(I)):
        assert np.asarray(GR.breadth_first_order(x, 0).node_array).tolist() == [0, 1]
        T = GR.breadth_first_tree(x, 0)
        assert T.dtype == sd.float64 and T.data[0] == (3.0 if x.dtype in (np.int64, sd.int64) else 1.0)
        assert GR.csgraph_to_dense(x).dtype == sd.float64
    for x in (np.float32([[0, 1], [0, 0]]), np.int32([[0, 1], [0, 0]]), np.uint8([[0, 1], [0, 0]])):
        assert GR.connected_components(x).n_components == 1


def test_n_zero():
    for x in (sd.sparse.csr_array((0, 0)), np.zeros((0, 0)), sd.zeros((0, 0))):
        assert GR.connected_components(x).n_components == 0
        assert GR.topological_sort(x).shape == (0,)
        assert GR.csgraph_to_dense(x).shape == (0, 0)
        with pytest.raises(
            ValueError, match=r"^breadth_first_order\(\): 'i_start' must be a node of the graph, which has no nodes$"
        ):
            GR.breadth_first_order(x, 0)


@pytest.mark.parametrize("k", range(len(ALL)))
def test_input_errors(k):
    f, name = ALL[k], NAMES[k]
    with pytest.raises(TypeError, match=rf"^{name}\(\): 'csgraph' must be real, not a complex128 array$"):
        f(np.array([[0, 1j], [0, 0]]))
    with pytest.raises(TypeError, match=rf"^{name}\(\): 'csgraph' must be real, not a complex128 array$"):
        f(sd.sparse.csr_array(np.array([[0, 1j], [0, 0]])))
    with pytest.raises(
        TypeError,
        match=rf"^{name}\(\): 'csgraph' must be .*, not a scipy\.sparse csr_array; "
        r"convert it with sd\.sparse\.csr_array\(A\)$",
    ):
        f(S.csr_array(np.eye(2)))
    with pytest.raises(TypeError, match=rf"^{name}\(\): 'csgraph' must be .*, not a scipy\.sparse csr_matrix; convert"):
        f(S.csr_matrix(np.eye(2)))
    with pytest.raises(
        TypeError, match=rf"^{name}\(\): 'csgraph' must be an sd\.sparse array or a dense matrix .*, not DataFrame$"
    ):
        f(sd.frame.DataFrame({"a": [0.0, 1.0], "b": [0.0, 0.0]}))
    for bad in (None, "ab", {1: 2}, 3.0):
        with pytest.raises(TypeError, match=rf"^{name}\(\): 'csgraph' must be "):
            f(bad)
    with pytest.raises(TypeError, match=rf"^{name}\(\): 'csgraph' must be "):
        f([["a", "b"], ["c", "d"]])
    with pytest.raises(ValueError, match=rf"^{name}\(\): 'csgraph' must have 2 dimensions, not 1$"):
        f([0.0, 1.0])
    with pytest.raises(ValueError, match=rf"^{name}\(\): 'csgraph' must have 2 dimensions, not 3$"):
        f(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match=rf"^{name}\(\): 'csgraph' must have 2 dimensions, not 1$"):
        f(sd.sparse.csr_array(np.array([0.0, 1.0])))
    with pytest.raises(ValueError, match=rf"^{name}\(\): 'csgraph' must be a square matrix, not of shape \(2, 3\)$"):
        f(np.zeros((2, 3)))
    with pytest.raises(ValueError, match=rf"^{name}\(\): 'csgraph' must be a square matrix, not of shape \(3, 2\)$"):
        f(sd.sparse.csr_array((3, 2)))
    with pytest.raises(ValueError, match=rf"^{name}\(\): "):
        f([[0.0, 1.0], [0.0]])  # ragged


@pytest.mark.parametrize("name", ["breadth_first_order", "depth_first_order", "breadth_first_tree", "depth_first_tree"])
def test_node_arguments(name):
    f = getattr(GR, name)
    A = csr(G.get("path/10"))
    for bad in (True, np.bool_(False), 1.0, 1.5, np.float64(2.0), "1", None, sd.asarray([1])):
        with pytest.raises(TypeError, match=rf"^{name}\(\): 'i_start' must be an int, not "):
            f(A, bad)
    for bad in (-1, 10, 2**70, -(2**70)):
        with pytest.raises(
            ValueError, match=rf"^{name}\(\): 'i_start' must be a node of the graph \(0 \.\. 9\), not {bad}$"
        ):
            f(A, bad)
    for ok in (np.int64(3), np.int32(3), np.uint8(3)):  # numpy integers are ints (operator.index)
        f(A, ok)


def test_flags_and_options():
    A = csr(G.get("path/10"))
    for name in ("breadth_first_order", "depth_first_order"):
        for arg in ("directed", "return_predecessors"):
            with pytest.raises(TypeError, match=rf"^{name}\(\): '{arg}' must be True or False, not int$"):
                getattr(GR, name)(A, 0, **{arg: 1})
    for name in ("breadth_first_tree", "depth_first_tree"):
        with pytest.raises(TypeError, match=rf"^{name}\(\): 'directed' must be True or False, not NoneType$"):
            getattr(GR, name)(A, 0, directed=None)
    for arg in ("directed", "return_labels"):
        with pytest.raises(TypeError, match=rf"^connected_components\(\): '{arg}' must be True or False, not str$"):
            GR.connected_components(A, **{arg: "yes"})
    with pytest.raises(
        ValueError, match=r"^connected_components\(\): 'connection' must be 'weak' or 'strong', not 'Strong'$"
    ):
        GR.connected_components(A, connection="Strong")
    with pytest.raises(ValueError, match=r"connection"):  # checked even when directed=False reads none
        GR.connected_components(A, directed=False, connection="both")
    with pytest.raises(
        TypeError, match=r"^connected_components\(\): 'connection' must be 'weak' or 'strong', not NoneType$"
    ):
        GR.connected_components(A, connection=None)


def test_call_shape():
    A = csr(G.get("path/10"))
    with pytest.raises(TypeError, match=r"^connected_components\(\): "):
        GR.connected_components(A, False)  # GR21: options are keyword only
    with pytest.raises(TypeError, match=r"^breadth_first_order\(\): "):
        GR.breadth_first_order(A, 0, True)
    with pytest.raises(TypeError, match=r"^breadth_first_order\(\): .*'i_start'"):
        GR.breadth_first_order(A)
    with pytest.raises(TypeError, match=r"^depth_first_tree\(\): unexpected keyword argument 'overwrite'$"):
        GR.depth_first_tree(A, 0, overwrite=True)
    with pytest.raises(TypeError, match=r"^topological_sort\(\): "):
        GR.topological_sort(csgraph=A)


def test_inputs_are_not_changed():
    A = csr(G.get("er/float100"))
    before = (np.asarray(A.indptr).copy(), np.asarray(A.indices).copy(), np.asarray(A.data).copy())
    D = G.get("er/float100").dense()
    Dc = D.copy()
    for f in ALL:
        for x in (A, D):
            try:
                f(x)
            except ValueError:  # topological_sort: a cycle
                pass
    assert all(np.array_equal(np.asarray(a), b) for a, b in zip((A.indptr, A.indices, A.data), before))
    assert np.array_equal(D, Dc)
