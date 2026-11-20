"""Option arguments of linalg: one check per kind, the frame's message format
(DESIGN.md §10.3, API.md §2.1), and numpy integers as ints (DESIGN.md §10.1 #13)."""

import numpy as np
import pytest

import stardust as sd

LA = sd.linalg
A = sd.asarray([[4.0, 1.0], [1.0, 3.0]])
V = sd.asarray([3.0, 4.0])

BOOL_OPTIONS = [
    (lambda **kw: LA.cholesky(A, **kw), "cholesky", "upper"),
    (lambda **kw: LA.cholesky_solve(A, A, **kw), "cholesky_solve", "upper"),
    (lambda **kw: LA.solve_triangular(A, A, **{"upper": True, **kw}), "solve_triangular", "upper"),
    (lambda **kw: LA.solve_triangular(A, A, upper=True, **kw), "solve_triangular", "unit_diagonal"),
    (lambda **kw: LA.matrix_norm(A, **kw), "matrix_norm", "keepdims"),
    (lambda **kw: LA.vector_norm(V, **kw), "vector_norm", "keepdims"),
]


@pytest.mark.parametrize(("call", "name", "option"), BOOL_OPTIONS)
def test_bool_options_take_true_or_false(call, name, option):
    for bad, shown in ((1, "int"), (0, "int"), (None, "NoneType"), (np.True_, "numpy.bool"), ("yes", "str")):
        with pytest.raises(TypeError, match=rf"^{name}\(\): '{option}' must be True or False, not {shown}$"):
            call(**{option: bad})
    call(**{option: True})
    call(**{option: False})


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: LA.svd(A, mode=None), r"^svd\(\): 'mode' must be 'reduced' or 'complete', not NoneType$"),
        (lambda: LA.qr(A, mode=1), r"^qr\(\): 'mode' must be 'reduced' or 'complete', not int$"),
        (
            lambda: LA.qr_pivoted(A, mode=None),
            r"^qr_pivoted\(\): 'mode' must be 'reduced' or 'complete', not NoneType$",
        ),
        (
            lambda: LA.matrix_norm(A, ord=None),
            r"^matrix_norm\(\): 'ord' must be a str, an int or a float, not NoneType$",
        ),
        (lambda: LA.cond(A, ord=None), r"^cond\(\): 'ord' must be a str, an int or a float, not NoneType$"),
        (lambda: LA.vector_norm(V, ord=None), r"^vector_norm\(\): 'ord' must be an int or a float, not NoneType$"),
        (
            lambda: LA.vector_norm(V, axis=1.0),
            r"^vector_norm\(\): 'axis' must be an int or a tuple or list of ints, not float$",
        ),
        (lambda: LA.vecdot(V, V, axis=None), r"^vecdot\(\): 'axis' must be an int, not NoneType$"),
        (lambda: LA.cross(V, V, axis=None), r"^cross\(\): 'axis' must be an int, not NoneType$"),
        (lambda: LA.trace(A, k=None), r"^trace\(\): 'k' must be an int, not NoneType$"),
        (lambda: LA.diagonal(A, k=1.0), r"^diagonal\(\): 'k' must be an int, not float$"),
        (lambda: LA.matrix_power(A, None), r"^matrix_power\(\): 'n' must be an int, not NoneType$"),
        (
            lambda: LA.solve_banded(A, V, lower_bandwidth=None, upper_bandwidth=0),
            r"^solve_banded\(\): 'lower_bandwidth' must be an int, not NoneType$",
        ),
        (lambda: LA.lstsq(A, V, rtol="a"), r"^lstsq\(\): 'rtol' must be None, an int or a float, not str$"),
        (lambda: LA.inv(np.eye(2)), r"^inv\(\): 'x' must be an sd\.ndarray, not numpy\.ndarray$"),
    ],
)
def test_wrong_values_name_the_argument(call, message):
    with pytest.raises(TypeError, match=message):
        call()


def test_wrong_mode_value():
    with pytest.raises(ValueError, match=r"^svd\(\): 'mode' must be 'reduced' or 'complete', not 'full'$"):
        LA.svd(A, mode="full")


def test_numpy_integers_are_int_arguments():
    assert LA.matrix_power(A, np.int64(2)).tolist() == LA.matrix_power(A, 2).tolist()
    assert LA.diagonal(A, k=np.int32(1)).tolist() == [1.0]
    assert LA.trace(A, k=np.int64(-1)) == 1.0
    assert LA.vecdot(V, V, axis=np.int64(0)) == 25.0
    assert LA.cross(sd.asarray([1.0, 0, 0]), sd.asarray([0.0, 1, 0]), axis=np.int8(0)).tolist() == [0.0, 0.0, 1.0]
    assert LA.vector_norm(V, axis=(np.int64(0),)) == 5.0
    # ord: a numpy integer is the int it holds (DESIGN.md §10.3 listed vector_norm(ord=np.int64(1)))
    assert LA.vector_norm(V, ord=np.int64(1)) == LA.vector_norm(V, ord=1) == 7.0
    assert LA.matrix_norm(A, ord=np.int64(1)) == LA.matrix_norm(A, ord=1)
    assert LA.cond(A, ord=np.int64(-1)) == LA.cond(A, ord=-1)
    assert LA.matrix_rank(A, rtol=np.int64(0)) == 2
    ab = sd.asarray([[4.0, 3.0]])  # diagonal only
    assert LA.solve_banded(ab, V, lower_bandwidth=np.int64(0), upper_bandwidth=np.uint8(0)).tolist() == [0.75, 4 / 3]
    for bad, shown in ((np.True_, "numpy.bool"), (True, "bool")):
        with pytest.raises(TypeError, match=rf"^matrix_power\(\): 'n' must be an int, not {shown}$"):
            LA.matrix_power(A, bad)
        with pytest.raises(TypeError, match=rf"^vector_norm\(\): 'ord' must be an int or a float, not {shown}$"):
            LA.vector_norm(V, ord=bad)
    with pytest.raises(OverflowError, match=r"^matrix_power\(\): 'n' is out of range$"):
        LA.matrix_power(A, np.uint64(2**63))


# ------------------------------------------------- lstsq alignment, messages --


def test_messages_show_the_value_received():
    x = sd.asarray(np.eye(2))
    with pytest.raises(ValueError, match=r"^matrix_norm\(\): 'ord' must be .* or -inf, not 3$"):
        LA.matrix_norm(x, ord=3)
    with pytest.raises(ValueError, match=r"^cond\(\): 'ord' must be .* or -inf, not 'FRO'$"):
        LA.cond(x, ord="FRO")
    f = LA.lu(x)
    with pytest.raises(TypeError, match=r"^lu_solve\(\): 'perm' must be an int64 array, not float64$"):
        LA.lu_solve(f.L, f.U, sd.asarray([0.0, 1.0]), x)
