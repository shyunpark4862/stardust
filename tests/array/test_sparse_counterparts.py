"""The array area's counterparts of sd.sparse's builders and methods (docs/api/array.md §3.7,
§4.5, §6.4; the rule of API.md §7): diag, block, deg2rad and rad2deg, and the assignment that
stands for setdiag. Each against numpy bit for bit, and against the sd.sparse function it stands
for (its toarray(), or the method on the stored values)."""

import numpy as np
import pytest

import stardust as sd

KINDS = {
    "bool": lambda r, s: r.random(s) < 0.5,
    "int64": lambda r, s: r.integers(-50, 50, s),
    "float64": lambda r, s: r.standard_normal(s),
    "complex128": lambda r, s: r.standard_normal(s) + 1j * r.standard_normal(s),
}


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == np.ascontiguousarray(b).tobytes()


# ------------------------------------------------------------------- diag --


@pytest.mark.parametrize("dt", list(KINDS))
@pytest.mark.parametrize("k", [0, 1, -2, 5, -7])
def test_diag_is_numpys(dt, k):
    x = KINDS[dt](np.random.default_rng(0), 6)
    r = sd.diag(sd.asarray(x), k=k)
    assert same_bits(r, np.diag(x, k))
    a = np.asarray(r)
    assert a.flags.c_contiguous and a.flags.writeable  # a new C-order array


def test_diag_copies_values_as_they_are():
    x = np.array([np.nan, -0.0, np.inf, -np.inf, 5e-324])
    x[0] = np.frombuffer(np.uint64(0x7FF8_0000_0000_0123).tobytes(), np.float64)[0]  # a NaN payload
    assert same_bits(sd.diag(sd.asarray(x), k=-1), np.diag(x, -1))
    assert same_bits(sd.diag(sd.asarray(x)[::-2], k=2), np.diag(x[::-2], 2))  # strided
    assert np.asarray(sd.diag(sd.asarray(np.zeros(0)), k=3)).shape == (3, 3)


def test_diag_constructs_only():
    m = sd.asarray(np.eye(3))
    msg = "diag\\(\\): 'x' must have 1 dimension, got 2; for the diagonal of a matrix use sd.linalg.diagonal"
    with pytest.raises(ValueError, match=msg):
        sd.diag(m)
    with pytest.raises(ValueError, match="'x' must have 1 dimension, got 3$"):
        sd.diag(sd.zeros((2, 2, 2)))
    with pytest.raises(TypeError, match="diag\\(\\): 'x' must be an sd.ndarray, not list"):
        sd.diag([1.0, 2.0])
    with pytest.raises(TypeError, match="for a sparse array use sd.sparse.diags_array"):
        sd.diag(sd.sparse.csr_array(m))
    with pytest.raises(TypeError, match="'k' must be an int, not float"):
        sd.diag(sd.ones(2), k=1.0)
    with pytest.raises(ValueError, match="array is too big"):
        sd.diag(sd.ones(2), k=-(2**63))
    with pytest.raises(TypeError):
        sd.diag(sd.ones(2), 1)  # k is keyword-only


@pytest.mark.parametrize("dt", list(KINDS))
def test_diag_is_diags_arrays_toarray(dt):
    x = KINDS[dt](np.random.default_rng(1), 7)
    for k in (0, 2, -3):
        want = sd.sparse.diags_array(sd.asarray(x), offsets=k).toarray()
        assert same_bits(sd.diag(sd.asarray(x), k=k), want)


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "dok"])
@pytest.mark.parametrize("k", [0, 2, -1])
def test_assignment_is_setdiag(fmt, k):
    """setdiag's counterpart is index assignment (docs/api/array.md §3.7): no in-place function."""
    rng = np.random.default_rng(2)
    dense = rng.standard_normal((4, 5))
    v = rng.standard_normal(3)
    A = sd.sparse.csr_array(sd.asarray(dense)).asformat(fmt)
    A.setdiag(sd.asarray(v), k=k)
    x = sd.asarray(dense, copy=True)
    i = sd.arange(len(v))
    x[i + max(-k, 0), i + max(k, 0)] = sd.asarray(v)
    assert same_bits(x, A.toarray())


# ------------------------------------------------------------------ block --


def test_block_is_numpys():
    rng = np.random.default_rng(3)
    f = {k: KINDS[k](rng, (2, 3)) for k in KINDS}
    g = rng.standard_normal((2, 1))
    h = rng.standard_normal((1, 4))
    v = rng.standard_normal(4)
    nested = [
        [[f["float64"], g], [h]],  # 2x2 grid with a full-width row
        [f["bool"], f["int64"]],  # promotion along the last axis
        [[f["complex128"]], [f["bool"]]],
        [v, v[:2]],  # 1-D: concat
        [[v], [h[0]]],  # 1-D rows
        [[rng.standard_normal((2, 2, 3)), rng.standard_normal((2, 2, 1))]],  # 3-D at depth 2
        [[[f["float64"]], [f["float64"]]], [[f["float64"]], [f["float64"]]]],  # depth 3: a new axis
        [[np.zeros((2, 0)), g], [np.zeros((0, 0)), np.zeros((0, 1))]],
    ]

    def to_sd(n):
        return [to_sd(x) for x in n] if isinstance(n, list) else sd.asarray(n)

    for n in nested:
        assert same_bits(sd.block(to_sd(n)), np.block(n)), n
    assert same_bits(sd.block(sd.asarray(f["int64"])), np.block(f["int64"]))  # one array: a copy
    t = rng.standard_normal((30, 40))
    assert same_bits(sd.block([[sd.asarray(t).T, sd.asarray(t.T[:, :3])]]), np.block([[t.T, t.T[:, :3]]]))  # F order


def test_block_rules():
    p, q = sd.ones((2, 3)), sd.ones((2, 1))
    assert np.asarray(sd.block(((p, q), (sd.ones((1, 4)),)))).shape == (3, 4)  # tuples too
    with pytest.raises(ValueError, match="block\\(\\): 'arrays' must not be an empty list"):
        sd.block([])
    with pytest.raises(ValueError, match="'arrays\\[1\\]' must not be an empty list"):
        sd.block([[p], []])
    msg = "nested unevenly: 'arrays\\[0\\]\\[0\\]' is at depth 2 but 'arrays\\[1\\]' at depth 1"
    with pytest.raises(ValueError, match=msg):
        sd.block([[p, q], p])
    with pytest.raises(ValueError, match="the blocks of 'arrays\\[0\\]' must have the same shape except along axis 1"):
        sd.block([[p, sd.ones((3, 1))]])
    with pytest.raises(TypeError, match="'arrays\\[1\\]' must be an sd.ndarray or a list of them, not float"):
        sd.block([p, 1.0])
    with pytest.raises(TypeError, match="not numpy.ndarray"):
        sd.block([p, np.ones((2, 1))])
    with pytest.raises(TypeError, match="for a sparse array use sd.sparse.block_array"):
        sd.block([p, sd.sparse.csr_array(q)])
    with pytest.raises(ValueError, match="at most 32 dimensions"):
        x = sd.ones(1)
        for _ in range(33):
            x = [x]
        sd.block(x)


def test_block_is_block_arrays_toarray():
    rng = np.random.default_rng(4)
    a, b, c, d = (rng.standard_normal(s) for s in ((2, 3), (2, 2), (3, 3), (3, 2)))
    a[0, 1] = b[1, 0] = 0
    blocks = [[sd.asarray(a), sd.asarray(b)], [sd.asarray(c), sd.asarray(d)]]
    assert same_bits(sd.block(blocks), sd.sparse.block_array(blocks).toarray())
    # a None block of block_array is a zeros block here
    want = sd.sparse.block_array([[blocks[0][0], None], [None, blocks[1][1]]]).toarray()
    assert same_bits(sd.block([[blocks[0][0], sd.zeros((2, 2))], [sd.zeros((3, 3)), blocks[1][1]]]), want)


# ------------------------------------------------------- deg2rad, rad2deg --


@pytest.mark.parametrize("name", ["deg2rad", "rad2deg"])
def test_angles_are_numpys(name):
    rng = np.random.default_rng(5)
    x = np.r_[rng.standard_normal(1000) * 1e3, [0.0, -0.0, np.inf, -np.inf, np.nan, 180.0, np.pi, 1e308, 5e-324]]
    f, g = getattr(sd, name), getattr(np, name)
    with np.errstate(over="ignore"):
        assert same_bits(f(sd.asarray(x)), g(x))
        assert same_bits(f(sd.asarray(x)[::3]), g(x[::3]))
    assert same_bits(f(sd.asarray(np.arange(-5, 6))), g(np.arange(-5, 6).astype(np.float64)))  # int64 -> float64
    assert np.asarray(f(sd.asarray([True, False]))).dtype == np.float64  # numpy: float16
    assert f(90) == g(90.0) and type(f(90)) is float
    out = sd.zeros(3)
    assert f(sd.asarray([1.0, 2.0, 3.0]), out=out) is out
    with pytest.raises(TypeError, match=f"{name}\\(\\): not supported for complex128"):
        f(sd.asarray([1j]))


@pytest.mark.parametrize("name", ["deg2rad", "rad2deg"])
def test_angles_are_the_sparse_methods(name):
    rng = np.random.default_rng(6)
    d = rng.standard_normal((5, 6)) * 100
    d[d < 0] = 0
    A = sd.sparse.csr_array(sd.asarray(d))
    assert same_bits(getattr(A, name)().data, getattr(sd, name)(A.data))


def test_same_bits_for_any_thread_count():
    rng = np.random.default_rng(7)
    big = sd.asarray(rng.standard_normal((700, 700)))
    calls = {
        "block": lambda: sd.block([[big, big], [big, big]]),
        "diag": lambda: sd.diag(big[0]),
        "deg2rad": lambda: sd.deg2rad(big),
        "rad2deg": lambda: sd.rad2deg(big.T),
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


# ------------------------------------------------ sparse operands' messages --


def test_asarray_of_a_sparse_array_names_toarray():
    # np.asarray(A)'s sentence (docs/api/array.md §3.2, sparse.md §2)
    for fmt in ("csr", "csc", "coo", "dok"):
        A = sd.sparse.csr_array(sd.eye(2)).asformat(fmt)
        name = type(A).__name__
        with pytest.raises(TypeError) as e:
            sd.asarray(A)
        assert str(e.value) == (
            f"asarray(): a sparse array is not converted to a dense array implicitly; use A.toarray() ({name})"
        )
        with pytest.raises(TypeError) as n:
            np.asarray(A)
        assert str(e.value) == "asarray(): " + str(n.value)


def test_concat_and_stack_of_sparse_arrays_name_the_sparse_joins():
    S = sd.sparse.csr_array(sd.asarray([[1.0, 2.0], [3.0, 4.0]]))
    head = "'arrays' must hold sd.ndarray only, not csr_array; for sparse arrays use "
    for call, name, hint in (
        (lambda: sd.concat([S, S]), "concat", "sd.sparse.hstack or sd.sparse.vstack"),
        (lambda: sd.concat((sd.ones((2, 2)), S), axis=1), "concat", "sd.sparse.hstack or sd.sparse.vstack"),
        (lambda: sd.stack([S, S]), "stack", "sd.sparse.vstack"),
    ):
        with pytest.raises(TypeError) as e:
            call()
        assert str(e.value) == f"{name}(): {head}{hint}"
    # other objects keep the plain message
    with pytest.raises(TypeError) as e:
        sd.concat([sd.ones(2), [1.0]])
    assert str(e.value) == "concat(): 'arrays' must hold sd.ndarray only, not list"
