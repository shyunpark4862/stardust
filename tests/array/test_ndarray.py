"""dtype objects, ndarray attributes and Python protocols."""

import operator

import numpy as np
import pytest

import stardust as sd

from .conftest import run_guarded


def test_dtype_objects_are_four_distinct_singletons():
    dts = [sd.bool, sd.int64, sd.float64, sd.complex128]
    assert len({id(d) for d in dts}) == 4
    assert [repr(d) for d in dts] == ["bool", "int64", "float64", "complex128"]
    assert sd.asarray([1.0]).dtype is sd.float64


@pytest.mark.parametrize("bad", [float, int, np.float64, np.dtype("float64"), sd.string, "string", "float32"])
def test_dtype_argument_accepts_only_dtype_objects_and_names(bad):
    with pytest.raises(TypeError):
        sd.zeros(3, dtype=bad)


@pytest.mark.parametrize("name", ["bool", "int64", "float64", "complex128"])
def test_dtype_argument_accepts_names(name):
    assert sd.zeros(3, dtype=name).dtype is getattr(sd, name)


def test_ndarray_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        sd.ndarray()


def test_attributes(dtype):
    a = sd.zeros((2, 3, 4), dtype=dtype)
    item = {sd.bool: 1, sd.int64: 8, sd.float64: 8, sd.complex128: 16}[dtype]
    assert a.shape == (2, 3, 4)
    assert a.ndim == 3
    assert a.size == 24
    assert a.itemsize == item
    assert a.nbytes == 24 * item
    assert a.strides == (12 * item, 4 * item, item)
    assert a.base is None
    assert a._flags() == {"c_contiguous": True, "f_contiguous": False, "writeable": True}


@pytest.mark.parametrize("name", ["dtype", "shape", "strides", "ndim", "size", "itemsize", "nbytes", "base", "T", "mT"])
def test_attributes_are_read_only_and_say_which(name):
    a = sd.zeros((2, 2))
    with pytest.raises(AttributeError, match=f"property '{name}' of 'ndarray' object has no setter"):
        setattr(a, name, getattr(a, name))


def test_shape_one_axes_do_not_break_contiguity():
    a = sd.zeros((1, 5, 1))
    assert a._flags()["c_contiguous"] and a._flags()["f_contiguous"]


@pytest.mark.parametrize("convert", [bool, int, float, complex, operator.index])
def test_arrays_never_convert_to_scalars(convert):
    with pytest.raises(TypeError):
        convert(sd.asarray([1]))  # even a single element (docs/api/array.md §10.4)


def test_arrays_are_unhashable():
    with pytest.raises(TypeError):
        hash(sd.zeros(2))


def test_max_ndim_is_32():
    assert sd.zeros((1,) * 32).ndim == 32
    with pytest.raises(ValueError):
        sd.zeros((1,) * 33)


def test_no_reference_leaks():
    """Stands in for nanobind's (disabled) shutdown leak check."""
    import gc
    import sys

    dts = [sd.bool, sd.int64, sd.float64, sd.complex128]
    src = np.arange(6.0).reshape(2, 3)

    def work():
        for d in dts:
            a = sd.zeros((3, 2), dtype=d)
            a.dtype, a.base, repr(a), a.tolist()
            np.asarray(sd.asarray(np.asarray(a)))
        sd.asarray(src)
        try:
            sd.asarray([1j], dtype=sd.float64)
        except TypeError:
            pass

    work()
    gc.collect()
    before = [sys.getrefcount(o) for o in [*dts, src]]
    for _ in range(2000):
        work()
    gc.collect()
    assert [sys.getrefcount(o) for o in [*dts, src]] == before


def test_memory_is_cache_line_aligned():
    for n in (1, 3, 1000):
        addr = np.asarray(sd.empty(n)).__array_interface__["data"][0]
        assert addr % 128 == 0


# ---- sizes that overflow int64 -----------------------------------------------


def test_new_array_strided_checks_the_shape():
    # outer of two 32-d arrays has 33 dimensions: unchecked, it writes past the shape array.
    rc, out = run_guarded(
        """
        import stardust as sd
        x = sd.ones((1,) * 31 + (2,))
        try:
            r = sd.linalg.outer(x, x)
            print("returned", r.ndim)
        except ValueError as e:
            print(e)
        """
    )
    assert rc == 0, out
    assert out.strip() == "outer(): arrays may have at most 32 dimensions"


def test_messages_follow_the_rules():
    x = sd.ones(3)
    with pytest.raises(ValueError, match=r"^add\(\): operand of shape \(2, 2\) cannot be broadcast to \(2,\)$"):
        sd.add(sd.ones(2), sd.ones((2, 2)), out=sd.ones(2))
    with pytest.raises(ValueError, match=r"^operand of shape \(2, 3\) cannot be broadcast to \(3,\)$"):
        x += sd.ones((2, 3))
    with pytest.raises(MemoryError, match=r"^copy\(\): cannot allocate"):
        sd.broadcast_to(sd.ones(1), (2**40,)).copy()
    with pytest.raises(TypeError, match=r"^DType\(\): there is no constructor; use sd.bool"):
        type(sd.int64)()
    with pytest.raises(TypeError, match=r"^'==' not supported between instances of 'ndarray' and 'Series'$"):
        x == sd.frame.Series([1, 2])  # noqa: B015
    with pytest.raises(TypeError, match=r"^'!=' not supported between instances of 'ndarray' and 'Series'$"):
        x != sd.frame.Series([1, 2])  # noqa: B015
    with pytest.raises(TypeError, match=r"^asarray\(\): uint64 buffers cannot be converted to float64 without loss$"):
        sd.asarray(np.ones(2, dtype=np.uint64), dtype=sd.float64)
    with pytest.raises(TypeError, match=r"^asarray\(\): uint64 buffers cannot be converted to int64 without loss$"):
        sd.asarray(np.ones(2, dtype=np.uint64))
    with pytest.raises(
        TypeError, match=r"^asarray\(\): 'obj' must be an sd.ndarray, a list, a tuple, a range or a buffer, not str$"
    ):
        sd.asarray("abc")
    with pytest.raises(
        TypeError,
        match=r"^only a scalar, an sd.ndarray, a list, a tuple, a range or a buffer can be assigned, not NoneType$",
    ):
        x[...] = None
