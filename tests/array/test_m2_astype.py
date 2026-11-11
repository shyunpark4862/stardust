"""astype, copy, and the assignment engine (overlap, upward-only conversion:
docs/api/array.md §1.5, §1.7)."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust as sd
from .conftest import ALL_DTYPES, NP_DTYPE, assert_same

INT64_MAX, INT64_MIN = 2**63 - 1, -(2**63)
REAL = {sd.bool, sd.int64, sd.float64}


def allowed(src, dst):
    return not (src is sd.complex128 and dst in (sd.int64, sd.float64))


@pytest.mark.parametrize("src", ALL_DTYPES, ids=str)
@pytest.mark.parametrize("dst", ALL_DTYPES, ids=str)
def test_astype_all_pairs(src, dst):
    values = np.array([[0, 1, 2], [-3, 5, 0]]).astype(NP_DTYPE[src])
    a = sd.asarray(values)
    if not allowed(src, dst):
        with pytest.raises(TypeError):
            sd.astype(a, dst)
        return
    assert_same(sd.astype(a, dst), values.astype(NP_DTYPE[dst]))


def test_float_to_int_special_values_follow_s3():
    a = sd.asarray([math.nan, math.inf, -math.inf, 1e300, -1e300, 2.9, -2.9, -0.0])
    assert sd.astype(a, sd.int64).tolist() == [0, INT64_MAX, INT64_MIN, INT64_MAX, INT64_MIN, 2, -2, 0]


def test_to_bool_is_nonzero():
    a = sd.asarray([0j, 1j, math.nan + 0j, -0.0 + 0j])
    assert sd.astype(a, sd.bool).tolist() == [False, True, True, False]


def test_astype_copy_semantics():
    a = sd.asarray([1.0, 2.0])
    assert sd.astype(a, sd.float64) is not a  # default copy=True
    assert sd.astype(a, sd.float64, copy=None) is a
    assert sd.astype(a, sd.float64, copy=False) is a
    with pytest.raises(ValueError):
        sd.astype(a, sd.int64, copy=False)
    assert sd.astype(a, sd.int64, copy=None).dtype is sd.int64


def test_astype_method_and_errors():
    a = sd.asarray([1.5])
    assert a.astype(sd.int64).tolist() == [1]
    with pytest.raises(TypeError):
        a.astype(None)
    with pytest.raises(TypeError):
        a.astype(int)
    with pytest.raises(TypeError):
        a.astype(sd.int64, False)  # copy is keyword-only


def test_astype_preserves_memory_order():
    f = sd.asarray(np.asfortranarray(np.arange(6.0).reshape(2, 3)))
    out = f.astype(sd.int64)
    assert out._flags()["f_contiguous"] and not out._flags()["c_contiguous"]
    assert out.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_copy_is_independent_and_writeable():
    src = np.arange(4.0)
    src.flags.writeable = False
    a = sd.asarray(src)
    b = a.copy()
    assert b.base is None and b._flags()["writeable"]
    assert_same(b, src)


@given(
    hnp.arrays(
        dtype=st.sampled_from([np.bool_, np.int64, np.float64, np.complex128]),
        shape=hnp.array_shapes(min_dims=1, max_dims=4, min_side=0, max_side=5),
        elements={"allow_nan": False, "allow_infinity": False, "min_value": -1e15, "max_value": 1e15},
    ),
    st.sampled_from(ALL_DTYPES),
    st.data(),
)
def test_astype_on_strided_views_matches_numpy(base, target, data):
    # Take an arbitrary strided view: reversed and/or stepped axes, maybe transposed.
    index = tuple(data.draw(st.sampled_from([slice(None), slice(None, None, -1), slice(None, None, 2)]))
                  for _ in range(base.ndim))
    view = base[index]
    if data.draw(st.booleans()):
        view = view.T
    src_dtype = {np.dtype(v): k for k, v in NP_DTYPE.items()}[view.dtype]
    if not allowed(src_dtype, target):
        return
    assert_same(sd.asarray(view).astype(target), view.astype(NP_DTYPE[target]))


# --------------------------- assignment engine (docs/api/array.md §1.5, §1.7)

copyto = sd._core._copyto


def test_overlapping_forward_copy_reads_before_writing():
    n = np.arange(10.0)
    copyto(sd.asarray(n[1:]), sd.asarray(n[:-1]))
    want = np.arange(10.0)
    want[1:] = want[:-1].copy()
    assert n.tolist() == want.tolist()


def test_overlapping_backward_copy():
    n = np.arange(10.0)
    copyto(sd.asarray(n[:-1]), sd.asarray(n[1:]))
    assert n.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]


def test_reversed_self_copy():
    n = np.arange(6.0)
    copyto(sd.asarray(n), sd.asarray(n[::-1]))
    assert n.tolist() == [5, 4, 3, 2, 1, 0]


def test_same_view_needs_no_temporary():
    a = sd.asarray([1.0, 2.0])
    copyto(a, a)
    assert a.tolist() == [1.0, 2.0]


def test_copyto_broadcasts_and_upcasts():
    dst = sd.zeros((2, 3))
    copyto(dst, sd.asarray([1, 2, 3]))  # int64 -> float64 is allowed implicitly
    assert dst.tolist() == [[1, 2, 3], [1, 2, 3]]
    with pytest.raises(ValueError):
        copyto(dst, sd.asarray([1, 2]))


def test_copyto_refuses_implicit_downcast_and_readonly():
    with pytest.raises(TypeError):
        copyto(sd.zeros(2, dtype=sd.int64), sd.asarray([1.5, 2.5]))  # docs/api/array.md §1.5
    src = np.zeros(2)
    src.flags.writeable = False
    with pytest.raises(ValueError):
        copyto(sd.asarray(src), sd.asarray([1.0, 2.0]))


def test_may_share_memory_is_a_byte_range_test():
    n = np.arange(10.0)
    share = sd._core._may_share_memory
    assert share(sd.asarray(n[:5]), sd.asarray(n[4:]))
    assert not share(sd.asarray(n[:5]), sd.asarray(n[5:]))
    # Conservative: interleaved halves never touch the same element, but their ranges overlap.
    assert share(sd.asarray(n[::2]), sd.asarray(n[1::2]))
