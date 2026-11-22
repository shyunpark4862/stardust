"""sd.interp under the array area's rules (docs/api/array.md §1, §8.5): sd.ndarray inputs (and a Python scalar x), bool
and int64 read as float64, the result's shape, dtype and Python scalar type, any layout of the inputs. The numpy
semantics and the accuracy are tested with the interpolate area (tests/interpolate/test_interp.py,
test_reference.py, test_errors.py, test_threads.py)."""

import numpy as np

import stardust as sd

from .._bits import bits_equal


def test_shapes_and_types():
    xp, fp = sd.asarray([0.0, 1.0, 2.0]), sd.asarray([0.0, 10.0, 20.0])
    x = sd.asarray(np.linspace(-1, 3, 24).reshape(2, 3, 4))
    got = sd.interp(x, xp, fp)
    assert got.shape == (2, 3, 4) and got.dtype == sd.float64
    assert sd.interp(sd.asarray(np.zeros((0, 3))), xp, fp).shape == (0, 3)
    assert type(sd.interp(0.5, xp, fp)) is float and sd.interp(0.5, xp, fp) == 5.0
    assert type(sd.interp(1, xp, fp)) is float and type(sd.interp(True, xp, fp)) is float
    assert type(sd.interp(0.5, xp, sd.asarray([0j, 1.0, 2.0]))) is complex
    # bool and int64 nodes, values and points are float64
    assert sd.interp(sd.asarray([0, 1, 2]), sd.asarray([0, 2, 4]), sd.asarray([True, False, True])).tolist() == [
        1.0, 0.5, 0.0]  # fmt: skip
    assert sd.interp(sd.asarray([1]), sd.asarray([0, 2]), sd.asarray([0, 3])).dtype == sd.float64
    # strided and Fortran-ordered inputs: the same values as contiguous ones
    big = sd.asarray(np.linspace(-1, 3, 60).reshape(6, 10))
    assert bits_equal(sd.interp(big.T, xp, fp), np.asarray(sd.interp(sd.asarray(np.asarray(big.T).copy()), xp, fp)))
    xps = sd.asarray(np.repeat([0.0, 1.0, 2.0], 2))[::2]
    assert bits_equal(sd.interp(big, xps, fp[::-1][::-1]), sd.interp(big, xp, fp))
