"""Shared helpers of the frame tests."""

import stardust as sd
from tests._bits import bits_equal_any_nan

NA = sd.frame.NA


def same_cell(x, y):
    """One cell against its expected value: NA only matches NA, two floats by bits_equal_any_nan
    (any NaN matches any NaN, -0.0 differs from 0.0), anything else by == with the same type."""
    if x is NA or y is NA:
        return x is y
    if isinstance(x, float) or isinstance(y, float):
        return isinstance(x, float) and isinstance(y, float) and bits_equal_any_nan(x, y)
    return x == y and type(x) is type(y)
