"""An int beyond the double range for correction= is an OverflowError naming the method and
the argument, as in the array area (API.md §2)."""

import pytest
import stardust as sd


@pytest.mark.parametrize("method", ["var", "std"])
def test_series_correction_out_of_range(method):
    s = sd.frame.Series([1.0, 2.0, 3.0])
    with pytest.raises(OverflowError, match=rf"^{method}\(\): 'correction' is out of range$"):
        getattr(s, method)(correction=10**400)
    with pytest.raises(OverflowError, match=rf"^{method}\(\): 'correction' is out of range$"):
        getattr(sd, method)(sd.asarray([1.0, 2.0, 3.0]), correction=10**400)
