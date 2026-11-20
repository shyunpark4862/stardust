"""The element-wise functions on vForce, pow and the complex modulus against their Arb
references (tests/array/ref, tools/array_ref.py): every point within the function's ULP_BOUND
of the correctly rounded value (docs/api/array.md), judged by tests/array/refdata.py as the
accuracy measurement (bench/accuracy/array.py) judges it."""

import numpy as np
import pytest

import stardust as sd

from . import refdata


def _values(name, inputs):
    args = list(inputs.values())
    if name == "cabs":
        return np.asarray(sd.abs(sd.asarray(args[0] + 1j * args[1])))
    return np.asarray(getattr(sd, name)(*(sd.asarray(a) for a in args)))


@pytest.mark.parametrize("name", refdata.ELEMENTWISE)
def test_elementwise_contract(name):
    inputs, hi, lo = refdata.elementwise(name)
    assert hi.size, name
    with np.errstate(all="ignore"):
        err, bound = refdata.elementwise_contract(name, _values(name, inputs), hi)
    i = int(np.argmax(err))
    assert err[i] <= bound, (name, {k: float(v[i]) for k, v in inputs.items()}, float(err[i]))
