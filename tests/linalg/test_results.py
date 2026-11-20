import numpy as np
import pytest

import stardust as sd

"""linalg's result objects (docs/api/linalg.md §1.6): the arrays of a result tuple are read-only
(a view of a field too; a copy is the caller's), and a single-array result stays writable."""


LA = sd.linalg


def _results():
    a = sd.asarray(np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]))
    batch = sd.asarray(np.stack([np.eye(3) * 2, np.eye(3) * 3]))
    return {
        "slogdet": LA.slogdet(batch),
        "eigh": LA.eigh(a),
        "eigh batch": LA.eigh(batch),
        "svd": LA.svd(a),
        "qr": LA.qr(a),
        "qr_pivoted": LA.qr_pivoted(a),
        "lu": LA.lu(a),
        "lstsq": LA.lstsq(batch, sd.ones((2, 3, 1))),
        "eig": LA.eig(a),
        "schur": LA.schur(a),
        "hessenberg": LA.hessenberg(a),
        "ldl": LA.ldl(a),
    }


# ---------------------------------------------------- read-only result fields --


@pytest.mark.parametrize("name", list(_results()))
def test_namedtuple_array_fields_are_read_only(name):
    r = _results()[name]
    arrays = [f for f in r if isinstance(f, sd.ndarray)]
    assert arrays, name
    for f in arrays:
        before = np.asarray(f).copy()
        with pytest.raises(ValueError, match="read-only"):
            f[(0,) * f.ndim] = 7
        with pytest.raises(ValueError, match="read-only"):
            f[...] = 7  # a view of the field is read-only too
        assert np.array_equal(np.asarray(f), before)
        c = f.copy()
        c[(0,) * c.ndim] = 7  # a copy is the caller's to change


def test_single_array_results_stay_writable():
    x = LA.inv(sd.asarray(np.eye(2) * 2))
    x[0, 0] = 1.0
    s = LA.solve(sd.asarray(np.eye(2)), sd.ones(2))
    s[0] = 3.0
    assert s.tolist() == [3.0, 1.0]
