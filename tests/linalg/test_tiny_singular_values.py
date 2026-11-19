"""lstsq and pinv with kept singular values below 2^-1022 (2026-11-19): 1/sigma overflows,
and inf times an exact zero of the singular vectors was NaN. Exact zeros stay zero and the
entries beyond the doubles are +-inf (docs/api/linalg.md §4, docs/design/linalg.md §10.4)."""

import math

import numpy as np
import pytest
import stardust as sd

L = sd.linalg


def _a(x):
    return sd.asarray(x)


@pytest.mark.parametrize(
    "a,rtol",
    [([[1.0, 0.0], [0.0, 5e-324]], 0), ([[1.0, 0.0], [0.0, 1e-310]], 0), ([[1.0, 0.0], [0.0, 1e-310]], 1e-320)],
)
def test_diagonal(a, rtol):
    a = _a(a)
    r = L.lstsq(a, sd.ones(2), rtol=rtol)
    assert r.rank == L.matrix_rank(a, rtol=rtol) == 2
    assert r.solution.tolist() == [1.0, math.inf]  # exact [1, 2.02e323] (or 1e310)
    assert L.pinv(a, rtol=rtol).tolist() == [[1.0, 0.0], [0.0, math.inf]]


def test_finite_results_and_shapes():
    # the exact entry 1e-20 / 1e-310 = 1e290 is a double: not inf
    x = L.lstsq(_a([[1.0, 0.0], [0.0, 1e-310]]), _a([1.0, 1e-20]), rtol=0).solution.tolist()
    assert x[0] == 1.0 and x[1] == pytest.approx(1e-20 / 1e-310, rel=1e-15)
    # rectangular, several right-hand sides, a batch, complex
    assert L.lstsq(_a([[1.0, 0.0], [0.0, 5e-324], [0.0, 0.0]]), sd.ones(3), rtol=0).solution.tolist() == [1.0, math.inf]
    assert L.pinv(_a([[1.0, 0.0, 0.0], [0.0, 5e-324, 0.0]]), rtol=0).tolist() == [
        [1.0, 0.0],
        [0.0, math.inf],
        [0.0, 0.0],
    ]
    x = L.lstsq(_a([[1.0, 0.0], [0.0, 5e-324]]), _a([[1.0, 0.0], [0.0, 1.0]]), rtol=0).solution.tolist()
    assert x == [[1.0, 0.0], [0.0, math.inf]]
    p = L.pinv(_a([[[1.0, 0.0], [0.0, 5e-324]], [[2.0, 0.0], [0.0, 1.0]]]), rtol=0).tolist()
    assert p == [[[1.0, 0.0], [0.0, math.inf]], [[0.5, 0.0], [0.0, 1.0]]]
    c = _a([[1.0 + 0j, 0.0], [0.0, 5e-324]])
    assert L.lstsq(c, sd.ones(2), rtol=0).solution.tolist() == [1.0, complex(math.inf, 0.0)]
    assert L.pinv(c, rtol=0).tolist() == [[1.0, 0.0], [0.0, complex(math.inf, 0.0)]]


def test_rotated():
    # R diag(1, 1e-310): pinv = diag(1, 1e310) R^T, the second row beyond the doubles
    a = _a([[0.6, -0.8e-310], [0.8, 0.6e-310]])
    p = L.pinv(a, rtol=0).tolist()
    assert p[0] == pytest.approx([0.6, 0.8], rel=1e-15) and p[1] == [-math.inf, math.inf]
    x = L.lstsq(a, sd.ones(2), rtol=0).solution.tolist()
    assert x[0] == pytest.approx(1.4, rel=1e-15) and x[1] == -math.inf


@pytest.mark.parametrize("rtol", [None, 1e-300, 1e-12])
def test_gelsd_path_with_tiny_values(rtol):
    # rtol >= eps is gelsd (scales a tiny matrix itself; numpy's values), 1e-300 the SVD path
    a = _a([[5e-324, 0.0], [0.0, 5e-324]])
    r = L.lstsq(a, _a([1.0, 0.0]), rtol=rtol)
    assert r.rank == 2 and r.solution.tolist() == [math.inf, 0.0]
    assert L.pinv(a, rtol=rtol).tolist() == [[math.inf, 0.0], [0.0, math.inf]]
    x = L.lstsq(_a([[1e-300, 0.0], [0.0, 1e-310]]), _a([1.0, 1e-20]), rtol=1e-12).solution.tolist()
    assert x == pytest.approx([1e300, 1e-20 / 1e-310], rel=1e-15)


def test_ordinary_matrices_unchanged():
    # no tiny kept value: the same path as before (the numpy values to rounding)
    rng = np.random.default_rng(5)
    a = rng.standard_normal((6, 4))
    b = rng.standard_normal(6)
    for rtol in (0, None):
        x = np.asarray(L.lstsq(_a(a), _a(b), rtol=rtol).solution.tolist())
        assert np.allclose(x, np.linalg.lstsq(a, b, rcond=None)[0], rtol=1e-13, atol=0)
        assert np.allclose(np.asarray(L.pinv(_a(a), rtol=rtol).tolist()), np.linalg.pinv(a), rtol=1e-12, atol=1e-14)
