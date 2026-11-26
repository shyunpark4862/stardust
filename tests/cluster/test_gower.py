"""Gower's distance (src/stardust/cluster/_gower.py, csrc/cluster/metrics.cpp; docs/api/cluster.md §6;
docs/design/cluster.md §2.6; decisions CL38, CL89, CL90): R cluster's daisy(metric = 'gower') on mixed frames, the
exact rationals of the definition, the column kinds and types, missing values, weights, the input forms, the thread
counts and the errors."""

import math
from fractions import Fraction

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata as D
from . import samples as S

CL = sd.cluster
F = sd.frame


def _np(a):
    return np.asarray(a)


def _frame():
    return F.DataFrame({"a": [1.0, 2.0, 3.0, None, 7.5, 2.0], "b": ["u", "v", "u", "w", None, "v"],
                        "c": [True, False, True, True, False, None], "d": [1, 0, 0, 1, 0, 0],
                        "e": [3, 1, 2, 2, 5, 1], "f": [0.5, 2.0, 8.0, 1.0, 1.0, 4.0]})  # fmt: skip


def _exact(rows, kinds, weights):
    """The definition in exact rationals: rows of values (None missing), kinds per column ('I', 'N', 'S', 'A')."""
    p = len(kinds)
    rng = []
    for j in range(p):
        vals = [Fraction(r[j]) for r in rows if r[j] is not None and kinds[j] == "I"]
        r = max(vals) - min(vals) if vals else Fraction(1)
        rng.append(r if r else Fraction(1))
    out = []
    n = len(rows)
    for i in range(n):
        for k in range(i + 1, n):
            num = den = Fraction(0)
            for j in range(p):
                x, y = rows[i][j], rows[k][j]
                if x is None or y is None:
                    continue
                w = Fraction(weights[j])
                if kinds[j] == "A" and x == 0 and y == 0:
                    continue
                den += w
                if kinds[j] == "I":
                    num += w * abs(Fraction(x) - Fraction(y)) / rng[j]
                elif x != y:
                    num += w
            out.append(num / den if den else math.nan)
    return out


def test_r_daisy():
    rec = D.data("r")
    got = _np(CL.gower_distance(_frame(), types={"d": "asymm", "c": "symm", "e": "ordratio", "f": "logratio"},
                                weights=[1, 2, 1, 0.5, 1, 3]))  # fmt: skip
    assert np.allclose(got, rec["cluster/daisy_mixed"]["value"], rtol=4e-16, atol=0)
    # R makes a logical column asymmetric (with a warning); sd's bool column is symmetric unless types says so
    want = rec["cluster/daisy_default"]["value"]
    assert np.allclose(_np(CL.gower_distance(_frame(), types={"c": "asymm"})), want, rtol=4e-16, atol=0)
    assert not np.allclose(_np(CL.gower_distance(_frame())), want, rtol=1e-6)
    X = S.points("blobs/k3p2").X
    assert np.allclose(_np(CL.gower_distance(X)), rec["cluster/daisy_numeric"]["value"], rtol=1e-15, atol=1e-16)
    assert "asymm" in rec["cluster/daisy_gower"]["warnings"]
    df = F.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["u", "v", "u"], "c": [True, False, True]})
    assert np.allclose(
        _np(CL.gower_distance(df, types={"c": "asymm"})), rec["cluster/daisy_gower"]["value"], rtol=1e-15
    )


def test_the_exact_definition():
    rows = [[1.0, "u", 1, 1, 0.5], [2.0, "v", 0, 0, 2.0], [None, "u", 1, 0, 8.0], [7.25, None, 1, 1, 1.0],
            [2.0, "v", None, 0, 4.0]]  # fmt: skip
    df = F.DataFrame({"a": [r[0] for r in rows], "b": [r[1] for r in rows], "c": [r[2] for r in rows],
                      "d": [r[3] for r in rows], "e": [r[4] for r in rows]})  # fmt: skip
    w = [1.0, 0.25, 2.0, 1.0, 3.0]
    got = _np(CL.gower_distance(df, types={"c": "symm", "d": "asymm"}, weights=w))
    want = _exact([[r[0], r[1], r[2], r[3], r[4]] for r in rows], ["I", "N", "S", "A", "I"], w)
    for g, x in zip(got, want):
        assert g == pytest.approx(float(x), rel=4 * D.U, abs=0), (g, x)
    # ordratio: the dense ranks of the values, then interval; R's ordered factor is its codes as an interval column
    df = F.DataFrame({"r": [10.0, 30.0, 20.0, 30.0]})
    assert np.allclose(_np(CL.gower_distance(df, types={"r": "ordratio"})), [1, 0.5, 1, 0.5, 0, 0.5], rtol=1e-15)
    assert np.allclose(_np(CL.gower_distance(df)), [1, 0.5, 1, 0.5, 0, 0.5], rtol=1e-15)
    df = F.DataFrame({"s": ["lo", "hi", "mid"]})  # strings rank in sorted order: hi 1, lo 2, mid 3
    assert np.allclose(_np(CL.gower_distance(df, types={"s": "ordratio"})), [0.5, 0.5, 1], rtol=1e-15)


def test_missing_constant_and_weights():
    df = F.DataFrame({"a": [None, 1.0, 2.0], "b": [5.0, None, 5.0]})
    got = _np(CL.gower_distance(df))
    assert math.isnan(got[0]) and got[1] == 0.0 and got[2] == 1.0  # no variable in common: NaN
    assert np.all(_np(CL.gower_distance(F.DataFrame({"a": [3.0, 3.0, 3.0]}))) == 0.0)  # constant: range 1
    assert np.all(np.isnan(_np(CL.gower_distance(np.array([[0.0], [1.0]]), weights=[0.0]))))
    a = np.array([[0.0, 1.0], [1.0, 3.0], [4.0, 0.0]])
    assert bits_equal(_np(CL.gower_distance(a, weights=[2.0, 2.0])), _np(CL.gower_distance(a)))


def test_input_forms_and_threads():
    X = np.random.default_rng(1).normal(size=(500, 4))
    a = _np(CL.gower_distance(X))
    assert bits_equal(a, _np(CL.gower_distance(sd.asarray(X))))
    assert bits_equal(a, _np(CL.gower_distance(F.DataFrame({str(j): X[:, j].tolist() for j in range(4)}))))
    Xp = np.abs(X) + 0.5
    logged = Xp.copy()
    logged[:, 1] = np.log10(Xp[:, 1])
    assert bits_equal(_np(CL.gower_distance(Xp, types={1: "logratio"})), _np(CL.gower_distance(logged)))
    out = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        out.append(_np(CL.gower_distance(X)))
    sd.set_num_threads(10)
    assert bits_equal(out[0], out[1]) and bits_equal(out[0], out[2])
    # the result feeds the functions that take distances
    Z = CL.linkage(CL.gower_distance(X[:50]), method="average")
    assert _np(Z).shape == (49, 4)
    assert CL.kmedoids(CL.gower_distance(X[:50]), 3, metric="precomputed").n_clusters == 3
    assert _np(CL.gower_distance(np.zeros((1, 2)))).shape == (0,)


def test_errors():
    df = _frame()
    with pytest.raises(ValueError, match=r"^gower_distance\(\): 'types' names a column 'z' that data does not have$"):
        CL.gower_distance(df, types={"z": "symm"})
    with pytest.raises(
        ValueError,
        match=r"^gower_distance\(\): the types must be 'asymm', 'symm', 'ordratio' or 'logratio', not 'ordinal'$",
    ):
        CL.gower_distance(df, types={"a": "ordinal"})
    with pytest.raises(ValueError, match=r"^gower_distance\(\): the binary column 'e' must hold 0, 1"):
        CL.gower_distance(df, types={"e": "asymm"})
    with pytest.raises(ValueError, match=r"^gower_distance\(\): the logratio column 'b' must hold numbers$"):
        CL.gower_distance(df, types={"b": "logratio"})
    with pytest.raises(ValueError, match=r"^gower_distance\(\): the logratio column 'd' must hold values > 0$"):
        CL.gower_distance(df, types={"d": "logratio"})
    with pytest.raises(ValueError, match=r"^gower_distance\(\): 'weights' must have one value per column \(6\)$"):
        CL.gower_distance(df, weights=[1.0, 2.0])
    with pytest.raises(ValueError, match=r"^gower_distance\(\): 'weights' must be finite and >= 0$"):
        CL.gower_distance(df, weights=[1, 1, 1, 1, 1, -1])
    with pytest.raises(TypeError, match=r"^gower_distance\(\): 'types' must be a dict or None, not list$"):
        CL.gower_distance(df, types=["symm"])
    with pytest.raises(ValueError, match=r"^gower_distance\(\): an array 'data' must be 2-D, not 1-D$"):
        CL.gower_distance([1.0, 2.0])
    with pytest.raises(ValueError, match=r"^gower_distance\(\): column 0 contains infinity$"):
        CL.gower_distance(np.array([[1.0], [math.inf]]))
