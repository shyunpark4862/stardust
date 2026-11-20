"""The input layer of sd.htest (docs/api/htest.md §1.2-§1.4; decisions H3, H9, H11-H13): data as
sd.stats takes it, axis=None and batches broadcast outside the sample axes, nan_policy, tables
with the batch in front, and the option checks with the library's messages."""

import math

import numpy as np
import pytest

import stardust as sd
from stardust.htest import _common as C

NAN = math.nan


def S(xs, axis=None, policy="propagate", **kw):
    return C.Samples("ttest_ind", xs, ("a", "b", "c")[: len(xs)], axis, policy, **kw)


def test_axis_none_flattens_each_sample():
    s = S([[[1, 2], [3, 4]], [5, 6, 7]])
    assert s.batch == () and s.lanes == 1 and s.n == [4, 3]
    a, b = s.lane(0)
    assert a.tolist() == [1, 2, 3, 4] and b.tolist() == [5, 6, 7]
    assert s.out([2.5]) == 2.5 and isinstance(s.out([2.5]), float)
    assert s.out([3]) == 3 and isinstance(s.out([3]), int)


def test_batches_broadcast_outside_the_sample_axis():
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(5.0).reshape(1, 5)  # broadcasts against a's 3 lanes
    s = S([a, b], axis=-1)
    assert s.batch == (3,) and s.n == [4, 5]
    for i in range(3):
        x, y = s.lane(i)
        assert x.tolist() == a[i].tolist() and y.tolist() == b[0].tolist()
    out = s.out([0.0, 1.0, 2.0])
    assert isinstance(out, sd.ndarray) and out.shape == (3,)
    # the sample axis first: a column broadcasts over the lanes
    s = S([a.T, np.arange(4.0)[:, None]], axis=0)
    assert s.batch == (3,) and s.n == [4, 4]
    assert s.lane(2)[0].tolist() == a[2].tolist() and s.lane(2)[1].tolist() == [0, 1, 2, 3]
    # fewer dimensions get leading ones (numpy's rule, scipy's too): a 1-D sample along axis 0 of a
    # 2-D one has the wrong batch length
    with pytest.raises(ValueError, match="outside the sample axes: \\(4, 3\\), \\(1, 4\\)"):
        S([a.T, np.arange(4.0)], axis=0)


def test_several_sample_axes_and_keepdims():
    x = np.arange(24.0).reshape(2, 3, 4)
    s = S([x], axis=(0, 2), keepdims=True)
    assert s.batch == (3,) and s.n == [8]
    assert sorted(s.lane(1)[0].tolist()) == sorted(x[:, 1, :].ravel().tolist())
    assert s.out([1.0, 2.0, 3.0]).shape == (1, 3, 1)
    s = S([x], axis=[2, -3])
    assert s.out([1.0, 2.0, 3.0]).shape == (3,)


def test_axis_errors():
    with pytest.raises(ValueError, match="ttest_ind\\(\\): shapes could not be broadcast together outside the sample"):
        S([np.zeros((3, 5)), np.zeros((2, 7))], axis=-1)
    with pytest.raises(ValueError, match="ttest_ind\\(\\): axis 2 is out of bounds for an array of dimension 2"):
        S([np.zeros((3, 5))], axis=2)
    with pytest.raises(ValueError, match="ttest_ind\\(\\): repeated axis"):
        S([np.zeros((3, 5))], axis=(1, -1))
    with pytest.raises(TypeError, match="ttest_ind\\(\\): 'axis' must be an int or a tuple or list of ints, not float"):
        S([np.zeros((3, 5))], axis=1.0)
    with pytest.raises(OverflowError, match="ttest_ind\\(\\): 'axis' is out of range"):
        S([np.zeros(3)], axis=2**70)


def test_paired_samples_agree_in_length():
    s = C.Samples("ttest_rel", [[1, 2, 3], [4, 5, 6]], ("a", "b"), None, "propagate", paired=True)
    assert s.n == [3, 3]
    with pytest.raises(ValueError, match="ttest_rel\\(\\): the paired samples must agree in size, not 3 and 4"):
        C.Samples("ttest_rel", [[1, 2, 3], [4, 5, 6, 7]], ("a", "b"), None, "propagate", paired=True)
    with pytest.raises(ValueError, match="agree in length along the sample axes, not 3 and 4"):
        C.Samples("ttest_rel", [np.zeros((2, 3)), np.zeros((2, 4))], ("a", "b"), -1, "propagate", paired=True)


def test_data_as_sd_stats_takes_it():
    import stardust.frame as fr

    s = S([sd.asarray([1.0, 2.0]), (3, 4), np.array([5, 6])])
    assert [x.tolist() for x in s.lane(0)] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    s = S([fr.Series([1.0, None, 3.0])], policy="omit")
    assert s.lane(0)[0].tolist() == [1.0, 3.0]
    assert S([2.5]).n == [1]  # a real scalar is one observation (sd.stats D-30)
    with pytest.raises(TypeError, match="ttest_ind\\(\\): 'a' must be an sd.ndarray, a sequence of numbers or a frame"):
        S([{"x": 1}])
    with pytest.raises(TypeError, match="ttest_ind\\(\\): 'b' must be real, not a complex128 array"):
        S([[1.0], [1j]])
    assert S([[True, False]]).lane(0)[0].dtype == sd.float64


def test_nan_policy():
    a = [[1.0, NAN, 3.0], [4.0, 5.0, 6.0]]
    b = [[1.0, 2.0, 3.0], [4.0, NAN, NAN]]
    s = S([a, b], axis=-1)
    assert s.nan.tolist() == [True, True] and s.propagates(0) and s.propagates(1)
    s = S([a, b], axis=-1, policy="omit")
    assert not s.propagates(0)
    assert [x.tolist() for x in s.lane(0)] == [[1.0, 3.0], [1.0, 2.0, 3.0]]
    assert [x.tolist() for x in s.lane(1)] == [[4.0, 5.0, 6.0], [4.0]]
    p = C.Samples("ttest_rel", [a, b], ("a", "b"), -1, "omit", paired=True)
    assert [x.tolist() for x in p.lane(0)] == [[1.0, 3.0], [1.0, 3.0]]
    assert [x.tolist() for x in p.lane(1)] == [[4.0], [4.0]]
    with pytest.raises(ValueError, match="ttest_ind\\(\\): the input contains NaN \\(nan_policy='raise'\\)"):
        S([a, b], axis=-1, policy="raise")
    assert S([[1.0, 2.0]], policy="raise").nan is None
    with pytest.raises(ValueError, match="'nan_policy' must be 'propagate', 'omit' or 'raise', not 'skip'"):
        S([a], policy="skip")
    with pytest.raises(TypeError, match="'nan_policy' must be 'propagate', 'omit' or 'raise', not NoneType"):
        S([a], policy=None)


def test_empty_batches_and_samples():
    s = S([np.zeros((0, 4))], axis=-1)
    assert s.batch == (0,) and s.lanes == 0 and s.out([]).shape == (0,)
    s = S([np.zeros((3, 0))], axis=-1)
    assert s.batch == (3,) and s.n == [0] and s.lane(0)[0].tolist() == []


def test_tables_batch_in_front():
    t = C.Tables(np.arange(24.0).reshape(2, 3, 2, 2), "fisher_exact", "table", minimum=(2, 2))
    assert t.batch == (2, 3) and t.table == (2, 2) and t.rows.shape == (6, 2, 2)
    assert t.out(list(range(6))).shape == (2, 3)
    one = C.Tables([[3, 7], [9, 2]], "fisher_exact", "table")
    assert one.batch == () and one.out([0.5]) == 0.5
    with pytest.raises(ValueError, match="fisher_exact\\(\\): 'table' must hold non-negative integer counts"):
        C.Tables([[3, -1], [9, 2]], "fisher_exact", "table")
    with pytest.raises(ValueError, match="must hold non-negative integer counts"):
        C.Tables([[3, 1.5], [9, 2]], "fisher_exact", "table")
    with pytest.raises(ValueError, match="must hold non-negative integer counts"):
        C.Tables([[3, NAN], [9, 2]], "fisher_exact", "table")
    C.Tables([[3, 1.5], [9, 2]], "chi2_contingency", "observed", integer=False)
    with pytest.raises(ValueError, match="'observed' must hold non-negative finite counts"):
        C.Tables([[3, math.inf], [9, 2]], "chi2_contingency", "observed", integer=False)
    with pytest.raises(ValueError, match="'table' must have at least 2 dimensions \\(the table's\\), not 1"):
        C.Tables([3, 7], "fisher_exact", "table")
    with pytest.raises(ValueError, match="'table' must be at least a 2 x 2 table, not 1 x 2"):
        C.Tables([[3, 7]], "fisher_exact", "table", minimum=(2, 2))


def test_alternative():
    for v in C.ALTERNATIVES:
        assert C.alternative(v, "mannwhitneyu") == v
    with pytest.raises(
        ValueError, match="mannwhitneyu\\(\\): 'alternative' must be 'two-sided', 'less' or 'greater', "
    ):
        C.alternative("larger", "mannwhitneyu")
    with pytest.raises(TypeError, match="not NoneType"):
        C.alternative(None, "mannwhitneyu")


def test_rng_is_required_and_typed():
    g = sd.random.default_rng(1)
    assert C.rng(g, "permutation_test") is g
    msg = "permutation_test\\(\\): 'rng' must be an sd.random.Generator \\(sd.random.default_rng\\(seed\\)\\), not "
    with pytest.raises(TypeError, match=msg + "NoneType"):
        C.rng(None, "permutation_test")
    with pytest.raises(TypeError, match=msg + "numpy.random.Generator"):
        C.rng(np.random.default_rng(1), "permutation_test")
    with pytest.raises(TypeError, match=msg + "int"):
        C.rng(1, "permutation_test")


def test_method_and_auto():
    assert C.method("auto", "mannwhitneyu", ("asymptotic", "exact")) == "auto"
    assert C.resolved("auto", "exact") == "exact" and C.resolved("asymptotic", "exact") == "asymptotic"
    with pytest.raises(ValueError, match="'method' must be 'auto', 'asymptotic' or 'exact', not 'Exact'"):
        C.method("Exact", "mannwhitneyu", ("asymptotic", "exact"))
    with pytest.raises(ValueError, match="'method' must be 'asymptotic' or 'exact', not 'auto'"):
        C.method("auto", "chatterjeexi", ("asymptotic", "exact"), auto=False)


def test_exact_work_cap():
    C.check_exact_work(1e9, 1e10, "ks_2samp")
    with pytest.raises(
        ValueError, match="ks_2samp\\(\\): method='exact' needs about 4.1e\\+12 operations .* cap of 1e"
    ):
        C.check_exact_work(4.1e12, 1e10, "ks_2samp")
    with pytest.raises(ValueError, match="use method='asymptotic' or a PermutationMethod"):
        C.check_exact_work(2, 1, "ansari", instead="method='asymptotic' or a PermutationMethod")


def test_scalar_options():
    assert C.count(np.int64(3), "binomtest", "k") == 3
    with pytest.raises(TypeError, match="binomtest\\(\\): 'k' must be an int, not float"):
        C.count(3.0, "binomtest", "k")
    with pytest.raises(TypeError, match="'k' must be an int, not bool"):
        C.count(True, "binomtest", "k")
    with pytest.raises(ValueError, match="binomtest\\(\\): 'k' must be >= 0"):
        C.count(-1, "binomtest", "k")
    with pytest.raises(OverflowError, match="'n' is out of range"):
        C.count(2**64, "binomtest", "n")
    assert C.real(np.int64(2), "ttest_1samp", "popmean") == 2.0
    with pytest.raises(TypeError, match="ttest_1samp\\(\\): 'popmean' must be an int or a float, not str"):
        C.real("0", "ttest_1samp", "popmean")
    assert C.flag(True, "mannwhitneyu", "use_continuity") is True
    with pytest.raises(TypeError, match="'use_continuity' must be True or False, not int"):
        C.flag(1, "mannwhitneyu", "use_continuity")
    assert C.choice("median", "levene", "center", ("median", "mean", "trimmed")) == "median"
