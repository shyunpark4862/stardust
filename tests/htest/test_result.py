"""The result base of sd.htest (docs/api/htest.md §1.5; decisions H6, H7, H10, H12, H16, H21):
read-only fields, read-only array fields without copies (API.md §2.2), absent fields as
AttributeError, scipy's tuple unpacking only where scipy's result unpacks, regression's
conf_int(*, alpha=0.05) where scipy has an interval, a short repr."""

import pytest

import stardust as sd
from stardust.htest._result import _TestResult


class TtestIndResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "df")
    _UNPACK = ("statistic", "pvalue")
    _CI = True


class Chi2ContingencyResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "df", "expected_freq")
    _UNPACK = ("statistic", "pvalue", "df", "expected_freq")


class BinomtestResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("k", "n", "alternative", "statistic", "pvalue")


class WilcoxonResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "zstatistic", "method")
    _UNPACK = ("statistic", "pvalue")


def ttest(**kw):
    fields = dict(statistic=2.5, pvalue=0.0123, df=27.0)
    fields.update(kw)
    return TtestIndResult._make(ci=lambda c: (-0.5 * c, 1.5 * c), **fields)


def test_fields_and_python_types():
    r = ttest()
    assert r.statistic == 2.5 and r.pvalue == 0.0123 and r.df == 27.0
    assert isinstance(r.df, float)
    b = BinomtestResult._make(k=3, n=10, alternative="two-sided", statistic=0.3, pvalue=0.34375)
    assert b.k == 3 and isinstance(b.k, int) and b.alternative == "two-sided"


def test_unpacks_like_scipy():
    stat, p = ttest()
    assert (stat, p) == (2.5, 0.0123)
    r = ttest()
    assert len(r) == 2 and r[0] == 2.5 and r[-1] == 0.0123 and tuple(r) == (2.5, 0.0123)
    t = Chi2ContingencyResult._make(statistic=3.8, pvalue=0.05, df=1.0, expected_freq=sd.ones((2, 2)))
    s, p, df, e = t
    assert df == 1.0 and e.shape == (2, 2)


def test_no_unpacking_where_scipy_has_none():
    b = BinomtestResult._make(k=3, n=10, alternative="two-sided", statistic=0.3, pvalue=0.34375)
    with pytest.raises(TypeError, match="cannot unpack non-iterable BinomtestResult object"):
        stat, p = b
    with pytest.raises(TypeError):
        len(b)
    with pytest.raises(TypeError):
        b[0]


def test_fields_are_read_only():
    r = ttest()
    for name in ("statistic", "pvalue", "df"):
        with pytest.raises(AttributeError, match=f"property '{name}' of 'TtestIndResult' object has no setter"):
            setattr(r, name, 1.0)
    with pytest.raises(AttributeError):
        r.foo = 1
    assert not hasattr(r, "__dict__")


def test_array_fields_are_read_only_views():
    stat = sd.asarray([1.0, 2.0, 3.0])
    r = TtestIndResult._make(batch=(3,), statistic=stat, pvalue=sd.ones(3), df=sd.full(3, 9.0), ci=lambda c: (0, 0))
    a = r.statistic
    assert r.statistic is a  # no copy at each access
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        a[0] = 9.0
    with pytest.raises(ValueError, match="read-only"):
        a[0:2][0] = 9.0
    # the public constructor copies the caller's arrays once
    src = sd.asarray([1.0, 2.0])
    r2 = TtestIndResult(batch=(2,), statistic=src, pvalue=sd.ones(2), df=sd.ones(2), ci=lambda c: (0, 0))
    src[0] = -5.0
    assert r2.statistic.tolist() == [1.0, 2.0]


def test_absent_fields():
    w = WilcoxonResult._make(statistic=10.0, pvalue=0.2, method="exact")
    assert not hasattr(w, "zstatistic")
    with pytest.raises(AttributeError, match="has no attribute 'zstatistic' \\(the test does not give it"):
        w.zstatistic  # noqa: B018
    assert w.method == "exact"
    a = WilcoxonResult._make(statistic=900.0, pvalue=0.8, zstatistic=-0.2, method="asymptotic")
    assert a.zstatistic == -0.2


def test_unknown_field_and_missing_ci():
    with pytest.raises(TypeError, match="TtestIndResult\\(\\): unexpected keyword argument 'dof'"):
        TtestIndResult._make(statistic=1.0, pvalue=0.5, dof=3.0, ci=lambda c: (0, 0))
    with pytest.raises(TypeError, match="'ci' is required"):
        TtestIndResult._make(statistic=1.0, pvalue=0.5)


def test_conf_int():
    # H16: regression's conf_int(*, alpha=0.05) (G99), the (..., 2) array of the two ends
    r = ttest()
    ci = r.conf_int()
    assert isinstance(ci, sd.ndarray) and ci.shape == (2,) and ci.tolist() == [-0.5 * 0.05, 1.5 * 0.05]
    lo, hi = r.conf_int(alpha=0.5).tolist()
    assert (lo, hi) == (-0.25, 0.75)
    ci[0] = 7.0  # a new array at each call, writable (as regression's conf_int(), G120)
    assert r.conf_int()[0] == -0.5 * 0.05
    with pytest.raises(TypeError):
        r.conf_int(0.05)  # keyword-only
    for bad in (0, 1, 1.5, -0.1, float("nan")):
        with pytest.raises(ValueError, match="conf_int\\(\\): 'alpha' must be between 0 and 1 \\(exclusive\\)"):
            r.conf_int(alpha=bad)
    for bad in (None, "0.05", True):
        with pytest.raises(TypeError, match="conf_int\\(\\): 'alpha' must be an int or a float, not"):
            r.conf_int(alpha=bad)
    assert not hasattr(r, "confidence_interval")  # scipy's form is not kept (H16)
    assert not hasattr(BinomtestResult._make(statistic=0.3, pvalue=0.3), "conf_int")


def test_batched_interval():
    r = TtestIndResult._make(
        batch=(3,),
        statistic=sd.zeros(3),
        pvalue=sd.ones(3),
        df=sd.ones(3),
        ci=lambda a: (sd.full(3, -a), sd.full(3, a)),
    )
    ci = r.conf_int(alpha=0.25)
    assert ci.shape == (3, 2) and ci[:, 0].tolist() == [-0.25] * 3 and ci[:, 1].tolist() == [0.25] * 3


def test_repr():
    assert repr(ttest()) == "TtestIndResult(statistic=2.5, pvalue=0.0123, df=27)"
    assert repr(ttest(pvalue=float("nan"))) == "TtestIndResult(statistic=2.5, pvalue=nan, df=27)"
    b = BinomtestResult._make(k=3, n=10, alternative="two-sided", statistic=0.3, pvalue=0.34375)
    assert repr(b) == "BinomtestResult(k=3, n=10, alternative='two-sided', statistic=0.3, pvalue=0.34375)"
    r = TtestIndResult._make(batch=(2, 5), statistic=sd.zeros((2, 5)), pvalue=sd.ones((2, 5)), ci=lambda c: (0, 0))
    assert repr(r) == "TtestIndResult(batch shape (2, 5))"


def test_unpack_must_name_fields():
    with pytest.raises(TypeError, match="_UNPACK names fields it does not have"):

        class Bad(_TestResult):
            __slots__ = ()
            _FIELDS = ("statistic",)
            _UNPACK = ("statistic", "pvalue")
