"""The common base of the regression-like results (src/stardust/smooth/_results.py; docs/api/smooth.md §1.3,
decision SM32): read-only fields and arrays, lazy fields computed once, the summary box's header and the
one-line repr. The subclasses come with milestones SM1 to SM4; a stand-in subclass checks the base here."""

import pytest

import stardust as sd
from stardust.smooth._results import SmootherResult


class _Fit(SmootherResult):
    TITLE = "Lowess Results"
    METHOD = "LOWESS"
    TUNING = (("Frac:", "frac"), ("Robust iterations:", "it"), ("Delta:", "delta"))
    REPR = ("nobs", "df", "frac", "it")


def _fit(**extra):
    calls = []

    def leverage():
        calls.append(1)
        return sd.asarray([0.5, 0.25, 0.125])

    fields = dict(
        nobs=3,
        df=1.875,
        scale=0.3125,
        frac=2 / 3,
        it=3,
        delta=0.0,
        fittedvalues=sd.asarray([1.0, 2.0, 3.0]),
        leverage=leverage,
    )
    fields.update(extra)
    return _Fit(**fields), calls


def test_fields_and_arrays_are_read_only():
    r, _ = _fit()
    with pytest.raises(ValueError, match="read-only"):
        r.fittedvalues[0] = 5.0
    with pytest.raises(AttributeError, match="read-only"):
        r.nobs = 4
    with pytest.raises(AttributeError, match="read-only"):
        del r.nobs
    with pytest.raises(AttributeError, match="has no field 'gcv'"):
        r.gcv  # noqa: B018


def test_lazy_fields_are_computed_once_and_read_only():
    r, calls = _fit()
    assert calls == []
    a = r.leverage
    b = r.leverage
    assert a is b and calls == [1]
    assert a.tolist() == [0.5, 0.25, 0.125]
    with pytest.raises(ValueError, match="read-only"):
        a[0] = 1.0
    assert "leverage" in dir(r) and "fittedvalues" in dir(r)


def test_summary_is_the_regression_box_header():
    r, _ = _fit()
    text = r.summary()
    lines = text.splitlines()
    assert lines[0].strip() == "Lowess Results"
    assert lines[1] == "=" * 78 and lines[-1] == "=" * 78
    assert len(lines) == 7  # title, rule, four header rows, rule
    assert lines[2].startswith("Method:") and lines[2].split()[1] == "LOWESS"
    assert "No. Observations:" in lines[3] and "Robust iterations:" in lines[3] and lines[3].endswith("3")
    assert "Df (effective):" in lines[4] and "1.875" in lines[4]
    assert "Scale:" in lines[5] and "0.3125" in lines[5]
    assert all(len(line) <= 78 for line in lines)


def test_repr_is_one_line():
    r, _ = _fit()
    assert repr(r) == "_Fit(nobs=3, df=1.875, frac=0.666667, it=3)"
