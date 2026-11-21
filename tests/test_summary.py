"""The shared text box of every result's summary() (stardust._summary; API.md §2.3, regression
G189, htest H74, optimize O119): its widths, the header pairs, the tables' columns and
alignment, the cut of long tables, the number formats and the interval's column names."""

import math

import pytest

import stardust as sd
from stardust import _summary as S


def test_box_widths_and_rules():
    text = S.render(
        "A title",
        header=([("Left:", "1"), ("Second:", "abc")], [("Right:", "2.50")]),
        tables=[S.Table(["coef", "std err"], [("x1", ["1.0000", "0.100"])])],
        notes=["A note."],
    )
    lines = text.splitlines()
    assert lines[0] == "A title".center(78).rstrip()
    assert lines[1] == "=" * 78 and lines.count("=" * 78) == 3 and "-" * 78 in lines
    # each header cell is 38 wide, the value right-aligned, two spaces between the columns
    assert lines[2] == "Left:" + "1".rjust(33) + "  " + "Right:" + "2.50".rjust(32)
    assert lines[3] == "Second:" + "abc".rjust(31)
    # the stub is at least 10 wide, every column 11
    assert lines[5] == " " * 10 + "coef".rjust(11) + "std err".rjust(11)
    assert lines[7] == "x1".ljust(10) + "1.0000".rjust(11) + "0.100".rjust(11)
    assert lines[-3:] == ["", "Notes:", "[1] A note."]
    assert all(line == line.rstrip() for line in lines)


def test_wide_table_widens_rules_not_title_or_pairs():
    cols = [f"c{j}" for j in range(8)]
    t = S.Table(cols, [("row", ["1.0"] * 8)])
    assert t.width() == 10 + 8 * 11 == 98
    text = S.render("T", header=([("A:", "1")], []), tables=[t], notes=["n " * 60])
    lines = text.splitlines()
    assert lines[0] == "T".center(78).rstrip()
    assert lines[1] == "=" * 98 and lines[2] == "A:" + "1".rjust(36)
    # the notes wrap at the table's width
    assert max(len(line) for line in lines[lines.index("Notes:") + 1 :]) <= 98


def test_columns_grow_with_their_cells_and_left_alignment():
    t = S.Table(
        ["value", "names"], [("0", ["1.5", "x1, x2, x3, x4"]), ("10", ["2.0", ""])], left=(1,), stub_head="size"
    )
    head, body = t.lines()
    assert head == ["size".ljust(10) + "value".rjust(11) + "  names"]
    assert body[0] == "0".ljust(10) + "1.5".rjust(11) + "  x1, x2, x3, x4"
    assert body[1] == "10".ljust(10) + "2.0".rjust(11)  # trailing blanks stripped
    long = S.Table(["x"], [("a", ["-1.234567890123e+300"])])
    assert long.width() == 10 + 21


def test_without_header_the_table_follows_the_title():
    text = S.render("Test for Constraints", tables=[S.Table(["coef"], [("c0", ["1.0000"])])])
    assert text.splitlines()[1:3] == ["=" * 78, " " * 10 + "coef".rjust(11)]


def test_long_tables_keep_both_ends():
    assert S.shown(20) == list(range(20))
    assert S.shown(21) == [0, 1, 2, 3, 4, None, 16, 17, 18, 19, 20]
    rows = S.lane_rows((7, 4), [(lambda i: float(i), S.forg)])
    assert len(rows) == 11 and rows[5] is None
    assert rows[0] == ("[0, 0]", ["0.000"]) and rows[-1] == ("[6, 3]", ["27.000"])
    head, body = S.Table(["v"], rows).lines()
    assert body[5] == "...".ljust(10) + "...".rjust(11)
    assert S.lane_label(5, (6,)) == "[5]" and S.shape_text((6,)) == "(6,)" and S.shape_text((2, 3)) == "(2, 3)"


def test_number_formats():
    assert S.forg(1.23456, 4) == "1.2346" and S.forg(12345.678) == "1.23e+04" and S.forg(0.00001234) == "1.23e-05"
    assert S.forg(0.0) == "0.000" and S.forg(math.nan) == "nan" and S.forg(-math.inf) == "-inf"
    assert S.count(60.0) == "60" and S.count(math.nan) == "nan" and S.count(math.inf) == "inf"
    assert S.df(47.0) == "47" and S.df(45.3218) == "45.32" and S.df(math.inf) == "inf"
    assert S.stat(2.0) == "2.000" and S.prob(0.0012345) == "0.00123" and S.pvalue(0.0012345) == "0.001"
    assert S.sizes(25) == "25" and S.sizes((25, 24)) == "25, 24" and S.sizes((1, 2, 3, 4, 5)) == "15"


def test_interval_columns():
    assert S.interval_columns(0.05) == ["[0.025", "0.975]"]
    assert S.interval_columns(0.1) == ["[0.050", "0.950]"]
    assert S.interval_columns(0.05, "less") == ["[0.000", "0.950]"]
    assert S.interval_columns(0.05, "greater") == ["[0.050", "1.000]"]
    assert S.interval_columns(0.0123) == ["[0.00615", "0.99385]"]


def test_wide_rows_fit_or_wrap():
    assert S.wide("Message:", "done") == ["Message:" + "done".rjust(70)]
    long = S.wide("Message:", "word " * 30)
    assert len(long) == 3 and long[1].startswith("    word") and all(len(line) <= 78 for line in long)


def test_compact_repr_values():
    assert S.compact(0.123456789) == "0.123457" and S.compact(math.nan) == "nan" and S.compact(3) == "3"
    assert S.compact(sd.asarray([1.0, 2.5])) == "[1, 2.5]"
    assert S.compact(sd.zeros((7,))) == "<array of shape (7,)>"
    assert S.compact(sd.zeros((2, 3))) == "<array of shape (2, 3)>"
    assert S.compact(object()) is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: sd.regression.ols([1.0, 2.0, 4.0, 3.0], [[0.0], [1.0], [2.0], [3.0]]).summary(alpha=1.0),
        lambda: sd.htest.ttest_1samp([1.0, 2.0, 4.0], 0.0).summary(alpha=0.0),
        lambda: (
            sd.regression.ols([1.0, 2.0, 4.0, 3.0], [[0.0], [1.0], [2.0], [3.0]]).t_test([0.0, 1.0]).summary(alpha=2)
        ),
    ],
)
def test_alpha_is_checked_as_conf_int(call):
    with pytest.raises(ValueError, match=r"^summary\(\): 'alpha' must be between 0 and 1 \(exclusive\)$"):
        call()


def test_alpha_only_where_there_is_an_interval():
    with pytest.raises(TypeError, match=r"^summary\(\): unexpected keyword argument 'alpha'$"):
        sd.htest.kruskal([1.0, 2.0, 3.0], [4.0, 5.0]).summary(alpha=0.1)
    with pytest.raises(TypeError, match=r"^summary\(\): unexpected keyword argument 'alpha'$"):
        sd.optimize.root_scalar(lambda x: x - 1.0, bracket=(0.0, 3.0)).summary(alpha=0.1)
    with pytest.raises(TypeError, match=r"^summary\(\): takes 0 positional arguments but 1 was given$"):
        sd.htest.ttest_1samp([1.0, 2.0, 4.0], 0.0).summary(0.1)


def test_empty_batch_with_per_lane_names():
    """A batch of no lanes whose result names each lane's method (htest H48): an empty table, not
    an IndexError."""
    r = sd.htest.mannwhitneyu(sd.zeros((0, 5)), sd.zeros((0, 4)), axis=1)
    text = r.summary()
    assert "Batch shape:                      (0,)" in text
    assert "statistic" in text and "Method" not in text


def test_table_fields_by_name():
    """Table fields are named by the result class (htest's _TABLES), not guessed from their size:
    one test's 1 x 1 table is a table (not a header number), a batch's tables are left out (not a
    column of the lanes); other areas' boxes are unchanged."""
    one = sd.htest.chi2_contingency([[5]]).summary()
    assert "Expected freq:" not in one and "expected_freq" in one.splitlines()[-4]
    batch = sd.htest.chi2_contingency(sd.ones((3, 1, 1))).summary()
    assert "expected_freq" not in batch and "Batch shape:                      (3,)" in batch
    t = sd.htest.proportions_chisquare([3, 5], [10, 12]).summary()
    assert "table" in t and "expected_freq" in t
    # a per-lane array field of one test kept as an array (keepdims) is still the header's
    k = sd.htest.ttest_1samp([1.0, 2, 4], 0.0, keepdims=True).summary()
    assert "Statistic:" in k
