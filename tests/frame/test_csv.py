"""read_csv (docs/api/frame.md §10).

The oracle is the rules of docs/api/frame.md §10 written out in Python: random fields
(numbers, NaN/inf, bools, strings with separators, quotes and line breaks,
quoted and unquoted empties) are written as CSV, and the oracle infers each
column's dtype and values from the same field texts.
"""

import math
import os
import re

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

NA = sd.frame.NA
EXAMPLES = int(os.environ.get("SDPD_EXAMPLES", 200))
INT_RE = re.compile(r"\A[+-]?[0-9]+\Z")
FLOAT_RE = re.compile(r"\A[+-]?(([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?|(?i:nan|inf|infinity))\Z")


def write(path, rows, sep=","):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(sep.join(field_text(v, sep) for v in row) + "\n")


def field_text(field, sep):
    text, quoted = field
    if text is None:
        return ""
    if quoted or any(c in text for c in (sep, '"', "\n", "\r")) or text == "":
        return '"' + text.replace('"', '""') + '"'
    return text


def oracle_column(fields):
    """(dtype, values) of one column by docs/api/frame.md §10.4."""
    present = [t for t, q in fields if not (t is None)]
    if not present:
        return "float64", [NA] * len(fields)
    ints = all(INT_RE.match(t) for t in present)
    if ints and all(-(2**63) <= int(t) < 2**63 for t in present):
        return "int64", [NA if t is None else int(t) for t, _ in fields]
    floats = all(FLOAT_RE.match(t) for t in present) and all(
        not INT_RE.match(t) or (-(2**63) <= int(t) < 2**63 and float(int(t)) == int(t)) for t in present)
    if floats:
        return "float64", [NA if t is None else float(t) for t, _ in fields]
    if all(t in ("True", "False") for t in present):
        return "bool", [NA if t is None else t == "True" for t, _ in fields]
    return "string", [NA if t is None else t for t, _ in fields]


def same(a, b):
    if a is NA or b is NA:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or np.float64(a).tobytes() == np.float64(b).tobytes()
    return a == b and type(a) is type(b)


FIELD = st.one_of(
    st.integers(-(2**64), 2**64).map(lambda v: (str(v), False)),
    st.floats().map(lambda v: (repr(v), False)),
    st.sampled_from(["True", "False", "nan", "-inf", "1e400", "+5", "007", ".5", " 1", "1_0", "true"]).map(lambda t: (t, False)),
    st.text(st.characters(codec="utf-8", exclude_categories=["Cs"]), max_size=6).map(lambda t: (t, True)),
    st.just((None, False)),  # NA: an unquoted empty field
    st.just(("", True)),  # a quoted empty field: the empty string
)


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_inference_and_values_follow_the_rules(tmp_path_factory, data):
    ncols = data.draw(st.integers(1, 4))
    nrows = data.draw(st.integers(0, 12))
    kinds = [data.draw(st.sampled_from(["int", "float", "bool", "any"])) for _ in range(ncols)]
    pools = {
        "int": st.integers(-(2**63), 2**63 - 1).map(lambda v: (str(v), data.draw(st.booleans()))) | st.just((None, False)),
        "float": st.floats().map(lambda v: (repr(v), False)) | st.just((None, False)),
        "bool": st.sampled_from([("True", False), ("False", True), (None, False)]),
        "any": FIELD,
    }
    columns = [data.draw(st.lists(pools[k], min_size=nrows, max_size=nrows)) for k in kinds]
    names = [f"c{j}" for j in range(ncols)]
    path = tmp_path_factory.mktemp("csv") / "t.csv"
    rows = [[(n, False) for n in names]] + [[columns[j][i] for j in range(ncols)] for i in range(nrows)]
    write(path, rows)
    df = sd.frame.read_csv(path)
    assert df.columns.to_list() == names and len(df) == nrows
    for j, k in enumerate(names):
        dtype, values = oracle_column(columns[j])
        assert df[k].dtype == dtype, (columns[j], df[k].dtype)
        assert all(same(a, b) for a, b in zip(df[k].to_list(), values)), (df[k].to_list(), values)


def test_quoting_line_ends_and_bom(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes(b'\xef\xbb\xbfk,s\r\n1,"he said ""hi"""\r\n2,""\r\n3,"a,b\nc"\r\n4,\r\n5,"12"')
    df = sd.frame.read_csv(p)
    assert df.columns.to_list() == ["k", "s"]
    assert df["s"].to_list() == ['he said "hi"', "", "a,b\nc", NA, "12"]
    assert df["k"].to_list() == [1, 2, 3, 4, 5]


def test_lone_carriage_return_is_data(tmp_path):
    # Only \r\n ends a line: a \r anywhere else is a byte of the value (docs/api/frame.md §10.2).
    p = tmp_path / "t.csv"
    p.write_bytes(b"a,b\r\nx\r,y\r\nz,w\r")
    df = sd.frame.read_csv(p)
    assert df["a"].to_list() == ["x\r", "z"]
    assert df["b"].to_list() == ["y", "w\r"]
    q = tmp_path / "q.csv"
    q.write_bytes(b'a,b\n"q"\r,z\n')  # a closing quote followed by \r, not a separator
    with pytest.raises(ValueError, match="line 2: a closing quote"):
        sd.frame.read_csv(q)


def test_quoted_numbers_are_numbers(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text('a\n"12"\n3\n')
    assert sd.frame.read_csv(p)["a"].to_list() == [12, 3]


def test_na_values_and_nan(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1.5,x\nNaN,NA\n,null\n")
    df = sd.frame.read_csv(p)
    assert math.isnan(df["a"].to_list()[1]) and df["a"].to_list()[2] is NA  # NaN is a value, empty is NA
    assert df["b"].to_list() == ["x", "NA", "null"]
    df = sd.frame.read_csv(p, na_values=["NA", "null"])
    assert df["b"].to_list() == ["x", NA, NA]


def test_blank_lines(tmp_path):
    one = tmp_path / "one.csv"
    one.write_text("a\n1\n\n3\n")
    assert sd.frame.read_csv(one)["a"].to_list() == [1, NA, 3]  # a blank line is a row with one empty field
    two = tmp_path / "two.csv"
    two.write_text("a,b\n1,2\n\n3,4\n")
    with pytest.raises(ValueError, match="line 3"):
        sd.frame.read_csv(two)


@pytest.mark.parametrize(
    "text,match",
    [
        ("a,b\n1\n", "line 2: expected 2 fields, got 1"),
        ("a,b\n1,2,3\n", "line 2: expected 2 fields, got 3"),
        ('a\n"x"y\n', "line 2"),
        ('a\nx"y\n', "line 2"),
        ('a\n"never closed\n', "line 2"),
        ("a,a\n1,2\n", "repeats"),
    ],
)
def test_malformed_files(tmp_path, text, match):
    p = tmp_path / "t.csv"
    p.write_text(text)
    with pytest.raises(ValueError, match=match):
        sd.frame.read_csv(p)


def test_options(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("k;x;y\nb;1;2.5\na;2;3.5\nc;3;4.5\n")
    df = sd.frame.read_csv(p, sep=";", usecols=["y", "k"], index_col="k", nrows=2, dtype={"y": "string"})
    assert df.columns.to_list() == ["y"] and df.index.to_list() == ["b", "a"] and df.index.name == "k"
    assert df["y"].to_list() == ["2.5", "3.5"]
    assert sd.frame.read_csv(p, sep=";", dtype="string")["x"].dtype == "string"
    q = tmp_path / "noheader.csv"
    q.write_text("1,2\n3,4\n")
    assert sd.frame.read_csv(q, header=False, names=["a", "b"])["b"].to_list() == [2, 4]
    assert sd.frame.read_csv(q, names=["p", "q"]).columns.to_list() == ["p", "q"]  # renames the header (1, 2)
    for bad, error in (
        (lambda: sd.frame.read_csv(q, header=False), ValueError),
        (lambda: sd.frame.read_csv(p, sep=";;"), ValueError),
        (lambda: sd.frame.read_csv(p, sep=";", usecols=["zz"]), KeyError),
        (lambda: sd.frame.read_csv(p, sep=";", dtype={"x": "bool"}), ValueError),
        (lambda: sd.frame.read_csv(p, sep=";", dtype={"zz": "int64"}), KeyError),
        (lambda: sd.frame.read_csv(p, sep=";", index_col="y"), ValueError),  # float labels (docs/api/frame.md §10.1)
        (lambda: sd.frame.read_csv(p, sep=";", nrows=-1), ValueError),
        (lambda: sd.frame.read_csv(p, ";"), TypeError),  # keyword-only (docs/api/frame.md §1.1)
    ):
        with pytest.raises(error):
            bad()


def test_empty_and_header_only(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError):
        sd.frame.read_csv(p)
    h = tmp_path / "h.csv"
    h.write_text("a,b\n")
    df = sd.frame.read_csv(h)
    assert df.shape == (0, 2) and df.dtypes.to_list() == ["float64", "float64"]


def test_invalid_utf8(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes(b"a\n\xff\xfe\n")
    with pytest.raises(ValueError, match="UTF-8"):
        sd.frame.read_csv(p)


def test_numbers_match_pandas(tmp_path):
    rng = np.random.default_rng(5)
    ints = rng.integers(-(10**15), 10**15, 2000)
    path = tmp_path / "n.csv"
    with open(path, "w") as f:
        f.write("i\n" + "\n".join(map(str, ints.tolist())) + "\n")
    assert sd.frame.read_csv(path)["i"].to_list() == pd.read_csv(path)["i"].tolist()
