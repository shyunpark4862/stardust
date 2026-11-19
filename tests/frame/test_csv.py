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
    # an int beyond int64 or 2**53 is float syntax: rounded to float64, as list inference (DESIGN.md §10.1 #3)
    if all(FLOAT_RE.match(t) for t in present):
        return "float64", [NA if t is None else float(int(t)) if INT_RE.match(t) else float(t) for t, _ in fields]
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
    st.sampled_from(["True", "False", "nan", "-inf", "1e400", "+5", "007", ".5", " 1", "1_0", "true"]).map(
        lambda t: (t, False)
    ),
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
        "int": st.integers(-(2**63), 2**63 - 1).map(lambda v: (str(v), data.draw(st.booleans())))
        | st.just((None, False)),
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
    assert df.columns.tolist() == names and len(df) == nrows
    for j, k in enumerate(names):
        dtype, values = oracle_column(columns[j])
        assert str(df[k].dtype) == dtype, (columns[j], df[k].dtype)
        assert all(same(a, b) for a, b in zip(df[k].tolist(), values)), (df[k].tolist(), values)


def test_quoting_line_ends_and_bom(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes(b'\xef\xbb\xbfk,s\r\n1,"he said ""hi"""\r\n2,""\r\n3,"a,b\nc"\r\n4,\r\n5,"12"')
    df = sd.frame.read_csv(p)
    assert df.columns.tolist() == ["k", "s"]
    assert df["s"].tolist() == ['he said "hi"', "", "a,b\nc", NA, "12"]
    assert df["k"].tolist() == [1, 2, 3, 4, 5]


def test_lone_carriage_return_is_data(tmp_path):
    # Only \r\n ends a line: a \r anywhere else is a byte of the value (docs/api/frame.md §10.2).
    p = tmp_path / "t.csv"
    p.write_bytes(b"a,b\r\nx\r,y\r\nz,w\r")
    df = sd.frame.read_csv(p)
    assert df["a"].tolist() == ["x\r", "z"]
    assert df["b"].tolist() == ["y", "w\r"]
    q = tmp_path / "q.csv"
    q.write_bytes(b'a,b\n"q"\r,z\n')  # a closing quote followed by \r, not a separator
    with pytest.raises(ValueError, match="line 2: a closing quote"):
        sd.frame.read_csv(q)


def test_quoted_numbers_are_numbers(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text('a\n"12"\n3\n')
    assert sd.frame.read_csv(p)["a"].tolist() == [12, 3]


def test_na_values_and_nan(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1.5,x\nNaN,NA\n,null\n")
    df = sd.frame.read_csv(p)
    assert math.isnan(df["a"].tolist()[1]) and df["a"].tolist()[2] is NA  # NaN is a value, empty is NA
    assert df["b"].tolist() == ["x", "NA", "null"]
    df = sd.frame.read_csv(p, na_values=["NA", "null"])
    assert df["b"].tolist() == ["x", NA, NA]


def test_blank_lines(tmp_path):
    one = tmp_path / "one.csv"
    one.write_text("a\n1\n\n3\n")
    assert sd.frame.read_csv(one)["a"].tolist() == [1, NA, 3]  # a blank line is a row with one empty field
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


@pytest.mark.parametrize("text", ['a,b\n1,x"y\n', 'a,b\n1,x"y"z\n', 'a,b\n1,x"y\n2,3\n', 'a,b\n1,x"\n'])
def test_a_stray_quote_has_one_message(tmp_path, text):
    # the same mistake, wherever it is and however many quotes follow (DESIGN.md §8.4, §10.3)
    p = tmp_path / "t.csv"
    p.write_text(text)
    with pytest.raises(ValueError, match=r"line 2: a quote inside an unquoted field$"):
        sd.frame.read_csv(p)


def test_an_open_quote_is_never_closed(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text('a,b\n1,"y\n2,3\n')
    with pytest.raises(ValueError, match=r"line 2: a quoted field is never closed$"):
        sd.frame.read_csv(p)


def test_path_must_be_a_path():
    for bad, shown in ((1, "int"), (None, "NoneType"), (["a.csv"], "list")):
        with pytest.raises(
            TypeError, match=rf"^read_csv\(\): 'path' must be a str or a path-like object, not {shown}$"
        ):
            sd.frame.read_csv(bad)


@pytest.mark.parametrize(
    "values",
    [
        [2**53 + 1, 0.5],
        [2**63, 1],
        [-(2**63) - 1, 2],
        [2**70, 3.25, None],
        [12345678901234567890123, 1.0],
        [2**63 - 1, 1.5],
        [-(2**63), 0.5],
        [9007199254740993, 9007199254740995, 1e300],
    ],
)
def test_big_ints_round_as_list_inference(tmp_path, values):
    # an int beyond int64, or beyond 2**53 next to a float, makes a float64 column, rounded as
    # Series(list) rounds it (DESIGN.md §10.1 #3, docs/api/frame.md §10.4); it used to stay a string
    p = tmp_path / "t.csv"
    p.write_text(
        "x\n" + "".join(("" if v is None else repr(v) if isinstance(v, float) else str(v)) + "\n" for v in values)
    )
    got = sd.frame.read_csv(p)["x"]
    want = sd.frame.Series([float(v) if isinstance(v, int) and not -(2**63) <= v < 2**63 else v for v in values])
    assert got.dtype is sd.float64 and want.dtype is sd.float64
    assert [np.float64(v).tobytes() if v is not NA else None for v in got.tolist()] == [
        np.float64(v).tobytes() if v is not NA else None for v in want.tolist()
    ]


def test_options(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("k;x;y\nb;1;2.5\na;2;3.5\nc;3;4.5\n")
    df = sd.frame.read_csv(p, sep=";", usecols=["y", "k"], index_col="k", nrows=2, dtype={"y": "string"})
    assert df.columns.tolist() == ["y"] and df.index.tolist() == ["b", "a"] and df.index.name == "k"
    assert df["y"].tolist() == ["2.5", "3.5"]
    assert sd.frame.read_csv(p, sep=";", dtype="string")["x"].dtype is sd.string
    q = tmp_path / "noheader.csv"
    q.write_text("1,2\n3,4\n")
    assert sd.frame.read_csv(q, header=False, names=["a", "b"])["b"].tolist() == [2, 4]
    assert sd.frame.read_csv(q, names=["p", "q"]).columns.tolist() == ["p", "q"]  # renames the header (1, 2)
    for bad, error in (
        (lambda: sd.frame.read_csv(q, header=False), ValueError),
        (lambda: sd.frame.read_csv(p, sep=";;"), ValueError),
        (lambda: sd.frame.read_csv(p, sep=";", usecols=["zz"]), KeyError),
        (lambda: sd.frame.read_csv(p, sep=";", dtype={"x": "bool"}), ValueError),
        (lambda: sd.frame.read_csv(p, sep=";", dtype={"zz": "int64"}), KeyError),
        (lambda: sd.frame.read_csv(p, sep=";", index_col="y"), ValueError),  # float labels (docs/api/frame.md §10.1)
        (lambda: sd.frame.read_csv(p, sep=";", nrows=-1), ValueError),
        (lambda: sd.frame.read_csv(p, sep=";", nrows=True), TypeError),  # an int argument (DESIGN.md §10.1 #4b)
        (lambda: sd.frame.read_csv(p, sep=";", nrows=2.0), TypeError),
        (lambda: sd.frame.read_csv(p, sep=";", nrows=2**64), OverflowError),
        (lambda: sd.frame.read_csv(p, sep=0), TypeError),  # a string option (DESIGN.md §10.1 #4c)
        (lambda: sd.frame.read_csv(p, sep=";", header=1), TypeError),
        (lambda: sd.frame.read_csv(p, sep=";", index_col=0), TypeError),
        (lambda: sd.frame.read_csv(p, sep=";", usecols=(c for c in "k")), TypeError),  # no generators
        (lambda: sd.frame.read_csv(p, sep=";", usecols="k"), TypeError),
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
    assert df.shape == (0, 2) and [str(d) for d in df.dtypes.values()] == ["float64", "float64"]


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
    assert sd.frame.read_csv(path)["i"].tolist() == pd.read_csv(path)["i"].tolist()


def test_sep_must_be_ascii(tmp_path):  # docs/api/frame.md §10.1: the parser splits on one byte
    p = tmp_path / "t.csv"
    p.write_text("a·b\n1·2\n", encoding="utf-8")
    for sep in ("·", "é", "→", "\x80"):
        with pytest.raises(ValueError, match=r"^read_csv\(\): 'sep' must be a single ASCII character, not "):
            sd.frame.read_csv(p, sep=sep)
    with pytest.raises(ValueError, match=r"^read_csv\(\): 'sep' must be one character other than"):
        sd.frame.read_csv(p, sep="··")
    q = tmp_path / "u.csv"
    q.write_bytes(b"a\x7fb\n1\x7f2\n")
    assert sd.frame.read_csv(q, sep="\x7f")["b"].tolist() == [2]


@pytest.mark.parametrize("tail", [b"", b"\n", b"\r\n"])
@pytest.mark.parametrize("nrows", [1, 2, 9, 31, 32, 33, 200])
def test_nul_separator_does_not_split_the_padding_of_the_last_block(tmp_path, tail, nrows):
    # The tokenizer reads 64-byte blocks and pads the last one with zero
    # bytes; with sep="\0" the padding must not count as separators.
    p = tmp_path / "t.csv"
    p.write_bytes(b"a\0b\n" + b"\n".join(b"%d\0%d" % (i, i * 2) for i in range(nrows)) + tail)
    df = sd.frame.read_csv(p, sep="\0")
    assert df["a"].tolist() == list(range(nrows)) and df["b"].tolist() == [i * 2 for i in range(nrows)]
    one = tmp_path / "one.csv"
    one.write_bytes(b"a\n" + b"\n".join(b"%d" % i for i in range(nrows)) + tail)
    assert sd.frame.read_csv(one, sep="\0")["a"].tolist() == list(range(nrows))


def test_na_values_apply_to_quoted_fields(tmp_path):  # docs/api/frame.md §10.3
    p = tmp_path / "t.csv"
    p.write_text('a,b,c\n"NA",x,"1"\nNA,"NA",2\n"",y,"-"\n')
    df = sd.frame.read_csv(p, na_values=["NA", "-"])
    assert df["a"].tolist() == [NA, NA, ""] and df["b"].tolist() == ["x", NA, "y"] and df["c"].tolist() == [1, 2, NA]
    df = sd.frame.read_csv(p, na_values=[""])
    assert df["a"].tolist() == ["NA", "NA", NA]  # the quoted empty field too
