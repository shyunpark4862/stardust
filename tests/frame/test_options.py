"""Options take what the array area's options take (DESIGN.md §10.1 #4, #4b,
#4c; API.md §2, §2.1): a bool option True or False, an int argument a Python
int in int64, a string option a str. Every public signature is searched for
bool defaults, so a new bool option is checked without being listed here."""

import inspect

import numpy as np
import pytest

import stardust as sd
from stardust.frame._groupby import GroupBy

S = sd.frame.Series([1, 2, None])
DF = sd.frame.DataFrame({"a": [1, 2, 2], "b": [1.0, None, 3.0]})

# the main arguments a method needs before its options
NEEDS = {
    ("DataFrame", "sort_values"): (["a"], {}),
    ("DataFrame", "merge"): ([DF], {"on": "a"}),
    ("DataFrame", "groupby"): (["a"], {}),
    ("frame", "concat"): ([[S, S]], {}),
    ("frame", "merge"): ([DF, DF], {"on": "a"}),
}
OBJECTS = {"Series": S, "DataFrame": DF, "GroupBy": DF.groupby("a"), "frame": sd.frame}
CLASSES = {"Series": sd.frame.Series, "DataFrame": sd.frame.DataFrame, "GroupBy": GroupBy}


def bool_options():
    out = []
    for owner, obj in OBJECTS.items():
        names = sd.frame.__all__ if owner == "frame" else [n for n in dir(CLASSES[owner]) if not n.startswith("_")]
        for name in names:
            f = getattr(obj, name)
            if not callable(f) or isinstance(f, type):
                continue
            for p in inspect.signature(f).parameters.values():
                if isinstance(p.default, bool):
                    out.append((owner, name, p.name))
    return out


OPTIONS = bool_options()


def test_the_search_finds_the_options():
    found = {(o, n, p) for o, n, p in OPTIONS}
    assert ("Series", "sum", "skipna") in found and ("frame", "concat", "ignore_index") in found
    assert ("GroupBy", "first", "skipna") in found and ("DataFrame", "value_counts", "normalize") in found
    assert len(OPTIONS) >= 40


@pytest.mark.parametrize(("owner", "name", "option"), OPTIONS, ids=lambda x: str(x))
def test_bool_options_take_true_or_false(owner, name, option, tmp_path):
    args, kwargs = NEEDS.get((owner, name), ([], {}))
    if (owner, name) == ("frame", "read_csv"):
        (tmp_path / "t.csv").write_text("a\n1\n")
        args, kwargs = [tmp_path / "t.csv"], {"names": ["a"]}
    f = getattr(OBJECTS[owner], name)
    for bad, shown in ((1, "int"), (0, "int"), (None, "NoneType"), (np.True_, "bool"), ("yes", "str")):
        with pytest.raises(TypeError, match=rf"^{name}\(\): '{option}' must be True or False, not {shown}$"):
            f(*args, **kwargs, **{option: bad})
    f(*args, **kwargs, **{option: True})
    f(*args, **kwargs, **{option: False})


@pytest.mark.parametrize("obj", [S, DF])
@pytest.mark.parametrize("name", ["head", "tail"])
def test_int_arguments(obj, name):
    f = getattr(obj, name)
    assert len(f(2)) == 2 and len(f(-1)) == 2 and len(f(2**63 - 1)) == 3 and len(f(-(2**63))) == 0
    for bad, shown in ((True, "bool"), (2.0, "float"), ("2", "str"), (None, "NoneType"), (np.int64(2), "int64")):
        with pytest.raises(TypeError, match=rf"^{name}\(\): 'n' must be an int, not {shown}$"):
            f(bad)
    for big in (2**63, -(2**63) - 1, 2**100):
        with pytest.raises(OverflowError, match=rf"^{name}\(\): 'n' is out of range$"):
            f(big)


@pytest.mark.parametrize(("call", "option", "choices"), [
    (lambda v: S.duplicated(keep=v), "keep", "'first', 'last' or 'none'"),
    (lambda v: DF.drop_duplicates(keep=v), "keep", "'first', 'last' or 'none'"),
    (lambda v: DF.dropna(how=v), "how", "'any' or 'all'"),
    (lambda v: sd.frame.merge(DF, DF, on="a", how=v), "how", "'inner', 'left', 'right' or 'outer'"),
    (lambda v: DF.merge(DF, on="a", how=v), "how", "'inner', 'left', 'right' or 'outer'"),
    (lambda v: sd.frame.concat([S], join=v), "join", "'outer' or 'inner'"),
    (lambda v: S.sort_values(na_position=v), "na_position", "'first' or 'last'"),
    (lambda v: DF.sort_values(["a", "b"], na_position=["first", v]), "na_position", "'first' or 'last'"),
])
def test_string_options(call, option, choices):
    # not a str: TypeError; a str that is not a choice: ValueError, as in linalg (API.md §2)
    for bad, shown in ((0, "int"), (None, "NoneType"), (False, "bool"), (b"any", "bytes")):
        with pytest.raises(TypeError, match=rf"'{option}' must be {choices}, not {shown}$"):
            call(bad)
    with pytest.raises(ValueError, match=rf"'{option}' must be {choices}, not 'x'$"):
        call("x")


@pytest.mark.parametrize("call", [
    lambda g: DF.sort_values(g),
    lambda g: DF.groupby(g),
    lambda g: DF.duplicated(g),
    lambda g: DF.drop_duplicates(g),
    lambda g: DF.value_counts(g),
    lambda g: DF.dropna(subset=g),
    lambda g: sd.frame.merge(DF, DF, on=g),
    lambda g: sd.frame.merge(DF, DF, left_on=g, right_on=["a"]),
    lambda g: sd.frame.DataFrame({"a": [1]}, columns=g),
    lambda g: sd.frame.concat(g),
    lambda g: sd.frame.Index(g),
    lambda g: S.reindex(g),
])
def test_list_arguments_take_no_generators(call):
    # a list or a tuple, as sd.concat and the array area's list arguments (DESIGN.md §10.1 #12)
    with pytest.raises(TypeError, match="not generator"):
        call(x for x in ["a"])
    assert sd.frame.concat((S,)).tolist() == S.tolist() and DF.sort_values(("a", "b")).shape == DF.shape
