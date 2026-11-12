"""One error-message format for the whole library (API.md §2.1, DESIGN.md §9.7):
a public function or method names itself, `func(): `; operators and indexing
keep Python's form; no message cites a document section."""

import re

import numpy as np
import pytest

import stardust as sd

F = sd.frame
S, DF = F.Series, F.DataFrame


def message(call):
    with pytest.raises(Exception) as info:
        call()
    e = info.value
    return e.args[0] if isinstance(e, KeyError) else str(e)  # str(KeyError) is a repr


NAMED = [
    # array: functions, the same function as a method, a property
    (lambda: sd.zeros(-1), ValueError, "zeros(): negative dimensions are not allowed"),
    (lambda: sd.sum(sd.ones(3), axis=[0]), TypeError, "sum(): 'axis' must be an int or a tuple of ints, not list"),
    (lambda: sd.ones(3).sum(axis=[0]), TypeError, "sum(): 'axis' must be an int or a tuple of ints, not list"),
    (lambda: sd.ones(3).astype("float32"), TypeError,
     "astype(): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128 (or their names), not 'float32'"),
    (lambda: sd.add(sd.ones(2), sd.ones(3)), ValueError, "add(): shapes could not be broadcast together: (2,) (3,)"),
    (lambda: sd.less(sd.ones(1, dtype=sd.complex128), 1), TypeError, "less(): not supported for complex128 and int64"),
    (lambda: sd.ones((2, 2, 2)).T, ValueError, r".T: the array must have 2 dimensions, got 3"),
    (lambda: sd.ones(3).mT, ValueError, ".mT: the array must have at least 2 dimensions, got 1"),
    (lambda: sd.set_num_threads(0), ValueError, "set_num_threads(): 'n' must be between 1 and 1024"),
    # linalg: LinAlgError too
    (lambda: sd.linalg.inv(sd.zeros((2, 2))), sd.linalg.LinAlgError, "inv(): singular matrix"),
    (lambda: sd.linalg.trace(sd.eye(2), dtype=sd.bool), TypeError, "trace(): cannot accumulate in bool; use any() or all()"),
    # frame: constructors, methods shared by Series/DataFrame/GroupBy, kernels named by the caller
    (lambda: S([1], dtype=float), TypeError,
     "Series(): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.string (or their names), not <class 'float'>"),
    (lambda: S([{}]), TypeError, "Series(): cannot store an object of type 'dict' in a column"),
    (lambda: F.RangeIndex(0, 3, 0), ValueError, "RangeIndex(): 'step' must not be zero"),
    (lambda: S(["a"]).astype("int64"), ValueError, "astype(): cannot convert string 'a' to int64"),
    (lambda: S(["a"]).sum(), TypeError, "sum(): cannot compute sum of a string column"),
    (lambda: DF({"k": [1], "v": ["a"]}).groupby("k").agg("sum"), TypeError, "agg(): cannot compute sum of a string column"),
    (lambda: DF({"a": [1]}).drop_duplicates(keep="x"), ValueError,
     "drop_duplicates(): 'keep' must be 'first', 'last' or 'none', got 'x'"),
    (lambda: DF({"a": [1]}).sort_values("zz"), KeyError, "sort_values(): no column 'zz'"),
    (lambda: DF({"a": [1]}).sort_values(1), TypeError, "sort_values(): 'by' must be a column name or a list of them, not int"),
    (lambda: F.merge(DF({"a": [1]}), DF({"a": [1.0]}), on="a"), TypeError,
     "merge(): join keys must have one dtype, got int64 and float64"),
    (lambda: F.read_csv("/nonexistent", header=False), ValueError, "read_csv(): 'names' is required when header=False"),
]


@pytest.mark.parametrize(("call", "kind", "text"), NAMED)
def test_public_functions_name_themselves(call, kind, text):
    with pytest.raises(kind):
        call()
    assert message(call).startswith(text)


PYTHON_FORM = [
    (lambda: sd.ones(1, dtype=sd.complex128) < 1, TypeError, "unsupported operand dtypes for <: complex128 and int64"),
    (lambda: sd.ones(1, dtype=sd.complex128) // 1, TypeError, "unsupported operand dtypes for //: complex128 and int64"),
    (lambda: ~sd.ones(1), TypeError, "unsupported operand dtype for unary ~: float64"),
    (lambda: sd.ones(2) + sd.ones(3), ValueError, "shapes could not be broadcast together: (2,) (3,)"),
    (lambda: sd.ones(2) == [1, 2], TypeError, "'==' not supported between instances of"),
    (lambda: -S(["a"]), TypeError, "unsupported operand dtype for unary -: string"),
    (lambda: abs(S(["a"])), TypeError, "unsupported operand dtype for abs(): string"),
    (lambda: DF({"a": [1]})["zz"], KeyError, "zz"),
    (lambda: S([1]).iloc[5], IndexError, "position 5 is out of range for 1 rows"),
    (lambda: setattr(sd.ones(1), "shape", (1,)), AttributeError, "property 'shape' of 'ndarray' object has no setter"),
]


@pytest.mark.parametrize(("call", "kind", "text"), PYTHON_FORM)
def test_operators_and_indexing_keep_pythons_form(call, kind, text):
    with pytest.raises(kind):
        call()
    msg = message(call)
    assert msg.startswith(text)
    assert not re.match(r"^\w+\(\): ", msg)


def test_no_message_cites_a_document():
    calls = [c for c, _, _ in NAMED + PYTHON_FORM] + [
        lambda: DF({"a": [1]}) + S([1]),
        lambda: S([1, 2], index=[0, 0]).reindex([0]),
        lambda: F.concat([S([1]), S([1.0])]),
        lambda: F.Index([1.5]),
        lambda: F.read_csv("/nonexistent", header=False),
        lambda: np.asarray(S([1, None])),
    ]
    for call in calls:
        assert "§" not in message(call) and "API.md" not in message(call)
