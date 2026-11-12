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
    (lambda: sd.sum(sd.ones(3), axis=1.5), TypeError, "sum(): 'axis' must be an int or a tuple or list of ints, not float"),
    (lambda: sd.ones(3).sum(axis=1.5), TypeError, "sum(): 'axis' must be an int or a tuple or list of ints, not float"),
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
     "drop_duplicates(): 'keep' must be 'first', 'last' or 'none', not 'x'"),
    (lambda: DF({"a": [1]}).sort_values("zz"), KeyError, "sort_values(): no column 'zz'"),
    (lambda: DF({"a": [1]}).sort_values(1), TypeError, "sort_values(): 'by' must be a column name or a list of them, not int"),
    (lambda: F.merge(DF({"a": [1]}), DF({"a": [1.0]}), on="a"), TypeError,
     "merge(): join keys must have one dtype, got int64 and float64"),
    (lambda: F.read_csv("/nonexistent", header=False), ValueError, "read_csv(): 'names' is required when header=False"),
    # frame options: one check per kind, the array's wording (API.md §2.1, DESIGN.md §10.1 #4)
    (lambda: S([1]).head(True), TypeError, "head(): 'n' must be an int, not bool"),
    (lambda: S([1]).tail(2**64), OverflowError, "tail(): 'n' is out of range"),
    (lambda: S([1]).dropna(ignore_index=1), TypeError, "dropna(): 'ignore_index' must be True or False, not int"),
    (lambda: DF({"a": [1]}).groupby("a", sort=None), TypeError, "groupby(): 'sort' must be True or False, not NoneType"),
    (lambda: F.merge(DF({"a": [1]}), DF({"a": [1]}), on="a", how=None), TypeError,
     "merge(): 'how' must be 'inner', 'left', 'right' or 'outer', not NoneType"),
    (lambda: F.concat(x for x in [S([1])]), TypeError,
     "concat(): 'objs' must be a list or tuple of DataFrames or Series, not generator"),
    (lambda: F.read_csv("/nonexistent", nrows=True), TypeError, "read_csv(): 'nrows' must be an int, not bool"),
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
    (lambda: S([1, 2]) == [1, 2], TypeError, "'==' not supported between instances of 'Series' and 'list'"),
    (lambda: F.Index([1]) != F.Index([1]), TypeError, "'!=' not supported between instances of 'Index' and 'Index'"),
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
        lambda: F.concat([S([1]), S(["a"])]),
        lambda: F.Index([1.5]),
        lambda: F.read_csv("/nonexistent", header=False),
        lambda: np.asarray(S([1, None])),
    ]
    for call in calls:
        assert "§" not in message(call) and "API.md" not in message(call)


# --- the same mistake, the same message, in every area (DESIGN.md §10.3) ---------

X3 = sd.asarray([1.0, 2.0, 3.0])
M2 = sd.asarray([[4.0, 1.0], [1.0, 3.0]])

SAME_MISTAKE = [
    # a bool option given something else
    (lambda: sd.sort(X3, stable=1), "sort(): 'stable' must be True or False, not int"),
    (lambda: sd.argsort(X3, descending=None), "argsort(): 'descending' must be True or False, not NoneType"),
    (lambda: sd.sum(X3, keepdims=np.True_), "sum(): 'keepdims' must be True or False, not bool"),
    (lambda: X3.mean(keepdims="no"), "mean(): 'keepdims' must be True or False, not str"),
    (lambda: sd.cumulative_sum(X3, include_initial=1), "cumulative_sum(): 'include_initial' must be True or False, not int"),
    (lambda: sd.linspace(0, 1, 3, endpoint=0), "linspace(): 'endpoint' must be True or False, not int"),
    (lambda: sd.linalg.cholesky(M2, upper=1), "cholesky(): 'upper' must be True or False, not int"),
    (lambda: sd.linalg.matrix_norm(M2, keepdims=None), "matrix_norm(): 'keepdims' must be True or False, not NoneType"),
    (lambda: S([1]).sum(skipna=1), "sum(): 'skipna' must be True or False, not int"),
    (lambda: DF({"a": [1]}).groupby("a", dropna=None), "groupby(): 'dropna' must be True or False, not NoneType"),
    # None where None is not a value of the option
    (lambda: sd.linalg.svd(M2, mode=None), "svd(): 'mode' must be 'reduced' or 'complete', not NoneType"),
    (lambda: sd.linalg.matrix_norm(M2, ord=None), "matrix_norm(): 'ord' must be a str, an int or a float, not NoneType"),
    (lambda: sd.std(X3, correction=None), "std(): 'correction' must be an int or a float, not NoneType"),
    (lambda: S([1.0]).std(correction=None), "std(): 'correction' must be an int or a float, not NoneType"),
    (lambda: sd.linalg.vecdot(X3, X3, axis=None), "vecdot(): 'axis' must be an int, not NoneType"),
    (lambda: sd.sort(X3, axis=None), "sort(): 'axis' must be an int, not NoneType"),
    (lambda: sd.astype(X3, None),
     "astype(): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128 (or their names), not NoneType"),
    (lambda: sd.zeros(None), "zeros(): 'shape' must be an int or a tuple of ints, not NoneType"),
    (lambda: sd.sin(None), "sin(): 'x' must be an sd.ndarray or a bool, int, float or complex, not NoneType"),
    (lambda: sd.matmul(None, X3), "matmul(): "),
    (lambda: sd.linalg.inv(None), "inv(): 'x' must be an sd.ndarray, not NoneType"),
    (lambda: S([1]).head(None), "head(): 'n' must be an int, not NoneType"),
    # short type names: no module paths
    (lambda: sd.zeros(sd.ones(1)), "zeros(): 'shape' must be an int or a tuple of ints, not ndarray"),
    (lambda: sd.zeros(np.float64(2)), "zeros(): 'shape' must be an int or a tuple of ints, not float64"),
    (lambda: sd.zeros(2, dtype=np.dtype("int64")),
     "zeros(): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128 (or their names), not Int64DType"),
    (lambda: sd.zeros(2, dtype=np.int64),
     "zeros(): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.complex128 (or their names), not <class 'int64'>"),
    (lambda: S([1]).astype(np.int64),
     "astype(): 'dtype' must be sd.bool, sd.int64, sd.float64 or sd.string (or their names), not <class 'int64'>"),
    (lambda: sd.linalg.det(np.eye(2)), "det(): 'x' must be an sd.ndarray, not ndarray"),
    (lambda: sd.concat(X3), "concat(): 'arrays' must be a list or tuple of sd.ndarray, not ndarray"),
    (lambda: sd.arange(X3), "arange(): 'start' must be a bool, int or float, not ndarray"),
    # wrong messages fixed (DESIGN.md §10.3)
    (lambda: sd.arange("a"), "arange(): 'start' must be a bool, int or float, not str"),
    (lambda: sd.linalg.vector_norm(X3, axis=1.0),
     "vector_norm(): 'axis' must be an int or a tuple or list of ints, not float"),
    (lambda: S([1.0, None]).fillna(1j), "fillna(): 'value' must be None, a bool, int, float or str, not complex"),
    (lambda: S([1]).to_array(dtype=sd.string), "to_array(): 'dtype' must be sd.bool, sd.int64 or sd.float64"),
    (lambda: F.Index([None]), "Index(): an index cannot hold NA"),
    (lambda: DF({"a": [1]}).astype({"zz": "float64"}), "astype(): no column 'zz'"),
    (lambda: S([1, None]).to_numpy(), "to_numpy(): cannot convert an int64 column with NA to numpy"),
    (lambda: S(["a"]).mean(), "mean(): cannot compute mean of a string column"),
    (lambda: F.read_csv(5), "read_csv(): 'path' must be a str or a path-like object, not int"),
]


@pytest.mark.parametrize(("call", "text"), SAME_MISTAKE)
def test_the_same_mistake_has_the_same_message(call, text):
    with pytest.raises(TypeError if "cannot hold NA" not in text and "with NA" not in text and "no column" not in text
                       else (ValueError, KeyError)):
        call()
    assert message(call).startswith(text)


DUMP = re.compile(r"incompatible function arguments|stardust\._core|numpy\.|nanobind")


def _calls(module, names):
    """For every public function of `module` bound in C++: each argument and each
    keyword option given a few wrong values, the other arguments valid."""
    bad_values = [None, 1, True, "a", 1.5, [1], np.int64(1), np.True_, object(), 2**70]
    for name in names:
        f = getattr(module, name)
        sigs = getattr(f, "__nb_signature__", None)
        if not sigs:
            continue
        head = sigs[0][0].split("(", 1)[1].rsplit(")", 1)[0]
        params = [p.strip() for p in head.split(",") if p.strip() not in ("", "/")]
        kw_only = params[params.index("*") + 1:] if "*" in params else []
        npos = len(params) - len(kw_only) - ("*" in params)
        base_kw = {"solve_triangular": {"upper": True}, "squeeze": {"axis": 0}, "expand_dims": {"axis": 0},
                   "solve_banded": {"lower_bandwidth": 0, "upper_bandwidth": 0}}.get(name, {})
        good = [M2] * npos
        for i in range(npos):
            for b in bad_values:
                yield name, good[:i] + [b] + good[i + 1:], dict(base_kw)
        for p in kw_only:
            for b in bad_values:
                yield name, good, {**base_kw, p.split(":")[0].strip(): b}


def test_no_argument_value_shows_nanobinds_dump_or_a_module_path():
    # DESIGN.md §10.3, API.md §2.1: every wrong value reaches a check that names the argument
    found = []
    for module, names in ((sd, sd.__all__), (sd.linalg, sd.linalg.__all__)):
        for name, args, kwargs in _calls(module, names):
            try:
                getattr(module, name)(*args, **kwargs)
            except Exception as e:
                text = str(e)
                if DUMP.search(text) or "\n" in text:
                    found.append(f"{name}{tuple(type(a).__name__ for a in args)} {kwargs}: {text.splitlines()[0]}")
    assert not found, "\n".join(found[:20])
