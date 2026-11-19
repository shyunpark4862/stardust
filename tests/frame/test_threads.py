"""Results do not depend on the thread count (docs/api/frame.md §12, docs/design/frame.md §11.5).

Inputs are big enough to cross the parallel threshold and many chunk
boundaries. Every operation runs with 1, 3 and the default thread count;
the results must be identical (bit for bit for floats). Separate checks
against numpy/pandas make sure the chunked code is also right, since a
1-thread run goes through the same chunks.
"""

import hashlib
import os
import struct
import threading

import numpy as np
import pytest

import stardust as sd

N = 200_003  # not a multiple of the chunk size
DEFAULT = sd.get_num_threads()
THREADS = sorted({1, 3, DEFAULT})


def bits(v):
    if isinstance(v, float):
        return ("f", struct.pack("<d", v))
    return v


def canon(x):
    """Something == can compare bit for bit."""
    if isinstance(x, (sd.frame.Series,)):
        return (x.dtype, [bits(v) for v in x.tolist()], canon(x.index))
    if isinstance(x, sd.frame.DataFrame):
        return (list(x.columns), [canon(x[c]) for c in x.columns], canon(x.index))
    if isinstance(x, (sd.frame.Index, sd.frame.RangeIndex)):
        return [bits(v) for v in x.tolist()]
    if isinstance(x, float):
        return bits(x)
    return x


def digest(c):
    """A digest of a canon() value (its repr: bytes, numbers, strings, NA and dtypes)."""
    return hashlib.sha256(repr(c).encode()).digest()


def same_for_all_thread_counts(fn):
    # the first thread count's canon() (returned) and its digest; later results are compared
    # by digest and dropped
    first = d0 = None
    try:
        for t in THREADS:
            sd.set_num_threads(t)
            c = canon(fn())
            if first is None:
                first, d0 = c, digest(c)
            else:
                assert digest(c) == d0
            del c
    finally:
        sd.set_num_threads(DEFAULT)
    return first


rng = np.random.default_rng(11)
na = rng.random(N) < 0.1
ints = rng.integers(-1000, 1000, N)
floats = rng.standard_normal(N)
floats[rng.random(N) < 0.01] = np.nan
floats[rng.random(N) < 0.01] = -0.0
words = [f"w{v}" for v in rng.integers(0, 5000, N).tolist()]
PERM = rng.permutation(N)[: N // 2].tolist()


class D:
    """The inputs, built by a module fixture and dropped after the module:
    Columns left in module globals would outlive nanobind's leak check at
    interpreter exit."""


@pytest.fixture(scope="module", autouse=True)
def data():
    D.I = sd.frame.Series([None if m else int(v) for m, v in zip(na.tolist(), ints.tolist())])
    D.F = sd.frame.Series([None if m else float(v) for m, v in zip(na.tolist(), floats.tolist())])
    D.S = sd.frame.Series([None if m else v for m, v in zip(na.tolist(), words)])
    D.B = D.F > 0
    D.MASK = sd.frame.Series(rng.random(N) < 0.3)
    D.DF = sd.frame.DataFrame({"i": D.I, "f": D.F, "s": D.S, "b": D.B})
    yield
    for k in ["I", "F", "S", "B", "MASK", "DF"]:
        delattr(D, k)


ELEMENTWISE = {
    "int add": lambda: D.I + D.I,
    "int floordiv": lambda: D.I // (D.I % 7),
    "float mul": lambda: D.F * D.F,
    "mixed compare": lambda: D.I < D.F,
    "string compare": lambda: D.S < "w2500",
    "string concat": lambda: D.S + D.S,
    "kleene and": lambda: D.B & (D.I > 0),
    "kleene or": lambda: D.B | (D.I > 0),
    "neg": lambda: -D.F,
    "invert": lambda: ~D.B,
    "isna": lambda: D.F.isna(),
    "notna": lambda: D.S.notna(),
    "fillna int": lambda: D.I.fillna(0),
    "fillna string": lambda: D.S.fillna("x"),
    "dropna": lambda: D.F.dropna(),
    "mask filter": lambda: D.DF[D.MASK],
    "iloc take": lambda: D.DF.iloc[PERM],
    "index equals": lambda: sd.frame.Index(words).equals(sd.frame.Index(list(words))),
    "astype": lambda: D.F.astype("string"),
}


@pytest.mark.parametrize("name", ELEMENTWISE)
def test_elementwise_and_selection(name):
    same_for_all_thread_counts(ELEMENTWISE[name])


def test_chunked_kernels_match_numpy():
    # The chunk boundaries must not lose or repeat rows.
    sd.set_num_threads(DEFAULT)
    m = D.MASK.tolist()
    assert sd.frame.Series(ints)[D.MASK].tolist() == ints[np.array(m)].tolist()
    assert (sd.frame.Series(ints) + 1).tolist() == (ints + 1).tolist()
    expect_na = [v is sd.frame.NA for v in D.F.tolist()]
    assert D.F.isna().tolist() == expect_na
    kept = [v for v in D.S.tolist() if v is not sd.frame.NA]
    assert D.S.dropna().tolist() == kept


def test_snapshot_under_concurrent_writes():
    # A writer thread changes rows one by one while the main thread runs
    # kernels on the same Series. Each result must be the Series as it was
    # at one moment: rows [0, k) written, the rest untouched (docs/api/frame.md §12).
    n, writes = 400_000, 300
    s = sd.frame.Series(np.zeros(n, dtype=np.int64))
    done = threading.Event()

    def writer():
        for k in range(writes):
            s.iloc[k] = 1
        done.set()

    t = threading.Thread(target=writer)
    t.start()
    seen = 0
    while not done.is_set() or seen == 0:
        r = np.asarray((s + 0).tolist())
        k = int(r[:writes].sum())
        assert (r[:k] == 1).all() and (r[k:] == 0).all()
        seen += 1
    t.join()


def error_message(fn):
    with pytest.raises(ValueError) as e:
        fn()
    return str(e.value)


def test_errors_name_the_first_bad_row():
    # Several bad rows in different chunks: the message names the first one
    # for every thread count.
    vals = np.arange(N, dtype=np.float64)
    vals[[150_001, 70_000, 199_999]] = [np.inf, 0.5e300, np.nan]
    f = sd.frame.Series(vals)  # float -> int64 saturates now, with no error (docs/api/frame.md §3.5)
    got = same_for_all_thread_counts(lambda: f.astype("int64").iloc[[70_000, 150_001, 199_999]].tolist())
    assert got == [2**63 - 1, 2**63 - 1, 0]
    words2 = [str(i) for i in range(N)]
    words2[180_000], words2[90_000] = "x1", "x2"
    s = sd.frame.Series(words2)
    msg = same_for_all_thread_counts(lambda: error_message(lambda: s.astype("int64")))
    assert "'x2'" in msg


REDUCTIONS = {
    f"{op} {name}": (lambda op=op, key=key: getattr(getattr(D, key), op)())
    for op in ["sum", "mean", "min", "max", "std", "var", "count"]
    for name, key in [("int", "I"), ("float", "F"), ("bool", "B"), ("string", "S")]
    if not (name == "string" and op not in ("min", "max", "count")) and not (name == "bool" and op in ("std", "var"))
}


@pytest.mark.parametrize("name", REDUCTIONS)
def test_reductions(name):
    same_for_all_thread_counts(REDUCTIONS[name])


@pytest.mark.parametrize("n", [16_384, 16_385, 200_003, 1_048_583])
def test_float_sum_is_numpy_pairwise(n):
    # The parallel tree adds the same nodes as numpy's serial recursion.
    v = np.random.default_rng(n).standard_normal(n) * 1e6
    s = sd.frame.Series(v)
    got = same_for_all_thread_counts(lambda: s.sum())
    assert got == bits(float(np.add.reduce(v)))
    assert same_for_all_thread_counts(lambda: s.mean()) == bits(float(np.add.reduce(v)) / n)


HASHING = {
    "unique int": lambda: D.I.unique_values(),
    "unique float": lambda: D.F.unique_values(),
    "unique string": lambda: D.S.unique_values(),
    "nunique high": lambda: sd.frame.Series(rng_hi).nunique(),
    "value_counts": lambda: D.S.value_counts(),
    "duplicated": lambda: D.F.duplicated(),
    "drop_duplicates frame": lambda: D.DF[["i", "s"]].drop_duplicates(),
    "groupby int": lambda: D.DF.groupby("i").agg({"f": "sum", "s": "max"}),
    "groupby two keys": lambda: D.DF.groupby(["b", "s"]).agg({"f": "mean", "i": "count"}),
    "groupby string std": lambda: D.DF.groupby("s", sort=False)["f"].std(),
    # one right row per key, so the result has N rows (on "s" alone the 198,140 (s, i) pairs gave
    # 8.1 million); both sides stay above the parallel threshold
    "merge inner": lambda: sd.frame.merge(D.DF, D.DF[["s", "i"]].drop_duplicates(), on=["s", "i"], how="inner"),
    "merge outer": lambda: sd.frame.merge(D.DF[["i", "f"]], D.DF[["i", "b"]].head(5000), on="i", how="outer"),
    "merge left two keys": lambda: sd.frame.merge(
        D.DF[["i", "b", "f"]], D.DF[["i", "b", "s"]].head(30_000), on=["i", "b"], how="left"
    ),
    "merge right": lambda: sd.frame.merge(
        D.DF[["s", "f"]].head(40_000), D.DF[["s", "i"]].head(60_000), on="s", how="right"
    ),
    "loc labels": lambda: sd.frame.Series(range(N), index=[f"r{k}" for k in range(N)]).loc[["r5", "r199999", "r77"]],
}
rng_hi = np.random.default_rng(5).integers(0, N, N)


@pytest.mark.parametrize("name", HASHING)
def test_hashing(name):
    same_for_all_thread_counts(HASHING[name])


def test_factorize_order_is_first_appearance():
    sd.set_num_threads(DEFAULT)
    for s in (D.I, D.S, sd.frame.Series(rng_hi)):
        expect = list(dict.fromkeys(s.tolist()))
        assert s.drop_duplicates().tolist() == expect


SORTS = {
    "int": lambda: D.I.sort_values(),
    "float desc na first": lambda: D.F.sort_values(descending=True, na_position="first"),
    "string": lambda: D.S.sort_values(),
    "bool": lambda: D.B.sort_values(),
    "two keys": lambda: D.DF.sort_values(["s", "f"], descending=[True, False]),
    "sort_index": lambda: D.F.sort_values().sort_index(),
}


@pytest.mark.parametrize("name", SORTS)
def test_sorts(name):
    same_for_all_thread_counts(SORTS[name])


def test_sort_matches_python_sorted():
    sd.set_num_threads(DEFAULT)
    vals = D.S.tolist()
    order = sorted((i for i, v in enumerate(vals) if v is not sd.frame.NA), key=lambda i: vals[i])
    order += [i for i, v in enumerate(vals) if v is sd.frame.NA]
    assert D.S.sort_values().index.tolist() == order


# --- read_csv in chunks (docs/design/frame.md §10.6): files of several 512 KB chunks ---

from .test_csv import oracle_column, same, write  # test_csv's serial oracle (docs/design/frame.md §13.1)  # noqa: E402


def big_rows(seed, nrows, kinds):
    r = np.random.default_rng(seed)
    cols = []
    for k in kinds:
        if k == "int":
            vals = [
                (str(int(v)), bool(q)) for v, q in zip(r.integers(-(10**12), 10**12, nrows), r.random(nrows) < 0.05)
            ]
        elif k == "float":
            vals = [(repr(float(v)), False) for v in r.standard_normal(nrows)]
        elif k == "bool":
            vals = [(("True", "False")[int(v)], False) for v in r.integers(0, 2, nrows)]
        else:  # text with separators, quotes, line breaks, CRLF, unicode
            pool = ["plain", "a,b", 'say "hi"', "two\nlines", "cr\r\nlf", "유니코드", "", "x" * 40]
            vals = [(pool[int(v)], bool(q)) for v, q in zip(r.integers(0, len(pool), nrows), r.random(nrows) < 0.3)]
        na = r.random(nrows) < 0.05
        cols.append([(None, False) if m else v for m, v in zip(na.tolist(), vals)])
    return cols


@pytest.mark.parametrize("kinds", [["int", "float"], ["text", "int", "bool"], ["float", "text"]])
def test_csv_chunks_match_the_oracle(tmp_path, kinds):
    nrows = 120_000  # a few MB: many chunks, with records and quoted line breaks across the cuts
    cols = big_rows(len(kinds), nrows, kinds)
    names = [f"c{j}" for j in range(len(kinds))]
    path = tmp_path / "big.csv"
    write(path, [[(n, False) for n in names]] + [[c[i] for c in cols] for i in range(nrows)])
    assert os.path.getsize(path) > 4 * (1 << 19)
    df = same_for_all_thread_counts(lambda: sd.frame.read_csv(path))
    sd.set_num_threads(DEFAULT)
    df = sd.frame.read_csv(path)
    for j, k in enumerate(names):
        dtype, values = oracle_column(cols[j])
        assert str(df[k].dtype) == dtype
        assert all(same(a, b) for a, b in zip(df[k].tolist(), values))


def test_csv_long_quoted_field_spans_chunks(tmp_path):
    long = "q" * 1_500_000 + "\n" + '"' * 2 + "end"  # three chunks inside one quoted field
    path = tmp_path / "long.csv"
    write(path, [[("a", False), ("b", False)], [("1", False), (long, True)], [("2", False), ("x", False)]])
    df = same_for_all_thread_counts(lambda: sd.frame.read_csv(path))
    sd.set_num_threads(DEFAULT)
    df = sd.frame.read_csv(path)
    assert df["a"].tolist() == [1, 2] and df["b"].tolist() == [long, "x"]


def test_csv_errors_are_the_first_in_file_order(tmp_path):
    rows = [["k,v"]] + [[f"{i},{i}"] for i in range(200_000)]
    rows[150_000] = ["oops"]  # a short record late in the file
    rows[90_000] = ["1,2,3"]  # an earlier long one: this is the error
    path = tmp_path / "bad.csv"
    path.write_text("\n".join(r[0] for r in rows) + "\n")
    msg = same_for_all_thread_counts(lambda: error_message(lambda: sd.frame.read_csv(path)))
    assert "line 90001" in msg and "got 3" in msg
    rows[90_000] = ["1,x"]
    rows[150_000] = ["2,y"]
    path.write_text("\n".join(r[0] for r in rows) + "\n")
    msg = same_for_all_thread_counts(lambda: error_message(lambda: sd.frame.read_csv(path, dtype={"v": "int64"})))
    assert "line 90001" in msg and "'x'" in msg


def test_gil_is_released_even_with_one_thread():
    # Another Python thread keeps counting while a long kernel runs with
    # set_num_threads(1) (docs/api/frame.md §12). Holding the GIL would freeze it.
    s = sd.frame.Series(np.random.default_rng(3).integers(0, 1 << 40, 3_000_000))
    count, stop = [0], threading.Event()

    def counter():
        while not stop.is_set():
            count[0] += 1

    t = threading.Thread(target=counter)
    try:
        sd.set_num_threads(1)
        t.start()
        while count[0] == 0:
            pass
        before = count[0]
        s.sort_values()  # tens of ms in C++
        during = count[0] - before
    finally:
        stop.set()
        t.join()
        sd.set_num_threads(DEFAULT)
    assert during > 1000


def test_gil_is_released_by_chunked_kernels_with_one_thread():
    # At or above the threshold the kernels that run in chunks -- element-wise
    # operations, reductions, filters, grouped reductions -- release the GIL
    # with set_num_threads(1) too (docs/api/frame.md §12): a counting thread
    # makes progress during each call.
    n = 16_000_000
    s = sd.frame.Series(np.random.default_rng(4).standard_normal(n))
    mask = s > 0
    k = sd.frame.Series(np.random.default_rng(5).integers(0, 1000, n))
    g = sd.frame.DataFrame({"k": k, "v": s}).groupby("k")
    g.size()  # the grouping, computed before the timed calls
    calls = {
        "s + s": lambda: s + s,
        "sum": lambda: s.sum(),
        "min": lambda: s.min(),
        "var": lambda: s.var(),
        "filter": lambda: s[mask],
        "groupby sum": lambda: g.sum(),
    }
    count, stop, during = [0], threading.Event(), {}

    def counter():
        while not stop.is_set():
            count[0] += 1

    t = threading.Thread(target=counter)
    try:
        sd.set_num_threads(1)
        t.start()
        while count[0] == 0:
            pass
        for name, call in calls.items():
            before = count[0]
            call()  # several ms in C++
            during[name] = count[0] - before
    finally:
        stop.set()
        t.join()
        sd.set_num_threads(DEFAULT)
    assert all(v > 1000 for v in during.values()), during
