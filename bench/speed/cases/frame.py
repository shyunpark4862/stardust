"""sd.frame against pandas. The base set is bench/frame/final_compare.py (at commit 4151d2a)'s (15 calls at
1,000, 100,000 and 1,000,000 rows, four per-call-overhead rows at 10 rows) with its
inputs (data(n, seed)); the post-merge measurement of those rows is in bench/merge/m8/final/.
On top of it: NA columns, groupby aggregations beyond sum, left / outer / string-key /
two-key / many-to-many merges, pre-sorted inputs, duplicates and reductions, read_csv
options, 262,144 rows (above the 131,072-row split threshold, frame.md §11.2), a wide
frame, few groups and long common-prefix string keys.

Branchy calls (hash, sort, groupby, merge, parsing) take a different input of the same
shape each call, cycling through harness.fresh_count(n) inputs (4 at 1,000,000 rows,
64 at 1,000): on one input called over and over the branch predictor and the caches
learn that input and the loops look faster than they are (docs/design/frame.md §12.1).
Elementwise calls, reductions and mask filters are branch-free (§7.3) and take one
input. Both libraries get the same inputs in the same order.

pandas runs on one thread and does sd's work: the same options (stable sorts, ddof=0,
sort=True where pandas' outer merge sorts), NA as NaN in float columns and as the
nullable Int64 in int columns. The "pandas" column is pandas as installed: with pyarrow
present its str columns are pyarrow-backed and read_csv uses its default (C) engine.
String cases also have a "pandas python str" column, pandas' str with python storage
(what pandas uses without pyarrow, and what the old baseline measured), and read_csv a
"pandas pyarrow engine" column (engine="pyarrow"). Polars, a column of the old script, is
not a comparator here."""

from __future__ import annotations

import dataclasses
import functools
import operator

from ..harness import fresh_count, rotate
from ..registry import Case, Ref

SIZES = [1_000, 100_000, 262_144, 1_000_000]
GROUP = {1_000: "rows-1e3", 100_000: "rows-1e5", 262_144: "rows-262144", 1_000_000: "rows-1e6"}
BASE = (1_000, 100_000, 1_000_000)  # final_compare.py's sizes
MID = (100_000, 1_000_000)
SPLIT = (262_144,)
BIG = (1_000_000,)
SMALL = 10
SMALL_GROUP = "rows-10"
WIDE_ROWS, WIDE_COLS = 100_000, 100
WIDE_GROUP = "wide-1e5x100"
NA_RATE = 0.1


def _data_dir():
    from ..results import RUN_DIR

    d = RUN_DIR / "frame-data"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _Raw:
    """final_compare.py's data(n, seed), then the columns added since; the strings made
    only when asked for."""

    def __init__(self, n, seed):
        import numpy as np

        rng = np.random.default_rng(seed)
        self.n = n
        self.low, self.high = rng.integers(0, 1000, n), rng.integers(0, n, n)
        self._codes = rng.integers(0, max(n // 10, 1), n)
        self.f, self.g = rng.standard_normal(n), rng.standard_normal(n)
        m = max(n // 10, 1)
        self.m = m
        self.lk, self.rk, self.rb = rng.integers(0, m + m // 9, n), rng.permutation(m), rng.standard_normal(m)
        # added since final_compare.py
        self.na = rng.random(n) < NA_RATE
        self.i = rng.integers(-1000, 1000, n)
        self.k4 = rng.integers(0, 4, n)
        self.lk2 = rng.integers(0, 2, n)  # second merge key; the right side has both values of every key
        self.rk_dup = rng.integers(0, max(m // 2, 1), m)  # right keys about twice each: many-to-many

    @functools.cached_property
    def strs(self):
        return [f"key{v}" for v in self._codes.tolist()]

    @functools.cached_property
    def lstrs(self):
        return [f"key{v}" for v in self.lk.tolist()]

    @functools.cached_property
    def rstrs(self):
        return [f"key{v}" for v in self.rk.tolist()]

    @functools.cached_property
    def cust(self):
        return [f"customer_{v:06d}" for v in self._codes.tolist()]

    @functools.cached_property
    def digits(self):
        return [str(v) for v in self.high.tolist()]

    @functools.cached_property
    def low_order(self):
        import numpy as np

        return np.argsort(self.low, kind="stable")


def _py_str():
    import numpy as np
    import pandas as pd

    return pd.StringDtype("python", na_value=np.nan)


def _objects(lib, raw, keys):
    """The Series / DataFrames of `keys` in `lib`: "sd", "pandas" (pandas' default str,
    pyarrow-backed when pyarrow is installed) or "pystr" (pandas with python-storage str)."""
    import numpy as np

    if lib == "sd":
        import stardust as sd

        Series, DataFrame = sd.frame.Series, sd.frame.DataFrame

        def text(values):
            return Series(values)

        def with_na(values, mask):  # int columns stay int64 with NA
            s = Series(values)
            s[Series(mask)] = sd.frame.NA
            return s

    else:
        import pandas as pd

        Series, DataFrame = pd.Series, pd.DataFrame
        str_dtype = _py_str() if lib == "pystr" else None  # None: pandas' default str

        def text(values):
            return pd.Series(values, dtype=str_dtype)

        def with_na(values, mask):  # float: NaN; int: the nullable Int64
            if values.dtype.kind == "f":
                v = values.copy()
                v[mask] = np.nan
                return pd.Series(v)
            return pd.Series(pd.arrays.IntegerArray(values.astype(np.int64), mask.copy()))

    m = raw.m
    make = {
        "f": lambda: Series(raw.f),
        "g": lambda: Series(raw.g),
        "low": lambda: Series(raw.low),
        "high": lambda: Series(raw.high),
        "str": lambda: text(raw.strs),
        "df": lambda: DataFrame({"k": raw.low, "kh": raw.high, "ks": text(raw.strs), "f": raw.f}),
        "L": lambda: DataFrame({"k": raw.lk, "a": raw.f}),
        "R": lambda: DataFrame({"k": raw.rk, "b": raw.rb}),
        "f_na": lambda: with_na(raw.f, raw.na),
        "dfa": lambda: DataFrame({"k": raw.low, "k4": raw.k4, "f": raw.f, "i": raw.i, "f_na": with_na(raw.f, raw.na)}),
        "dfkf": lambda: DataFrame({"k": raw.low, "f": raw.f}),
        "high_sorted": lambda: Series(np.sort(raw.high)),
        "dfs": lambda: DataFrame({"k": raw.low[raw.low_order], "f": raw.f[raw.low_order]}),
        "Lna": lambda: DataFrame({"k": with_na(raw.lk, raw.na), "a": raw.f}),
        "Ls": lambda: DataFrame({"k": text(raw.lstrs), "a": raw.f}),
        "Rs": lambda: DataFrame({"k": text(raw.rstrs), "b": raw.rb}),
        "L2": lambda: DataFrame({"k": raw.lk, "k2": raw.lk2, "a": raw.f}),
        "R2": lambda: DataFrame({"k": np.repeat(raw.rk, 2), "k2": np.tile(np.arange(2), m), "b": np.repeat(raw.rb, 2)}),
        "Rm": lambda: DataFrame({"k": raw.rk_dup, "b": raw.rb}),
        "cust": lambda: text(raw.cust),
        "dfc": lambda: DataFrame({"kc": text(raw.cust), "f": raw.f}),
        "digits": lambda: text(raw.digits),
        "shuffled": lambda: (
            Series(raw.high).sort_values() if lib == "sd" else Series(raw.high).sort_values(kind="stable")
        ),
    }
    return {k: make[k]() for k in keys}


# ---------------------------------------------------------------- calls


@dataclasses.dataclass(frozen=True)
class Call:
    """One row per size in `sizes`. sd / pd: objects -> zero-argument callable (pd
    defaults to sd's); fresh: a branchy call (rotate fresh inputs); strings: string
    columns (adds the "pandas python str" column)."""

    name: str
    keys: str
    sd: object
    pd: object = None
    fresh: bool = True
    note: str = ""
    sizes: tuple = MID
    strings: bool = False


def _sd_merge(left, right, **kw):
    def make(o):
        import stardust as sd

        return functools.partial(sd.frame.merge, o[left], o[right], **kw)

    return make


def _pd_merge(left, right, **kw):
    return lambda o: functools.partial(o[left].merge, o[right], **kw)


def _merge(name, left, right, note="", sizes=MID, strings=False, sd_kw=None, **kw):
    return Call(
        name,
        f"{left} {right}",
        _sd_merge(left, right, **{**kw, **(sd_kw or {})}),
        _pd_merge(left, right, **kw),
        note=note,
        sizes=sizes,
        strings=strings,
    )


def _gb(frame, by, col, method, **kw):
    """o[frame].groupby(by)[col].method(**kw), or o[frame].groupby(by).method(**kw) for col None."""
    if col is None:
        return lambda o: lambda df=o[frame]: getattr(df.groupby(by), method)(**kw)
    return lambda o: lambda df=o[frame]: getattr(df.groupby(by)[col], method)(**kw)


def _method(key, method, **kw):
    return lambda o: functools.partial(getattr(o[key], method), **kw)


def _np_sort(a):
    import numpy as np

    return np.sort(a)


def _concat_sd(o):
    import stardust as sd

    return functools.partial(sd.frame.concat, [o["dfkf"], o["dfkf"]])


def _concat_pd(o):
    import pandas as pd

    return functools.partial(pd.concat, [o["dfkf"], o["dfkf"]])


STABLE = {"kind": "stable"}

CALLS = [
    # final_compare.py's calls
    Call("f + g", "f g", lambda o: functools.partial(operator.add, o["f"], o["g"]), fresh=False, sizes=BASE + SPLIT),
    Call("f.sum()", "f", lambda o: o["f"].sum, fresh=False, sizes=BASE + SPLIT),
    Call("f[f > 0]", "f", lambda o: lambda f=o["f"]: f[f > 0], fresh=False, sizes=BASE + SPLIT),
    Call("low.nunique()", "low", lambda o: o["low"].nunique, note="정수 키 1,000가지", sizes=BASE),
    Call("high.nunique()", "high", lambda o: o["high"].nunique, note="정수 키 약 n가지", sizes=BASE),
    Call("strs.nunique()", "str", lambda o: o["str"].nunique, note="문자열 키 n/10가지", sizes=BASE, strings=True),
    Call('df.groupby("k")["f"].sum()', "df", _gb("df", "k", "f", "sum"), note="정수 키 1,000가지", sizes=BASE + SPLIT),
    Call('df.groupby("kh")["f"].sum()', "df", _gb("df", "kh", "f", "sum"), note="정수 키 약 n가지", sizes=BASE),
    Call(
        'df.groupby("ks")["f"].sum()',
        "df",
        _gb("df", "ks", "f", "sum"),
        note="문자열 키 n/10가지",
        sizes=BASE,
        strings=True,
    ),
    Call(
        'df.groupby(["k", "ks"])["f"].sum()',
        "df",
        _gb("df", ["k", "ks"], "f", "sum"),
        note="정수 1,000가지 × 문자열",
        sizes=BASE,
        strings=True,
    ),
    _merge('merge(left, right, on="k")', "L", "R", note="오른쪽 n/10행, inner", sizes=BASE + SPLIT, on="k"),
    Call(
        "high.sort_values()",
        "high",
        lambda o: o["high"].sort_values,
        _method("high", "sort_values", **STABLE),
        note='pandas는 kind="stable"',
        sizes=BASE + SPLIT,
    ),
    Call(
        "strs.sort_values()",
        "str",
        lambda o: o["str"].sort_values,
        _method("str", "sort_values", **STABLE),
        note='pandas는 kind="stable"',
        sizes=BASE,
        strings=True,
    ),
    # NA: f_na is f with 10% NA (pandas: NaN), the left key of merge(left_na, ...) is lk with 10% NA (pandas: Int64)
    Call("f_na + g", "f_na g", lambda o: functools.partial(operator.add, o["f_na"], o["g"]), fresh=False),
    Call("f_na.sum()", "f_na", lambda o: o["f_na"].sum, fresh=False, note="NA 10%"),
    Call("f_na.isna()", "f_na", lambda o: o["f_na"].isna, fresh=False),
    Call("f_na.dropna()", "f_na", lambda o: o["f_na"].dropna, fresh=False),
    Call("f_na.fillna(0)", "f_na", lambda o: functools.partial(o["f_na"].fillna, 0), fresh=False),
    Call('df.groupby("k")["f_na"].mean()', "dfa", _gb("dfa", "k", "f_na", "mean"), note="NA 10%"),
    Call(
        "f_na.sort_values()",
        "f_na",
        lambda o: o["f_na"].sort_values,
        _method("f_na", "sort_values", **STABLE),
        note='NA는 뒤로, pandas는 kind="stable"',
    ),
    _merge('merge(left_na, right, on="k")', "Lna", "R", note="왼쪽 키의 10%가 NA", on="k"),
    # groupby aggregations beyond sum
    Call('df.groupby("k")["f"].mean()', "dfa", _gb("dfa", "k", "f", "mean")),
    Call('df.groupby("k")["f"].max()', "dfa", _gb("dfa", "k", "f", "max")),
    Call(
        'df.groupby("k")["f"].std()',
        "dfa",
        _gb("dfa", "k", "f", "std"),
        _gb("dfa", "k", "f", "std", ddof=0),
        note="pandas는 ddof=0",
    ),
    Call('df.groupby("k").size()', "dfa", _gb("dfa", "k", None, "size")),
    Call(
        'df.groupby("k").agg(t=("f", "sum"), m=("f", "mean"))',
        "dfa",
        _gb("dfa", "k", None, "agg", t=("f", "sum"), m=("f", "mean")),
    ),
    Call('df.groupby("k")["i"].sum()', "dfa", _gb("dfa", "k", "i", "sum"), note="int 열"),
    # merges
    _merge('merge(left, right, on="k", how="left")', "L", "R", on="k", how="left"),
    _merge(
        'merge(left, right, on="k", how="outer", sort=True)',
        "L",
        "R",
        note="pandas의 outer는 키로 정렬하므로 sd도 sort=True",
        on="k",
        how="outer",
        sd_kw={"sort": True},
    ),
    _merge('merge(left_s, right_s, on="k")', "Ls", "Rs", note="문자열 키", strings=True, on="k"),
    _merge('merge(left2, right2, on=["k", "k2"])', "L2", "R2", note="키 둘", on=["k", "k2"]),
    _merge('merge(left, right_dup, on="k")', "L", "Rm", note="오른쪽 키도 겹친다(다대다)", on="k"),
    # pre-sorted inputs, descending, two keys
    Call(
        "high_sorted.sort_values()",
        "high_sorted",
        lambda o: o["high_sorted"].sort_values,
        _method("high_sorted", "sort_values", **STABLE),
        note='이미 정렬된 입력, pandas는 kind="stable"',
    ),
    Call(
        "f.sort_values(descending=True)",
        "f",
        _method("f", "sort_values", descending=True),
        _method("f", "sort_values", ascending=False, **STABLE),
        note='pandas는 ascending=False, kind="stable"',
    ),
    Call('df.sort_values(["k", "f"])', "dfkf", lambda o: functools.partial(o["dfkf"].sort_values, ["k", "f"])),
    Call('df_sorted.groupby("k")["f"].sum()', "dfs", _gb("dfs", "k", "f", "sum"), note="키가 이미 정렬돼 있다"),
    # duplicates, unique values, reductions, a filter with a string column
    Call("low.value_counts()", "low", lambda o: o["low"].value_counts),
    Call("strs.value_counts()", "str", lambda o: o["str"].value_counts, strings=True),
    Call("high.duplicated()", "high", lambda o: o["high"].duplicated),
    Call(
        'df.drop_duplicates(["k", "ks"])',
        "df",
        lambda o: functools.partial(o["df"].drop_duplicates, ["k", "ks"]),
        lambda o: functools.partial(o["df"].drop_duplicates, subset=["k", "ks"]),
        strings=True,
    ),
    Call(
        "high.unique_values()",
        "high",
        lambda o: o["high"].unique_values,
        lambda o: lambda s=o["high"]: _np_sort(s.unique()),
        note="sd는 정렬한 값, pandas는 unique() 뒤 np.sort",
    ),
    Call("f.mean()", "f", lambda o: o["f"].mean, fresh=False),
    Call("f.std()", "f", lambda o: o["f"].std, _method("f", "std", ddof=0), fresh=False, note="pandas는 ddof=0"),
    Call("f.min()", "f", lambda o: o["f"].min, fresh=False),
    Call("f.max()", "f", lambda o: o["f"].max, fresh=False),
    Call(
        'df[df["f"] > 0]',
        "df",
        lambda o: lambda df=o["df"]: df[df["f"] > 0],
        fresh=False,
        note="4열(문자열 하나)",
        strings=True,
    ),
    # key distribution
    Call('df.groupby("k4")["f"].sum()', "dfa", _gb("dfa", "k4", "f", "sum"), note="정수 키 4가지", sizes=BIG),
    Call("cust.nunique()", "cust", lambda o: o["cust"].nunique, note="customer_000123 꼴 n/10가지", strings=True),
    Call(
        'df.groupby("kc")["f"].sum()',
        "dfc",
        _gb("dfc", "kc", "f", "sum"),
        note="customer_000123 꼴 n/10가지",
        strings=True,
    ),
    Call(
        "cust.sort_values()",
        "cust",
        lambda o: o["cust"].sort_values,
        _method("cust", "sort_values", **STABLE),
        note='접두사가 같은 긴 문자열, pandas는 kind="stable"',
        strings=True,
    ),
    # less common calls
    Call("concat([df, df])", "dfkf", _concat_sd, _concat_pd, fresh=False, note="2n행", sizes=BIG),
    Call(
        'digits.astype("int64")',
        "digits",
        lambda o: functools.partial(o["digits"].astype, "int64"),
        note="숫자 문자열을 int로",
        sizes=BIG,
        strings=True,
    ),
    Call(
        "shuffled.sort_index()",
        "shuffled",
        lambda o: o["shuffled"].sort_index,
        _method("shuffled", "sort_index", **STABLE),
        note='라벨이 섞인 Series, pandas는 kind="stable"',
        sizes=BIG,
    ),
]


# ---------------------------------------------------------------- read_csv

# (name, file kind, sd keyword arguments, pandas keyword arguments (None: as sd's), sizes,
#  has a pyarrow-engine column, note)
_SD_DTYPES = {"id": "int64", "x": "float64", "name": "string", "flag": "bool"}
_PD_DTYPES = {"id": "int64", "x": "float64", "name": "str", "flag": "bool"}
CSV_CALLS = [
    ("read_csv(floats.csv)", "floats", {}, None, BASE, True, "float 열 하나"),
    ("read_csv(mixed.csv)", "mixed", {}, None, BASE, True, "int·float·문자열·bool 열 넷"),
    ("read_csv(ints.csv)", "ints", {}, None, MID, True, "int 열 둘"),
    ("read_csv(quoted.csv)", "quoted", {}, None, MID, True, "쉼표가 든 따옴표 문자열 열"),
    (
        'read_csv(mixed.csv, usecols=["id", "x"])',
        "mixed",
        {"usecols": ["id", "x"]},
        None,
        MID,
        True,
        "넷 중 둘만",
    ),
    ("read_csv(mixed.csv, dtype=...)", "mixed", {"dtype": _SD_DTYPES}, {"dtype": _PD_DTYPES}, MID, True, "넷 다 지정"),
    (
        "read_csv(mixed.csv, nrows=n)",
        "mixed",
        "nrows",
        None,
        MID,
        False,
        "sd는 nrows가 있으면 한 청크(직렬)로 읽는다, pyarrow 엔진은 nrows가 없다",
    ),
]
_CSV_STRINGS = {"mixed", "quoted"}


def _csv_paths(kind, n, k):
    """k CSV files (seeds 0..k-1) of `kind` at n rows, written once into build/speed/frame-data."""
    import numpy as np
    import pandas as pd

    d = _data_dir()
    out = []
    for seed in range(k):
        path = d / f"{kind}-{n}-{seed}.csv"
        if not path.exists():
            raw = _Raw(n, seed)
            if kind == "floats":
                frame = pd.DataFrame({"x": raw.f})
            elif kind == "mixed":
                frame = pd.DataFrame({"id": np.arange(n), "x": raw.f, "name": raw.strs, "flag": raw.f > 0})
            elif kind == "ints":
                frame = pd.DataFrame({"id": np.arange(n), "v": raw.i})
            else:  # quoted: every text field holds a comma, so to_csv quotes it
                text = [f"{a}, {b}" for a, b in zip(raw.strs, raw.cust)]
                frame = pd.DataFrame({"id": np.arange(n), "text": text, "x": raw.f})
            tmp = path.with_suffix(".tmp")
            frame.to_csv(tmp, index=False)
            tmp.rename(path)
        out.append(str(path))
    return out


def _read_csv(lib, kw, strings):
    def make(d):
        if lib == "sd":
            import stardust as sd

            fn, extra = sd.frame.read_csv, {}
        else:
            import pandas as pd

            fn, extra = pd.read_csv, ({"engine": "pyarrow"} if lib == "pyarrow" else {})
        args = {**kw, **extra}
        if args.get("nrows") == "n":
            args["nrows"] = d["n"]
        calls = [functools.partial(fn, p, **args) for p in d["paths"]]
        if lib == "pystr" and strings:
            calls = [_python_strings(c) for c in calls]
        return rotate(calls)

    return make


def _python_strings(fn):
    """fn() with pandas inferring python-storage strings (about 3 µs a call)."""
    import pandas as pd

    def call():
        with pd.option_context("mode.string_storage", "python"):
            return fn()

    return call


# ---------------------------------------------------------------- setup


def _setup(n, keys, k, libs):
    """k inputs (seeds 0..k-1) at n rows: each library's objects of `keys` per input."""
    out = {lib: [] for lib in libs}
    for seed in range(k):
        raw = _Raw(n, seed)
        for lib in libs:
            out[lib].append(_objects(lib, raw, keys))
    return out


def _factory(call, lib):
    return lambda d: rotate([call(o) for o in d[lib]])


def _wide_setup(k, libs):
    import numpy as np

    out = {lib: [] for lib in libs}
    for seed in range(k):
        rng = np.random.default_rng(seed)
        cols = {"k": rng.integers(0, 1000, WIDE_ROWS)}
        cols.update({f"c{j}": rng.standard_normal(WIDE_ROWS) for j in range(WIDE_COLS)})
        for lib in libs:
            if lib == "sd":
                import stardust as sd

                out[lib].append({"wide": sd.frame.DataFrame(cols)})
            else:
                import pandas as pd

                out[lib].append({"wide": pd.DataFrame(cols)})
    return out


WIDE_CALLS = [
    ("df.sum()", lambda o: o["wide"].sum, None, False, "열마다 합"),
    ('df.groupby("k").sum()', _gb("wide", "k", None, "sum"), None, True, "정수 키 1,000가지, 열 100개"),
    ('df[df["c0"] > 0]', lambda o: lambda df=o["wide"]: df[df["c0"] > 0], None, False, ""),
]


# ---------------------------------------------------------------- the 10-row rows


def _small_setup():
    import numpy as np
    import pandas as pd

    import stardust as sd

    a = np.random.default_rng(0).standard_normal(SMALL)
    return {
        "sd": (sd.frame.Series(a), sd.frame.DataFrame({"a": a, "b": a})),
        "pandas": (pd.Series(a), pd.DataFrame({"a": a, "b": a})),
    }


SMALL_CALLS = [
    ("s + s", lambda s, df: functools.partial(operator.add, s, s)),
    ("s.sum()", lambda s, df: s.sum),
    ('df["a"]', lambda s, df: functools.partial(operator.getitem, df, "a")),
    ('df[df["a"] > 0]', lambda s, df: lambda df=df: df[df["a"] > 0]),
]
SMALL_QUICK = {"s + s", 'df[df["a"] > 0]'}

QUICK = {
    ("strs.nunique()", 1_000),
    ('df.groupby("k")["f"].sum()', 1_000),
    ("f[f > 0]", 100_000),
    ("high.nunique()", 100_000),
    ('df.groupby("ks")["f"].sum()', 100_000),
    ("strs.sort_values()", 100_000),
    ("read_csv(mixed.csv)", 100_000),
    ('df.groupby("k")["f_na"].mean()', 100_000),
    ("f + g", 1_000_000),
    ("f.sum()", 1_000_000),
    ('df.groupby("k")["f"].sum()', 1_000_000),
    ('merge(left, right, on="k")', 1_000_000),
    ('merge(left, right, on="k", how="left")', 1_000_000),
    ("high.sort_values()", 1_000_000),
}


def _fresh_note(note, k):
    if k == 1:
        return note
    return f"{note}; 새 입력 {k}개를 돌려 가며 부른다" if note else f"새 입력 {k}개를 돌려 가며 부른다"


def _refs(sdc_or_pdc, strings):
    refs = [Ref("pandas", _factory(sdc_or_pdc, "pandas"))]
    if strings:
        refs.append(Ref("pandas python str", _factory(sdc_or_pdc, "pystr"), lib="pandas"))
    return tuple(refs)


# The section title of each group in bench/frame/speed.md (the group is the English key in
# the case ids).
TITLES = {
    "rows-10": "10행: 호출 한 번의 비용",
    "rows-1e3": "1,000행",
    "rows-1e5": "100,000행",
    "rows-262144": "262,144행",
    "rows-1e6": "1,000,000행",
    "wide-1e5x100": "100,000행 × 100열",
}


def cases() -> list[Case]:
    out = []
    for name, call in SMALL_CALLS:
        out.append(
            Case(
                area="frame",
                group=SMALL_GROUP,
                name=name,
                size=f"{SMALL} rows",
                n=SMALL,
                setup=_small_setup,
                sd=lambda d, call=call: call(*d["sd"]),
                refs=(Ref("pandas", lambda d, call=call: call(*d["pandas"])),),
                quick=name in SMALL_QUICK,
            )
        )
    for n in SIZES:
        for c in CALLS:
            if n not in c.sizes:
                continue
            k = fresh_count(n) if c.fresh else 1
            libs = ("sd", "pandas", "pystr") if c.strings else ("sd", "pandas")
            out.append(
                Case(
                    area="frame",
                    group=GROUP[n],
                    name=c.name,
                    size=f"{n:,} rows",
                    n=n,
                    setup=lambda n=n, keys=tuple(c.keys.split()), k=k, libs=libs: _setup(n, keys, k, libs),
                    sd=_factory(c.sd, "sd"),
                    refs=_refs(c.pd or c.sd, c.strings),
                    note=_fresh_note(c.note, k),
                    quick=(c.name, n) in QUICK,
                )
            )
        for name, kind, sdkw, pdkw, sizes, arrow, note in CSV_CALLS:
            if n not in sizes:
                continue
            k = fresh_count(n)
            strings = kind in _CSV_STRINGS
            if sdkw == "nrows":
                sdkw = pdkw = {"nrows": "n"}
            refs = [Ref("pandas", _read_csv("pandas", pdkw if pdkw is not None else sdkw, strings))]
            if strings:
                refs.append(
                    Ref(
                        "pandas python str",
                        _read_csv("pystr", pdkw if pdkw is not None else sdkw, strings),
                        lib="pandas",
                    )
                )
            if arrow:
                refs.append(
                    Ref(
                        "pandas pyarrow engine",
                        _read_csv("pyarrow", pdkw if pdkw is not None else sdkw, strings),
                        lib="pandas",
                    )
                )
            out.append(
                Case(
                    area="frame",
                    group=GROUP[n],
                    name=name,
                    size=f"{n:,} rows",
                    n=n,
                    setup=lambda n=n, kind=kind, k=k: {"paths": _csv_paths(kind, n, k), "n": n},
                    sd=_read_csv("sd", sdkw, strings),
                    refs=tuple(refs),
                    note=_fresh_note(note, k),
                    quick=(name, n) in QUICK,
                )
            )
    for name, sdc, pdc, fresh, note in WIDE_CALLS:
        k = fresh_count(WIDE_ROWS * WIDE_COLS) if fresh else 1
        out.append(
            Case(
                area="frame",
                group=WIDE_GROUP,
                name=name,
                size=f"({WIDE_ROWS:,}, {WIDE_COLS})",
                n=WIDE_ROWS * WIDE_COLS,
                setup=lambda k=k: _wide_setup(k, ("sd", "pandas")),
                sd=_factory(sdc, "sd"),
                refs=_refs(pdc or sdc, False),
                note=_fresh_note(note, k),
            )
        )
    return out
