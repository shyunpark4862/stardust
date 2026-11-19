"""sd.frame against pandas: the final case set of bench/frame/final_compare.py (15 calls at
1,000, 100,000 and 1,000,000 rows, and four per-call-overhead rows at 10 rows), with its
inputs (data(n, seed)). The post-merge measurement of the same rows is in
bench/merge/m8/final/ (frame-new*.json, frame-pandas.json). Polars, a column of the old
script, is not a comparator here.

pandas runs on one thread. Hash and sort calls (nunique, groupby, merge, sort_values)
at 1,000 and 100,000 rows take a different input of the same shape each call, cycling
through fresh(n) inputs (final_compare.py --fresh, bench/frame/_harness.py
per_call_ns_cycle): on one input called over and over the branch predictor learns that
input's branches and the hash loops look faster than they are (docs/design/frame.md
§12.1); at 1,000,000 rows it cannot, so those take one input. Both libraries get the
same inputs in the same order."""

from __future__ import annotations

import functools
import itertools
import operator

from ..registry import Case, Ref

SIZES = [1_000, 100_000, 1_000_000]
GROUP = {1_000: "1,000행", 100_000: "100,000행", 1_000_000: "1,000,000행"}
SMALL = 10
SMALL_GROUP = "10행: 호출 한 번의 비용"
FRESH_BELOW = 1_000_000


def fresh(n):
    """Inputs per size in fresh mode, as final_compare.py: many small ones, a few big ones."""
    return min(64, max(4, 4_000_000 // n))


def _data_dir():
    from ..results import RUN_DIR

    d = RUN_DIR / "frame-data"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _Raw:
    """final_compare.py's data(n, seed), the strings made only when asked for."""

    def __init__(self, n, seed):
        import numpy as np

        rng = np.random.default_rng(seed)
        self.n = n
        self.low, self.high = rng.integers(0, 1000, n), rng.integers(0, n, n)
        self._codes = rng.integers(0, max(n // 10, 1), n)
        self.f, self.g = rng.standard_normal(n), rng.standard_normal(n)
        m = max(n // 10, 1)
        self.lk, self.rk, self.rb = rng.integers(0, m + m // 9, n), rng.permutation(m), rng.standard_normal(m)

    @functools.cached_property
    def strs(self):
        return [f"key{v}" for v in self._codes.tolist()]


def _objects(lib, raw, keys):
    """The Series / DataFrames of `keys` in `lib` ("sd" or "pandas")."""
    if lib == "sd":
        import stardust as sd

        Series, DataFrame = sd.frame.Series, sd.frame.DataFrame
    else:
        import pandas as pd

        Series, DataFrame = pd.Series, pd.DataFrame
    make = {
        "f": lambda: Series(raw.f),
        "g": lambda: Series(raw.g),
        "low": lambda: Series(raw.low),
        "high": lambda: Series(raw.high),
        "str": lambda: Series(raw.strs),
        "df": lambda: DataFrame({"k": raw.low, "kh": raw.high, "ks": raw.strs, "f": raw.f}),
        "L": lambda: DataFrame({"k": raw.lk, "a": raw.f}),
        "R": lambda: DataFrame({"k": raw.rk, "b": raw.rb}),
    }
    return {k: make[k]() for k in keys}


def _csvs(n):
    """final_compare.py's CSV inputs at n rows (seed 0), written once into build/speed/frame-data."""
    import numpy as np

    d = _data_dir()
    fpath, mpath = d / f"f{n}.csv", d / f"m{n}.csv"
    if not (fpath.exists() and mpath.exists()):
        import pandas as pd

        raw = _Raw(n, 0)
        pd.DataFrame({"x": raw.f}).to_csv(fpath, index=False)
        pd.DataFrame({"id": np.arange(n), "x": raw.f, "name": raw.strs, "flag": raw.f > 0}).to_csv(mpath, index=False)
    return {"floats": str(fpath), "mixed": str(mpath)}


def _cycle(fns):
    """One callable that calls fns in turn (per_call_ns_cycle)."""
    if len(fns) == 1:
        return fns[0]
    nxt = itertools.cycle(fns).__next__
    return lambda: nxt()()


# ---------------------------------------------------------------- calls

# (name, objects it needs, sd call on the objects, pandas call, hash/sort (fresh inputs), note)
# A call is o -> zero-argument callable; a bound method or a partial where it can be.


def _sd_merge(o):
    import stardust as sd

    return functools.partial(sd.frame.merge, o["L"], o["R"], on="k")


CALLS = [
    ("f + g", "f g", lambda o: functools.partial(operator.add, o["f"], o["g"]), None, False, ""),
    ("f.sum()", "f", lambda o: o["f"].sum, None, False, ""),
    ("f[f > 0]", "f", lambda o: lambda f=o["f"]: f[f > 0], None, False, ""),
    ("low.nunique()", "low", lambda o: o["low"].nunique, None, True, "정수 키 1,000가지"),
    ("high.nunique()", "high", lambda o: o["high"].nunique, None, True, "정수 키 약 n가지"),
    ("strs.nunique()", "str", lambda o: o["str"].nunique, None, True, "문자열 키 n/10가지"),
    (
        'df.groupby("k")["f"].sum()',
        "df",
        lambda o: lambda df=o["df"]: df.groupby("k")["f"].sum(),
        None,
        True,
        "정수 키 1,000가지",
    ),
    (
        'df.groupby("kh")["f"].sum()',
        "df",
        lambda o: lambda df=o["df"]: df.groupby("kh")["f"].sum(),
        None,
        True,
        "정수 키 약 n가지",
    ),
    (
        'df.groupby("ks")["f"].sum()',
        "df",
        lambda o: lambda df=o["df"]: df.groupby("ks")["f"].sum(),
        None,
        True,
        "문자열 키 n/10가지",
    ),
    (
        'df.groupby(["k", "ks"])["f"].sum()',
        "df",
        lambda o: lambda df=o["df"]: df.groupby(["k", "ks"])["f"].sum(),
        None,
        True,
        "정수 1,000가지 × 문자열",
    ),
    (
        'merge(left, right, on="k")',
        "L R",
        _sd_merge,
        lambda o: functools.partial(o["L"].merge, o["R"], on="k"),
        True,
        "오른쪽 n/10행, inner",
    ),
    (
        "high.sort_values()",
        "high",
        lambda o: o["high"].sort_values,
        lambda o: functools.partial(o["high"].sort_values, kind="stable"),
        True,
        'pandas는 kind="stable"',
    ),
    (
        "strs.sort_values()",
        "str",
        lambda o: o["str"].sort_values,
        lambda o: functools.partial(o["str"].sort_values, kind="stable"),
        True,
        'pandas는 kind="stable"',
    ),
]
CSV_CALLS = [
    ("read_csv(floats.csv)", "floats", "float 열 하나"),
    ("read_csv(mixed.csv)", "mixed", "int·float·문자열·bool 열 넷"),
]

QUICK = {
    ("strs.nunique()", 1_000),
    ('df.groupby("k")["f"].sum()', 1_000),
    ("f[f > 0]", 100_000),
    ("high.nunique()", 100_000),
    ('df.groupby("ks")["f"].sum()', 100_000),
    ("strs.sort_values()", 100_000),
    ("read_csv(mixed.csv)", 100_000),
    ("f + g", 1_000_000),
    ("f.sum()", 1_000_000),
    ('df.groupby("k")["f"].sum()', 1_000_000),
    ('merge(left, right, on="k")', 1_000_000),
    ("high.sort_values()", 1_000_000),
}


def _setup(n, keys, k):
    """k inputs (seeds 0..k-1) at n rows: each library's objects of `keys` per input."""
    out = {"sd": [], "pandas": []}
    for seed in range(k):
        raw = _Raw(n, seed)
        out["sd"].append(_objects("sd", raw, keys))
        out["pandas"].append(_objects("pandas", raw, keys))
    return out


def _factory(call, lib):
    return lambda d: _cycle([call(o) for o in d[lib]])


def _csv_setup(n):
    return _csvs(n)


def _read_csv(lib, which):
    def make(d):
        if lib == "sd":
            import stardust as sd

            return functools.partial(sd.frame.read_csv, d[which])
        import pandas as pd

        return functools.partial(pd.read_csv, d[which])

    return make


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


def cases() -> list[Case]:
    out = []
    for name, call in SMALL_CALLS:
        out.append(
            Case(
                area="frame",
                group=SMALL_GROUP,
                name=name,
                n=SMALL,
                setup=_small_setup,
                sd=lambda d, call=call: call(*d["sd"]),
                refs=(Ref("pandas", lambda d, call=call: call(*d["pandas"])),),
                quick=name in SMALL_QUICK,
            )
        )
    for n in SIZES:
        for name, keys, sdc, pdc, hashy, note in CALLS:
            k = fresh(n) if hashy and n < FRESH_BELOW else 1
            if k > 1:
                note = f"{note}; 새 입력 {k}개를 돌려 가며 부른다" if note else f"새 입력 {k}개를 돌려 가며 부른다"
            out.append(
                Case(
                    area="frame",
                    group=GROUP[n],
                    name=name,
                    n=n,
                    setup=lambda n=n, keys=tuple(keys.split()), k=k: _setup(n, keys, k),
                    sd=_factory(sdc, "sd"),
                    refs=(Ref("pandas", _factory(pdc or sdc, "pandas")),),
                    note=note,
                    quick=(name, n) in QUICK,
                )
            )
        for name, which, note in CSV_CALLS:
            out.append(
                Case(
                    area="frame",
                    group=GROUP[n],
                    name=name,
                    n=n,
                    setup=lambda n=n: _csv_setup(n),
                    sd=_read_csv("sd", which),
                    refs=(Ref("pandas", _read_csv("pandas", which)),),
                    note=note,
                    quick=(name, n) in QUICK,
                )
            )
    return out
