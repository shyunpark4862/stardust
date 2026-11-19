"""Golden cases for the frame area: stardust_pandas, later stardust.frame
(see golden.py for the contract; the spec is stardust-pandas/API.md, the frozen repository).

Inputs are deterministic (fixed literals, L.rng(seed)) and built lazily per
cases(L) call, so every thread count gets fresh objects. A case that assigns
builds or copies its own objects first, so no case can change another's input.

Naming: a case whose name starts with "err " is an intended error case;
every other case must return a value (at recording time the two sets
matched exactly).

read_csv inputs are written to CSV_DIR (under the system temp directory),
one file per distinct content, named by its hash.

Sizes: "small" inputs are literals (~10-1000 rows); "big" inputs have
N = 300_007 or H = 1_000_003 rows, above the parallel threshold (2**17 rows)
and over many 2**14-row chunks.
"""

import functools
import hashlib
import json
import math
import operator
import pathlib
import tempfile

import numpy as np

EMBED = 32  # columns up to this length are embedded value by value, longer ones hashed
N = 300_007
H = 1_000_003
CSV_DIR = pathlib.Path(tempfile.gettempdir()) / "stardust-golden-frame"

# --------------------------------------------------------------------------
# encode: a JSON-able form of the frame objects, exact to the bit, through
# the public API only.


def _sha(b):
    return hashlib.sha256(b).hexdigest()[:24]


def _fkey(v):
    v = float(v)
    return v.hex() if not math.isnan(v) else "nan:" + np.float64(v).tobytes().hex()


def _jsha(obj):
    return _sha(json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode())


_FILL = {"bool": False, "int64": 0, "float64": 0.0, "string": None}


def _column(s):
    """dtype, length, NA mask and values of a Series (not its name or index)."""
    dt = str(s.dtype)
    n = len(s)
    mask = np.asarray(s.isna().to_numpy())
    nulls = int(mask.sum())
    out = {"dtype": dt, "len": n, "nulls": nulls}
    arr = s.to_numpy(na_value=_FILL.get(dt, None))
    if dt == "string":
        vals = [None if m else v for v, m in zip(arr.tolist(), mask.tolist())]
        if n <= EMBED:
            out["values"] = vals
        else:
            out["sha"] = _jsha(vals)
    else:
        if n <= EMBED:
            if dt == "float64":
                vals = [_fkey(v) for v in arr.tolist()]
            elif dt == "int64":
                vals = [str(v) for v in arr.tolist()]
            else:
                vals = arr.tolist()
            out["values"] = [None if m else v for v, m in zip(vals, mask.tolist())]
        else:
            out["np"] = arr.dtype.str
            out["sha"] = _sha(np.ascontiguousarray(arr).tobytes())
    if n > EMBED and nulls:
        out["mask"] = _sha(np.packbits(mask.astype(bool)).tobytes())
    return out


def _index(x, F):
    if isinstance(x, F.RangeIndex):
        return {"RangeIndex": [x.start, x.stop, x.step], "name": x.name, "dtype": str(x.dtype), "len": len(x)}
    if isinstance(x, F.Index):
        labels = _tl(x)
        dt = str(x.dtype)  # a name (old package) or a dtype object (M6): the same text
        out = {"Index": dt, "name": x.name, "len": len(x)}
        if len(labels) <= EMBED:
            out["labels"] = [str(v) if dt == "int64" else v for v in labels]
        elif dt == "int64":
            out["sha"] = _sha(np.asarray(labels, dtype=np.int64).tobytes())
        else:
            out["sha"] = _jsha(labels)
        return out
    return None


def _obj(v, F):  # an element of an object array
    if v is None or isinstance(v, (bool, str)):
        return v
    if v is F.NA:
        return {"NA": True}
    if isinstance(v, int):
        return {"i": str(v)}
    if isinstance(v, float):
        return {"f": _fkey(v)}
    return {"type": type(v).__name__, "repr": repr(v)}


def encode(x, L):
    F = L.F
    if type(x).__name__ == "DType":  # a dtype object (M6) is encoded as its name, as the old package's strings
        return str(x)
    if x is F.NA:
        return {"NA": True}
    if isinstance(x, F.Series):
        return {"Series": _column(x), "name": x.name, "index": _index(x.index, F)}
    if isinstance(x, F.DataFrame):
        names = _tl(x.columns)
        return {
            "DataFrame": names,
            "shape": list(x.shape),
            "cols": [_column(x[k]) for k in names],
            "index": _index(x.index, F),
        }
    idx = _index(x, F)
    if idx is not None:
        return idx
    if isinstance(x, np.ndarray):
        out = {
            "ndarray": str(x.dtype),
            "shape": list(x.shape),
            "strides": list(x.strides),
            "writeable": bool(x.flags.writeable),
        }
        if x.dtype == object:
            out["sha"] = _jsha([_obj(v, F) for v in x.ravel().tolist()])
        else:
            out["sha"] = _sha(np.ascontiguousarray(x).tobytes())
        return out
    if isinstance(x, np.generic):
        v = x.item()
        return {"npscalar": str(x.dtype), "v": _fkey(v) if isinstance(v, float) else repr(v)}
    if type(x).__name__ == "GroupBy":  # not expected: cases call a method instead
        return {"GroupBy": encode(x.size(), L)}
    return None


# --------------------------------------------------------------------------
# inputs

nan, inf = float("nan"), float("inf")
I64_MIN, I64_MAX = -(2**63), 2**63 - 1

SF = [1.5, None, nan, -0.0, 0.0, inf, -inf, 2.5, None, -3.25, 1e300, 5e-324, 7.0, nan, 2.5]
SF2 = [2.0, 0.0, -0.0, nan, None, inf, -2.5, 3.0, 0.5, -1.0, 1e-300, None, 4.0, -inf, 1.0]
SI = [3, None, -7, 0, I64_MAX, I64_MIN, 5, None, 3, 12, -1, 0, 7, 3, 100]
SI2 = [2, None, 0, -4, 2, -1, 3, 5, 0, 7, -1, 2, -3, 2, -6]
SB = [True, None, False, True, True, False, None, False, True, True, False, False, True, None, True]
SB2 = [True, True, None, False, True, False, False, None, True, False, True, True, False, None, False]
SS = ["b", None, "a", "", "é", "b", "가나", None, "a\nb", "zz", "B", "a", "", "😀", "b"]
SS2 = ["a", "b", None, "", "é", "c", "가", "x", "a\nb", "z", "b", None, "", "😀", "bb"]
LBL = [f"r{i}" for i in range(15)]
K3 = [True, True, True, False, False, False, None, None, None]
K4 = [True, False, None, True, False, None, True, False, None]


class _Data:
    """Big inputs, built on first use (once per cases(L) call)."""

    def __init__(self, L):
        self.L = L
        self.F = L.F

    def _na_list(self, values, mask):
        return [None if m else v for v, m in zip(values, mask)]

    @functools.cached_property
    def base(self):
        r = self.L.rng(11)
        na = (r.random(N) < 0.1).tolist()
        ints = r.integers(-1000, 1000, N).tolist()
        fl = r.standard_normal(N)
        fl[r.random(N) < 0.01] = np.nan
        fl[r.random(N) < 0.01] = -0.0
        fl[r.random(N) < 0.001] = np.inf
        words = [("é" if v % 7 == 0 else "w") + str(v) if v % 97 else "" for v in r.integers(0, 5000, N).tolist()]
        bools = (r.random(N) < 0.5).tolist()
        return {
            "na": na,
            "ints": ints,
            "floats": fl.tolist(),
            "words": words,
            "bools": bools,
            "mask": (r.random(N) < 0.3).tolist(),
            "mask_na": (r.random(N) < 0.05).tolist(),
            "perm": r.permutation(N)[: N // 2].tolist(),
        }

    @functools.cached_property
    def BI(self):
        b = self.base
        return self.F.Series(self._na_list(b["ints"], b["na"]), name="i")

    @functools.cached_property
    def BI_raw(self):  # no NA
        return self.F.Series(self.base["ints"])

    @functools.cached_property
    def BF(self):
        b = self.base
        return self.F.Series(self._na_list(b["floats"], b["na"]), name="f")

    @functools.cached_property
    def BF_raw(self):  # no NA
        return self.F.Series(np.asarray(self.base["floats"]))

    @functools.cached_property
    def BS(self):
        b = self.base
        return self.F.Series(self._na_list(b["words"], b["na"]), name="s")

    @functools.cached_property
    def BB(self):
        b = self.base
        return self.F.Series(self._na_list(b["bools"], b["na"]), name="b")

    @functools.cached_property
    def MASK(self):
        return self.F.Series(self.base["mask"])

    @functools.cached_property
    def MASK_NA(self):
        b = self.base
        return self.F.Series(self._na_list(b["mask"], b["mask_na"]))

    @functools.cached_property
    def PERM(self):
        return self.base["perm"]

    @functools.cached_property
    def HF(self):  # 1M floats, no NA
        return self.F.Series(self.L.rng(12).standard_normal(H) * 1e3)

    @functools.cached_property
    def HI(self):  # 1M large ints: the sum wraps, the mean needs int128
        return self.F.Series(self.L.rng(13).integers(-(2**62), 2**62, H))

    @functools.cached_property
    def HFN(self):  # 1M floats with NA
        r = self.L.rng(14)
        v = r.standard_normal(H).tolist()
        m = (r.random(H) < 0.2).tolist()
        return self.F.Series(self._na_list(v, m))

    @functools.cached_property
    def BDF(self):
        r = self.L.rng(21)
        F = self.F
        b = self.base
        kna = (r.random(N) < 0.01).tolist()
        k_i = self._na_list(r.integers(0, 1000, N).tolist(), kna)
        pool = [f"g{j:03d}" for j in range(200)]
        k_s = self._na_list([pool[j] for j in r.integers(0, 200, N).tolist()], (r.random(N) < 0.01).tolist())
        fpool = [-0.0, 0.0, nan, inf, -inf] + [float(j) / 4 for j in range(-20, 25)]
        k_f = self._na_list([fpool[j] for j in r.integers(0, len(fpool), N).tolist()], (r.random(N) < 0.01).tolist())
        k_b = self._na_list((r.random(N) < 0.3).tolist(), (r.random(N) < 0.01).tolist())
        return F.DataFrame(
            {
                "k_i": k_i,
                "k_s": k_s,
                "k_f": k_f,
                "k_b": k_b,
                "v_f": self._na_list(b["floats"], b["na"]),
                "v_i": self._na_list(b["ints"], b["na"]),
                "v_b": self._na_list(b["bools"], b["na"]),
                "v_s": self._na_list(b["words"], b["na"]),
            }
        )

    @functools.cached_property
    def BDF_noNaN(self):  # value columns without NaN, so sums do not just give NaN
        r = self.L.rng(22)
        F = self.F
        k = r.integers(0, 3000, N).tolist()
        k2 = [f"c{j}" for j in r.integers(0, 7, N).tolist()]
        v = r.standard_normal(N).tolist()
        m = (r.random(N) < 0.05).tolist()
        return F.DataFrame({"k": k, "k2": k2, "v": self._na_list(v, m), "w": r.integers(-5, 5, N).tolist()})

    @functools.cached_property
    def MERGE(self):
        r = self.L.rng(31)
        F = self.F
        nl, nr = N, 100_003
        lk = self._na_list(r.integers(0, 200_000, nl).tolist(), (r.random(nl) < 0.01).tolist())
        rk = self._na_list(r.integers(0, 250_000, nr).tolist(), (r.random(nr) < 0.01).tolist())
        tags = ["a", "b", "c", "d", "e"]
        left = F.DataFrame(
            {
                "key": lk,
                "tag": [tags[j] for j in r.integers(0, 5, nl).tolist()],
                "lv": r.standard_normal(nl).tolist(),
                "same": r.integers(0, 10, nl).tolist(),
            }
        )
        right = F.DataFrame(
            {
                "key": rk,
                "tag": [tags[j] for j in r.integers(0, 5, nr).tolist()],
                "rv": r.integers(-100, 100, nr).tolist(),
                "same": r.integers(0, 10, nr).tolist(),
            }
        )
        fpool = [-0.0, 0.0, nan, 1.5, -2.25, inf] + [float(j) for j in range(1000)]
        lf = F.DataFrame(
            {
                "fk": self._na_list(
                    [fpool[j] for j in r.integers(0, len(fpool), 200_003).tolist()], (r.random(200_003) < 0.01).tolist()
                ),
                "x": r.integers(0, 9, 200_003).tolist(),
            }
        )
        rf = F.DataFrame(
            {
                "fk": [fpool[j] for j in r.integers(0, len(fpool), 5_003).tolist()],
                "y": r.standard_normal(5_003).tolist(),
            }
        )
        spool = [f"name{j}" for j in range(150_000)]
        ls = F.DataFrame(
            {
                "s": self._na_list(
                    [spool[j] for j in r.integers(0, 150_000, nl).tolist()], (r.random(nl) < 0.01).tolist()
                ),
                "a": r.integers(0, 5, nl).tolist(),
            }
        )
        rs = F.DataFrame(
            {"s": [spool[j] for j in r.integers(0, 150_000, 120_000).tolist()], "b": r.integers(0, 5, 120_000).tolist()}
        )
        return left, right, lf, rf, ls, rs

    @functools.cached_property
    def ALIGN(self):
        r = self.L.rng(41)
        F = self.F
        la = r.permutation(N).tolist()
        lb = (r.permutation(N) + N // 2).tolist()
        a = F.Series(r.standard_normal(N), index=la, name="v")
        b = F.Series(r.standard_normal(N), index=lb, name="v")
        sa = F.Series(r.integers(0, 100, N), index=[f"k{j}" for j in la])
        sb = F.Series(r.integers(0, 100, N), index=[f"k{j}" for j in lb])
        return a, b, sa, sb

    @functools.cached_property
    def BIG_CSV(self):
        return _big_csv(self.L.rng(51), 150_000)


def _big_csv(r, rows):
    """A ~7 MB CSV text exercising every inference path and RFC 4180 quoting."""
    ids = r.permutation(rows).tolist()
    xs = r.standard_normal(rows).tolist()
    sel = r.integers(0, 20, rows).tolist()
    flags = r.integers(0, 3, rows).tolist()
    nm = r.integers(0, 10, rows).tolist()
    ints = r.integers(-(10**15), 10**15, rows).tolist()
    out = ["id,x,flag,name,qint,big,nanish,i"]
    specials = ["nan", "inf", "-inf", "-0.0", "1e300", "NaN", "+5", ".5", "5.", "1E-5"]
    for k in range(rows):
        s = sel[k]
        x = specials[s] if s < len(specials) and k % 13 == 0 else ("" if s == 19 else repr(xs[k]))
        flag = ("True", "False", "")[flags[k]]
        n = nm[k]
        if n == 0:
            name = '"he said ""hi"", twice"'
        elif n == 1:
            name = '"line one\nline two"'
        elif n == 2:
            name = '""'
        elif n == 3:
            name = ""
        elif n == 4:
            name = f"é{k}"
        elif n == 5:
            name = '"a,b\r\nc"'
        else:
            name = f"n{k % 1000}"
        qint = f'"{ints[k]}"' if k % 2 else str(ints[k])
        big = "18446744073709551616" if k == rows - 7 else str(ints[k])
        nanish = "NaN" if k % 101 == 0 else str(ints[k] % 1000)
        i = "" if k % 17 == 0 else str(k)
        out.append(",".join([str(ids[k]), x, flag, name, qint, big, nanish, i]))
    return "\n".join(out) + "\n"


def _csv_path(text):
    """Write `text` (str or bytes) under CSV_DIR, named by its hash, and give the path."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    p = CSV_DIR / (hashlib.sha256(data).hexdigest()[:20] + ".csv")
    if not p.exists() or p.read_bytes() != data:
        p.write_bytes(data)
    return p


# --------------------------------------------------------------------------
# cases


ARITH = [
    ("add", operator.add),
    ("sub", operator.sub),
    ("mul", operator.mul),
    ("truediv", operator.truediv),
    ("floordiv", operator.floordiv),
    ("mod", operator.mod),
    ("pow", operator.pow),
]
COMPARE = [
    ("eq", operator.eq),
    ("ne", operator.ne),
    ("lt", operator.lt),
    ("le", operator.le),
    ("gt", operator.gt),
    ("ge", operator.ge),
]
LOGIC = [("and", operator.and_), ("or", operator.or_), ("xor", operator.xor)]
REDUCTIONS = ["sum", "mean", "min", "max", "std", "var"]


# M6 renamed some arguments and methods (DESIGN.md §9.1, §9.3, §9.4); the
# cases spell the old ones and the helpers here translate them for stardust,
# so both packages compute the same thing. Main arguments are positional-only
# in M6, so the cases pass them by position (the old package takes both).
_NEW = [False]


def _negate(v):
    """descending is not ascending: one bool or one per key; other values (errors) stay."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, (list, tuple)):
        return type(v)(not x if isinstance(x, bool) else x for x in v)
    return v


def _kw(kw):
    if not _NEW[0] or kw is None:
        return kw
    out = {}
    for k, v in kw.items():
        if k == "ddof":
            out["correction"] = v
        elif k == "ascending":
            out["descending"] = _negate(v)
        else:
            out[k] = v
    return out


def _tl(x):
    """x.to_list() of the old package, x.tolist() in M6 (§9.4)."""
    return x.tolist() if _NEW[0] else x.to_list()


def _uv(s):
    """s.unique() of the old package, s.unique_values() in M6 (§9.4: sorted order)."""
    return s.unique_values() if _NEW[0] else s.unique()


def _red(obj, op, **kw):
    return getattr(obj, op)(**_kw(kw))


def cases(L):
    out = []
    add = lambda name, fn: out.append((name, fn))  # noqa: E731
    F, np_ = L.F, L.np
    NA = F.NA
    _NEW[0] = F.__name__ != "stardust_pandas"
    D = _Data(L)

    def S(values, **kw):
        return lambda: F.Series(list(values), **kw)

    sf, sf2, si, si2 = S(SF, name="f"), S(SF2, name="f2"), S(SI, name="i"), S(SI2, name="i2")
    sb, sb2, ss, ss2 = S(SB, name="b"), S(SB2, name="b2"), S(SS, name="s"), S(SS2, name="s2")
    small = {"f": sf, "i": si, "b": sb, "s": ss}

    def sdf():
        return F.DataFrame({"f": list(SF), "i": list(SI), "b": list(SB), "s": list(SS)})

    def sdf_l():
        return F.DataFrame({"f": list(SF), "i": list(SI), "b": list(SB), "s": list(SS)}, index=list(LBL))

    # ---------------------------------------------------------------- construction (§10, §14)
    add("ctor Series()", lambda: F.Series())
    add("ctor Series([])", lambda: F.Series([]))
    add("ctor Series(None, index)", lambda: F.Series(None, index=[1, 2, 3]))
    add("ctor Series(None, index, dtype)", lambda: F.Series(None, index=["a", "b"], dtype="string"))
    for k, mk in small.items():
        add(f"ctor small {k}", mk)
        add(f"ctor small {k} labelled", lambda mk=mk: F.Series(_tl(mk()), index=list(LBL)))
    add("ctor small f2", sf2)
    add("ctor small i2", si2)
    add("ctor int+float", lambda: F.Series([1, 2.5, None]))
    add("ctor nan and NA", lambda: F.Series([nan, None, NA, -0.0]))
    add("ctor all NA", lambda: F.Series([None, NA]))
    add("ctor range", lambda: F.Series(range(3, 30, 4)))
    add("ctor tuple", lambda: F.Series((True, False)))
    add("ctor 2**53 int+float exact", lambda: F.Series([2**53, 0.5]))
    add("ctor int64 bounds", lambda: F.Series([I64_MIN, I64_MAX]))
    add("ctor dtype float from int", lambda: F.Series([1, 2, None], dtype=float))
    add("ctor dtype int from float", lambda: F.Series([1.0, -2.0, None], dtype="int64"))
    add("ctor dtype string", lambda: F.Series(["x", None], dtype=str))
    add("ctor dtype bool", lambda: F.Series([True, None], dtype="bool"))
    add("ctor name", lambda: F.Series([1], name="nm"))
    add("ctor index list", lambda: F.Series([1, 2, 3], index=[30, 10, 20]))
    add("ctor index Index", lambda: F.Series([1, 2], index=F.Index(["a", "b"], name="ix")))
    add("ctor index RangeIndex", lambda: F.Series([1, 2, 3], index=F.RangeIndex(10, 16, 2, name="r")))
    add("ctor from Series reindex", lambda: F.Series(F.Series([1, 2, 3], index=["a", "b", "c"]), index=["c", "z", "a"]))
    add("ctor from Series dtype", lambda: F.Series(F.Series([1, 2], name="q"), dtype="float64"))
    add("ctor from Series name", lambda: F.Series(F.Series([1, 2], name="q"), name="r"))
    add("err ctor float to int lossy", lambda: F.Series([1.5], dtype="int64"))
    add("err ctor nan to int", lambda: F.Series([nan], dtype="int64"))
    add("err ctor bool+int", lambda: F.Series([True, 1]))
    add("err ctor str+int", lambda: F.Series(["a", 1]))
    add("err ctor int too big", lambda: F.Series([2**63]))
    add("err ctor int+float inexact", lambda: F.Series([2**53 + 1, 0.5]))
    add("err ctor dtype float inexact", lambda: F.Series([2**53 + 1], dtype="float64"))
    add("err ctor dtype bool from int", lambda: F.Series([1, 0], dtype="bool"))
    add("err ctor dtype string from int", lambda: F.Series([1], dtype="string"))
    add("err ctor dict", lambda: F.Series({"a": 1}))
    add("err ctor dtype unknown", lambda: F.Series([1], dtype="int32"))
    add("err ctor name not str", lambda: F.Series([1], name=1))
    add("err ctor index length", lambda: F.Series([1, 2], index=[1, 2, 3]))
    add("err ctor reindex range to string labels", lambda: F.Series(F.Series([1, 2]), index=["a", "b"]))
    add("err ctor index float labels", lambda: F.Series([1], index=[1.5]))
    add("err ctor scalar", lambda: F.Series(5))
    for dt in [
        "int8",
        "int16",
        "int32",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "int64",
        "float64",
        "bool",
    ]:
        add(f"ctor numpy {dt}", lambda dt=dt: F.Series(np_.array([0, 1, 7, 100], dtype=dt)))
    add("ctor numpy uint64 max int64", lambda: F.Series(np_.array([2**63 - 1], dtype="uint64")))
    add("err ctor numpy uint64 too big", lambda: F.Series(np_.array([2**63], dtype="uint64")))
    add("ctor numpy U", lambda: F.Series(np_.array(["x", "yy", "é"])))
    add("ctor numpy object", lambda: F.Series(np_.array([1, None, 3], dtype=object)))
    add("ctor numpy float nan -0", lambda: F.Series(np_.array([nan, -0.0, 1.0])))
    add("ctor numpy big-endian", lambda: F.Series(np_.array([1.5, 2.5], dtype=">f8")))
    add("ctor numpy copies", lambda: (lambda a: (lambda s: (a.__setitem__(0, 99), s)[1])(F.Series(a)))(np_.arange(5)))
    add("err ctor numpy 2d", lambda: F.Series(np_.zeros((2, 2))))
    add("err ctor numpy complex", lambda: F.Series(np_.array([1j])))
    add("ctor big from list NA float", lambda: D.BF)
    add("ctor big from list NA int", lambda: D.BI)
    add("ctor big from list NA string", lambda: D.BS)
    add("ctor big from list NA bool", lambda: D.BB)
    add("ctor big from numpy float", lambda: D.HF)
    add("ctor big from numpy int", lambda: D.HI)
    add("ctor big index", lambda: F.Series(D.BI, index=list(range(N - 1, -1, -1))))

    add("ctor DataFrame()", lambda: F.DataFrame())
    add("ctor DataFrame columns only", lambda: F.DataFrame(columns=["a", "b"]))
    add("ctor DataFrame index only", lambda: F.DataFrame(index=[1, 2]))
    add("ctor DataFrame dict", sdf)
    add("ctor DataFrame dict labelled", sdf_l)
    add("ctor DataFrame dict numpy", lambda: F.DataFrame({"a": np_.arange(4), "b": np_.linspace(0, 1, 4)}))
    add(
        "ctor DataFrame dict series aligned",
        lambda: F.DataFrame(
            {"a": F.Series([1, 2, 3], index=["x", "y", "z"]), "b": F.Series([10.0, 20.0], index=["w", "x"])}
        ),
    )
    add(
        "ctor DataFrame dict series + index",
        lambda: F.DataFrame({"a": F.Series([1, 2, 3], index=["x", "y", "z"])}, index=["z", "q"]),
    )
    add("ctor DataFrame dict scalar broadcast", lambda: F.DataFrame({"a": [1, 2, 3], "c": "k", "d": 1.5}))
    add("ctor DataFrame dict dtype", lambda: F.DataFrame({"a": [1, 2], "b": [3, None]}, dtype="float64"))
    add("ctor DataFrame dict columns select", lambda: F.DataFrame({"a": [1, 2], "b": [3, 4]}, columns=["b", "zz"]))
    add("ctor DataFrame records", lambda: F.DataFrame([{"a": 1, "b": "x"}, {"b": "y", "c": 2.5}, {}]))
    add("ctor DataFrame rows", lambda: F.DataFrame([[1, "x", True], [2, None, False]], columns=["a", "b", "c"]))
    add("ctor DataFrame rows tuples", lambda: F.DataFrame([(1, 2.5), (3, 4.0)], columns=["a", "b"]))
    add("ctor DataFrame 2d numpy", lambda: F.DataFrame(np_.arange(12.0).reshape(4, 3), columns=["a", "b", "c"]))
    add("ctor DataFrame from DataFrame", lambda: F.DataFrame(sdf(), columns=["s", "f", "new"]))
    add("ctor DataFrame from DataFrame index", lambda: F.DataFrame(sdf_l(), index=["r3", "r1", "zz"]))
    add("ctor DataFrame index labels", lambda: F.DataFrame({"a": [1, 2]}, index=["p", "q"]))
    add("ctor DataFrame empty rows", lambda: F.DataFrame([], columns=["a"]))
    add("err ctor DataFrame lengths", lambda: F.DataFrame({"a": [1, 2], "b": [1]}))
    add("err ctor DataFrame scalars only", lambda: F.DataFrame({"a": 1}))
    add("err ctor DataFrame rows no columns", lambda: F.DataFrame([[1, 2]]))
    add("err ctor DataFrame 2d no columns", lambda: F.DataFrame(np_.zeros((2, 2))))
    add("err ctor DataFrame row width", lambda: F.DataFrame([[1, 2], [3]], columns=["a", "b"]))
    add("err ctor DataFrame column name int", lambda: F.DataFrame({1: [1]}))
    add("err ctor DataFrame duplicate columns", lambda: F.DataFrame([[1, 2]], columns=["a", "a"]))
    add("err ctor DataFrame scalar", lambda: F.DataFrame(5))
    add("err ctor DataFrame mixed rows", lambda: F.DataFrame([{"a": 1}, [1]]))
    add("err ctor DataFrame 3d", lambda: F.DataFrame(np_.zeros((1, 1, 1)), columns=["a"]))
    add("err ctor DataFrame index length", lambda: F.DataFrame({"a": [1, 2]}, index=[1, 2, 3]))
    add("ctor DataFrame big", lambda: D.BDF)

    add("ctor Index str", lambda: F.Index(["a", "b", "a"]))
    add("ctor Index int name", lambda: F.Index([3, 1, 2], name="ix"))
    add("ctor Index empty", lambda: F.Index([]))
    add("ctor Index range", lambda: F.Index(range(2, 9, 3)))
    add("ctor Index from Index", lambda: F.Index(F.Index(["x"], name="a"), name="b"))
    add("ctor Index from RangeIndex", lambda: F.Index(F.RangeIndex(4)))
    add("ctor Index numpy", lambda: F.Index(np_.array([5, 6], dtype="int32")))
    add("err ctor Index float", lambda: F.Index([1.5]))
    add("err ctor Index bool", lambda: F.Index([True]))
    add("err ctor Index NA", lambda: F.Index([1, None]))
    add("err ctor Index name", lambda: F.Index([1], name=2))
    add("ctor RangeIndex stop", lambda: F.RangeIndex(5))
    add("ctor RangeIndex full", lambda: F.RangeIndex(2, 20, 3, name="r"))
    add("ctor RangeIndex negative step", lambda: F.RangeIndex(10, 0, -2))
    add("ctor RangeIndex empty", lambda: F.RangeIndex(0))
    add("err ctor RangeIndex step 0", lambda: F.RangeIndex(0, 5, 0))
    add("err ctor RangeIndex name", lambda: F.RangeIndex(3, name=1))
    add("ctor Index big", lambda: F.Index(D.base["words"]))

    # ---------------------------------------------------------------- properties, iteration, in
    for k, mk in small.items():
        add(f"prop dtype {k}", lambda mk=mk: mk().dtype)
        add(f"prop shape {k}", lambda mk=mk: mk().shape)
        add(f"prop len {k}", lambda mk=mk: len(mk()))
        add(f"prop index {k}", lambda mk=mk: mk().index)
        add(f"prop name {k}", lambda mk=mk: mk().name)
        add(f"iter {k}", lambda mk=mk: list(mk()))
        add(f"to_list {k}", lambda mk=mk: _tl(mk()))
    add("prop name set", lambda: (lambda s: (setattr(s, "name", "new"), s)[1])(F.Series([1])))
    add("prop name set None", lambda: (lambda s: (setattr(s, "name", None), s)[1])(F.Series([1], name="a")))
    add("err prop name set int", lambda: setattr(F.Series([1]), "name", 3))
    add("prop df columns", lambda: sdf().columns)
    add("prop df dtypes", lambda: sdf().dtypes)
    add("prop df index", lambda: sdf_l().index)
    add("prop df shape", lambda: sdf().shape)
    add("prop df len", lambda: len(sdf()))
    add("iter df", lambda: list(sdf()))
    add("in df", lambda: ("s" in sdf(), "zz" in sdf(), 1 in sdf()))
    add("prop big df dtypes", lambda: D.BDF.dtypes)
    add("prop big df columns", lambda: D.BDF.columns)
    add(
        "in series values",
        lambda: [x in F.Series([1.5, nan, None, -0.0]) for x in (1.5, nan, NA, None, 0.0, 2.0, "a", True)],
    )
    add("in series int", lambda: [x in F.Series([1, None, 3]) for x in (1, 1.0, 3.5, NA, "a", True, 2**70)])
    add("in series string", lambda: [x in F.Series(["a", None]) for x in ("a", "b", NA, 1)])
    add("in series bool", lambda: [x in F.Series([True, True]) for x in (True, False, NA, 1)])
    add("in big series", lambda: [x in D.BF for x in (nan, NA, 0.0, inf, 1234.5)])
    add("in big string series", lambda: [x in D.BS for x in ("w1", "é7", "", "nope", NA)])
    for nm, mk in [
        ("Index str", lambda: F.Index(["a", "b", "a"], name="n")),
        ("Index int", lambda: F.Index([3, 1, 2])),
        ("RangeIndex", lambda: F.RangeIndex(2, 20, 3, name="r")),
        ("RangeIndex neg", lambda: F.RangeIndex(10, 0, -2)),
        ("Index big", lambda: F.Index(D.base["words"])),
        ("RangeIndex big", lambda: F.RangeIndex(H)),
    ]:
        add(f"index dtype {nm}", lambda mk=mk: mk().dtype)
        add(f"index name {nm}", lambda mk=mk: mk().name)
        add(f"index is_unique {nm}", lambda mk=mk: mk().is_unique)
        add(f"index to_list {nm}", lambda mk=mk: _jsha(_tl(mk())) if len(mk()) > EMBED else _tl(mk()))
        add(f"index len {nm}", lambda mk=mk: len(mk()))
        add(f"index iter {nm}", lambda mk=mk: _jsha(list(mk())))
        add(f"index in {nm}", lambda mk=mk: [x in mk() for x in (1, 2, 5, 8, "a", "b", "w1", 1.0, True, "zz")])
    add("index start/stop/step", lambda: (lambda r: (r.start, r.stop, r.step))(F.RangeIndex(2, 20, 3)))
    add("index start/stop/step stop only", lambda: (lambda r: (r.start, r.stop, r.step))(F.RangeIndex(7)))
    add(
        "index equals",
        lambda: [
            F.Index([1, 2]).equals(F.Index([1, 2], name="x")),
            F.Index([1, 2]).equals(F.Index([2, 1])),
            F.Index([0, 1, 2]).equals(F.RangeIndex(3)),
            F.RangeIndex(3).equals(F.Index([0, 1, 2])),
            F.RangeIndex(3).equals(F.RangeIndex(0, 3, 1, name="q")),
            F.RangeIndex(0).equals(F.RangeIndex(5, 5)),
            F.Index(["a"]).equals(F.Index([1])),
            F.Index(["a"]).equals(["a"]),
            F.Index([1]).equals(F.Index([1, 1])),
        ],
    )
    add(
        "index equals big",
        lambda: [
            F.Index(D.base["words"]).equals(F.Index(list(D.base["words"]))),
            F.Index(list(range(N))).equals(F.RangeIndex(N)),
            F.Index(D.PERM).equals(F.Index(D.PERM[::-1])),
        ],
    )
    add("index is_unique big int", lambda: (F.Index(D.PERM).is_unique, F.Index(D.base["ints"]).is_unique))

    # ---------------------------------------------------------------- to_numpy, to_list (§10)
    add("to_numpy float view", lambda: F.Series([1.5, -0.0, nan]).to_numpy())
    add("to_numpy int view", lambda: F.Series([1, 2]).to_numpy())
    add("to_numpy bool view", lambda: F.Series([True, False]).to_numpy())
    add("to_numpy copy", lambda: F.Series([1.5, 2.5]).to_numpy(copy=True))
    add("to_numpy string", lambda: F.Series(["a", "é"]).to_numpy())
    add("to_numpy string NA None", lambda: F.Series(["a", None]).to_numpy(na_value=None))
    add("to_numpy string NA str", lambda: F.Series(["a", None]).to_numpy(na_value="?"))
    add("to_numpy float NA nan", lambda: F.Series([1.5, None]).to_numpy(na_value=nan))
    add("to_numpy int NA int", lambda: F.Series([1, None]).to_numpy(na_value=-1))
    add("to_numpy int NA as float", lambda: F.Series([1, None]).to_numpy(dtype="float64", na_value=nan))
    add("to_numpy bool NA", lambda: F.Series([True, None]).to_numpy(na_value=False))
    add("to_numpy dtype int->float", lambda: F.Series([1, 2]).to_numpy(dtype="float64"))
    add("to_numpy dtype float->int", lambda: F.Series([1.0, 2.0]).to_numpy(dtype="int64"))
    add("err to_numpy NA no na_value", lambda: F.Series([1, None]).to_numpy())
    add("err to_numpy int na_value nan", lambda: F.Series([1, None]).to_numpy(na_value=nan))
    add("err to_numpy bool na_value int", lambda: F.Series([True, None]).to_numpy(na_value=0))
    add("err to_numpy dtype lossy", lambda: F.Series([1.5]).to_numpy(dtype="int64"))
    add("err to_numpy view read-only", lambda: F.Series([1.0]).to_numpy().__setitem__(0, 2.0))
    add("to_numpy np.asarray", lambda: np_.asarray(F.Series([1, 2, 3])))
    add("to_numpy big float", lambda: D.HF.to_numpy())
    add("to_numpy big float NA", lambda: D.BF.to_numpy(na_value=nan))
    add("to_numpy big string NA", lambda: D.BS.to_numpy(na_value=None))
    add("to_numpy big int NA", lambda: D.BI.to_numpy(na_value=0))
    add("to_numpy big int->float", lambda: D.BI.to_numpy(dtype="float64", na_value=nan))
    add("df to_numpy int+float", lambda: F.DataFrame({"a": [1, 2], "b": [0.5, -0.0]}).to_numpy())
    add("df to_numpy strings", lambda: F.DataFrame({"a": ["x", "y"], "b": ["z", None]}).to_numpy(na_value=None))
    add("df to_numpy NA na_value", lambda: F.DataFrame({"a": [1, None], "b": [0.5, 1.5]}).to_numpy(na_value=nan))
    add("df to_numpy dtype", lambda: F.DataFrame({"a": [1, 2], "b": [3, 4]}).to_numpy(dtype="float64"))
    add("df to_numpy empty", lambda: F.DataFrame().to_numpy())
    add("df to_numpy bool", lambda: F.DataFrame({"a": [True], "b": [False]}).to_numpy())
    add("err df to_numpy mixed", lambda: sdf().to_numpy(na_value=None))
    add("err df to_numpy NA", lambda: F.DataFrame({"a": [1, None]}).to_numpy())
    add("df to_numpy big", lambda: D.BDF[["v_f", "v_i"]].to_numpy(na_value=nan))
    add("to_list big float", lambda: _jsha([_fkey(v) if isinstance(v, float) else repr(v) for v in _tl(D.BF)]))
    add("to_list big string", lambda: _jsha([v if isinstance(v, str) else repr(v) for v in _tl(D.BS)]))
    add(
        "to_list types",
        lambda: (
            [type(v).__name__ for v in _tl(F.Series([1, None]))] + [type(v).__name__ for v in _tl(F.Series([True]))]
        ),
    )

    # ---------------------------------------------------------------- head, tail
    for n in [None, 0, 3, -3, 100, -100]:
        args = () if n is None else (n,)  # n is positional-only in M6 (§9.3); the old package takes it either way
        add(f"head {n}", lambda args=args: sf().head(*args))
        add(f"tail {n}", lambda args=args: sf().tail(*args))
        add(f"df head {n}", lambda args=args: sdf_l().head(*args))
        add(f"df tail {n}", lambda args=args: sdf().tail(*args))
    add("head positional", lambda: si().head(2))
    add("head big", lambda: D.BS.head(70_000))
    add("tail big", lambda: D.BDF.tail(150_000))

    # ---------------------------------------------------------------- element-wise (§3, §12)
    pairs = [
        ("i,i2", si, si2),
        ("i,f2", si, sf2),
        ("f,i2", sf, si2),
        ("f,f2", sf, sf2),
        ("i,3", si, lambda: 3),
        ("3,i2", lambda: 3, si2),
        ("i,0", si, lambda: 0),
        ("i,-1", si, lambda: -1),
        ("f,0.0", sf, lambda: 0.0),
        ("f,-0.0", sf, lambda: -0.0),
        ("2.5,f2", lambda: 2.5, sf2),
        ("i,2.5", si, lambda: 2.5),
        ("i,NA", si, lambda: NA),
        ("NA,f", lambda: NA, sf),
        ("i,np.int64", si, lambda: np_.int64(3)),
        ("np.float64,f", lambda: np_.float64(2.5), sf),
        ("f,np.float32", sf, lambda: np_.float32(0.1)),
    ]
    for opn, op in ARITH:
        for pn, a, b in pairs:
            add(f"arith {opn} {pn}", lambda op=op, a=a, b=b: op(a(), b()))
    add("arith add s,s", lambda: ss() + ss2())
    add("arith add s,str", lambda: ss() + "_x")
    add("arith add str,s", lambda: "x_" + ss())
    add("arith add s,NA", lambda: ss() + NA)
    for opn, op in ARITH[1:]:
        add(f"err arith {opn} s,s", lambda op=op: op(ss(), ss2()))
    add("err arith mul s,int", lambda: ss() * 2)
    for opn, op in ARITH:
        add(f"err arith {opn} b,b", lambda op=op: op(sb(), sb2()))
        add(f"err arith {opn} b,i", lambda op=op: op(sb(), si()))
    add("err arith add i,s", lambda: si() + ss())
    add("err arith add i,None", lambda: si() + None)
    add("err arith add i,str", lambda: si() + "x")
    add("err arith add i,2**64", lambda: si() + 2**64)
    add("err arith add i,list", lambda: si() + [1] * 15)
    add(
        "arith int edge",
        lambda: [
            _tl(F.Series([I64_MIN]) // -1),
            _tl(abs(F.Series([I64_MIN]))),
            _tl(-F.Series([I64_MIN])),
            _tl(F.Series([I64_MAX]) + 1),
            _tl(F.Series([I64_MAX]) * F.Series([I64_MAX])),
            _tl(F.Series([7, -7, 7, -7]) // F.Series([2, 2, -2, -2])),
            _tl(F.Series([7, -7, 7, -7]) % F.Series([2, 2, -2, -2])),
            _tl(F.Series([7.5, -7.5, 7.5, -7.5]) % F.Series([2.0, 2.0, -2.0, -2.0])),
            _tl(F.Series([2, 0, -1, 3]) ** F.Series([-1, -2, -3, 40])),
            _tl(F.Series([0, 1, -1]) ** F.Series([0, 0, 0])),
            _tl(F.Series([1, 2]) / F.Series([0, 0])),
            _tl(F.Series([0]) / F.Series([0])),
            _tl(F.Series([1.0, -1.0, 0.0]) / 0.0),
            _tl(F.Series([1.0, -1.0]) // 0.0),
            _tl(F.Series([5.0]) % 0.0),
            _tl(F.Series([1]) ** NA),
            _tl(NA ** F.Series([0])),
        ],
    )
    for k in ["i", "f"]:
        add(f"unary neg {k}", lambda k=k: -small[k]())
        add(f"unary pos {k}", lambda k=k: +small[k]())
        add(f"unary abs {k}", lambda k=k: abs(small[k]()))
    for k in ["b", "s"]:
        add(f"err unary neg {k}", lambda k=k: -small[k]())
        add(f"err unary pos {k}", lambda k=k: +small[k]())
        add(f"err unary abs {k}", lambda k=k: abs(small[k]()))
    add("unary invert b", lambda: ~sb())
    add("err unary invert i", lambda: ~si())
    add("err unary invert s", lambda: ~ss())
    cpairs = [
        ("i,i2", si, si2),
        ("i,f2", si, sf2),
        ("f,f2", sf, sf2),
        ("f,i2", sf, si2),
        ("s,s2", ss, ss2),
        ("b,b2", sb, sb2),
        ("i,3", si, lambda: 3),
        ("3,i", lambda: 3, si),
        ("f,nan", sf, lambda: nan),
        ("f,2.5", sf, lambda: 2.5),
        ("s,b", ss, lambda: "b"),
        ("b,True", sb, lambda: True),
        ("i,NA", si, lambda: NA),
        ("f,-0.0", sf, lambda: -0.0),
        (
            "exact",
            lambda: F.Series([2**53 + 1, 2**53, I64_MIN, I64_MAX, 3]),
            lambda: F.Series([2.0**53, 2.0**53, -(2.0**63), 2.0**63, 3.0]),
        ),
        ("exact scalar", lambda: F.Series([2**53 + 1, 2**53, I64_MAX]), lambda: 2.0**53),
        ("int vs float scalar 2**63", lambda: F.Series([I64_MAX, I64_MIN]), lambda: 9.223372036854775807e18),
    ]
    for opn, op in COMPARE:
        for pn, a, b in cpairs:
            add(f"cmp {opn} {pn}", lambda op=op, a=a, b=b: op(a(), b()))
    add("err cmp eq s,i", lambda: ss() == si())
    add("err cmp lt b,i", lambda: sb() < si())
    add("err cmp eq s,1", lambda: ss() == 1)
    add("err cmp eq i,str", lambda: si() == "a")
    add("err cmp lt b,1", lambda: sb() < 1)
    add("err cmp eq f,b", lambda: sf() == sb())
    for opn, op in LOGIC:
        add(f"logic {opn} kleene", lambda op=op: op(F.Series(K3), F.Series(K4)))
        add(f"logic {opn} True", lambda op=op: op(F.Series(K3), True))
        add(f"logic {opn} False", lambda op=op: op(False, F.Series(K4)))
        add(f"logic {opn} NA", lambda op=op: op(F.Series(K3), NA))
        add(f"logic {opn} NA left", lambda op=op: op(NA, F.Series(K3)))
        add(f"err logic {opn} i,i", lambda op=op: op(si(), si2()))
        add(f"err logic {opn} b,i", lambda op=op: op(sb(), si()))
        add(f"err logic {opn} b,1", lambda op=op: op(sb(), 1))
    add("logic invert kleene", lambda: ~F.Series(K3))
    add("names same kept", lambda: (F.Series([1], name="a") + F.Series([2], name="a")).name)
    add("names differ None", lambda: (F.Series([1], name="a") + F.Series([2], name="b")).name)
    add("names scalar kept", lambda: (F.Series([1], name="a") * 2).name)
    add(
        "align union order",
        lambda: (
            F.Series([1, 2, 3], index=["a", "b", "c"], name="x")
            + F.Series([10, 20, 30], index=["c", "d", "a"], name="x")
        ),
    )
    add("align int labels", lambda: F.Series([1.5, 2.5], index=[5, 3]) * F.Series([2.0, 3.0, 4.0], index=[3, 9, 5]))
    add("align range vs int", lambda: F.Series([1, 2, 3]) + F.Series([10, 20], index=[2, 0]))
    add("align same dup positional", lambda: F.Series([1, 2], index=["a", "a"]) + F.Series([3, 4], index=["a", "a"]))
    add(
        "align index names",
        lambda: (F.Series([1], index=F.Index(["a"], name="n")) + F.Series([1], index=F.Index(["b"], name="n"))).index,
    )
    add("align compare", lambda: F.Series([1, 2], index=["a", "b"]) < F.Series([2, 1], index=["b", "c"]))
    add("err align duplicates", lambda: F.Series([1, 2], index=["a", "a"]) + F.Series([3], index=["a"]))
    add("err align int vs string", lambda: F.Series([1], index=[0]) + F.Series([1], index=["a"]))
    add("align range different lengths", lambda: F.Series([1, 2]) + F.Series([1, 2, 3]))

    def dfa():
        return F.DataFrame({"a": [1, 2, 3], "b": [0.5, None, -1.5], "c": [5, 6, 7]}, index=["x", "y", "z"])

    def dfb():
        return F.DataFrame({"c": [1, 1, 1], "a": [10, 20, 30], "d": [1.0, 2.0, 3.0]}, index=["z", "w", "x"])

    add("df arith aligned", lambda: dfa() + dfb())
    add("df arith same", lambda: dfa() * dfa())
    add("df arith scalar", lambda: dfa() * 2)
    add("df arith scalar reflected", lambda: 10 - dfa())
    add("df arith floordiv 0", lambda: dfa() // 0)
    add("df cmp", lambda: dfa() > dfb())
    add("df cmp scalar", lambda: dfa() == 2)
    add("df neg", lambda: -dfa())
    add("df abs", lambda: abs(dfa()))
    add("df pos", lambda: +dfa())
    add("df invert", lambda: ~F.DataFrame({"p": K3, "q": K4}))
    add("df logic", lambda: F.DataFrame({"p": K3}) & F.DataFrame({"p": K4}))
    add("err df pos string", lambda: +sdf())
    add("err df neg string", lambda: -sdf())
    add("err df plus series", lambda: dfa() + F.Series([1, 2, 3]))
    add("err df and int", lambda: dfa() & dfa())
    add("err df add list", lambda: dfa() + [1, 2, 3])
    add("err df align int vs string", lambda: dfa() + F.DataFrame({"a": [1, 2, 3]}))
    add("err bool(series)", lambda: bool(F.Series([True])))
    add("err bool(df)", lambda: bool(sdf()))
    add("err bool(NA)", lambda: bool(NA))
    add("err hash series", lambda: hash(F.Series([1])))
    add("NA repr", lambda: repr(NA))
    add("NA singleton", lambda: [NA is F.NA, _tl(F.Series([None]))[0] is NA, F.Series([1]).sum() is not NA])

    big_ew = {
        "int add": lambda: D.BI + D.BI,
        "int sub scalar": lambda: D.BI - 7,
        "int mul": lambda: D.BI * D.BI_raw,
        "int floordiv": lambda: D.BI // (D.BI % 7),
        "int mod neg": lambda: D.BI % -13,
        "int truediv": lambda: D.BI / D.BI_raw,
        "int pow": lambda: D.BI ** (D.BI_raw % 5 - 1),
        "float add": lambda: D.BF + D.BF_raw,
        "float mul": lambda: D.BF * D.BF,
        "float truediv": lambda: D.BF / D.BF_raw,
        "float floordiv": lambda: D.BF // 0.3,
        "float mod": lambda: D.BF % -0.7,
        "float pow": lambda: D.BF**2.5,
        "huge float add": lambda: D.HF + D.HF,
        "huge float mul scalar": lambda: D.HF * 1.1,
        "huge int add wrap": lambda: D.HI + D.HI,
        "huge int mul wrap": lambda: D.HI * 3,
        "mixed add": lambda: D.BI + D.BF,
        "mixed compare": lambda: D.BI < D.BF,
        "int compare": lambda: D.BI >= D.BI_raw,
        "float eq": lambda: D.BF == D.BF,
        "float ne nan": lambda: D.BF != D.BF_raw,
        "string compare": lambda: D.BS < "w2500",
        "string eq": lambda: D.BS == D.BS,
        "string concat": lambda: D.BS + D.BS,
        "string concat scalar": lambda: D.BS + "!",
        "bool compare": lambda: D.BB > D.MASK,
        "kleene and": lambda: D.BB & (D.BI > 0),
        "kleene or": lambda: D.BB | (D.BI > 0),
        "xor": lambda: D.BB ^ D.MASK_NA,
        "neg": lambda: -D.BF,
        "neg int": lambda: -D.BI,
        "abs": lambda: abs(D.BF),
        "abs int": lambda: abs(D.HI),
        "invert": lambda: ~D.BB,
        "align big": lambda: D.ALIGN[0] + D.ALIGN[1],
        "align big string": lambda: D.ALIGN[2] - D.ALIGN[3],
        "df add": lambda: D.BDF[["v_f", "v_i"]] + D.BDF[["v_i", "v_f"]],
        "df mul scalar": lambda: D.BDF[["v_f", "v_i"]] * 2,
    }
    for nm, fn in big_ew.items():
        add(f"big ew {nm}", fn)

    # ---------------------------------------------------------------- astype (§12)
    for src, mk in [("i", si), ("f", sf), ("b", sb), ("s", ss)]:
        for dt in ["bool", "int64", "float64", "string"]:
            ok = not ((src == "f" and dt == "int64") or (src == "s" and dt != "string"))
            add(f"{'' if ok else 'err '}astype {src}->{dt}", lambda mk=mk, dt=dt: mk().astype(dt))
    add("astype f->int ok", lambda: F.Series([1.9, -1.9, -0.0, None, 2.0**62]).astype("int64"))
    add("err astype f->int inf", lambda: F.Series([inf]).astype("int64"))
    add("err astype f->int range", lambda: F.Series([2.0**63]).astype("int64"))
    add("astype i->f rounding", lambda: F.Series([2**53 + 1, 2**63 - 1, -(2**63)]).astype("float64"))
    add(
        "astype f->string specials",
        lambda: F.Series([1.0, 1e20, nan, -0.0, inf, 0.1, 1e-7, 123456789.125, None]).astype("string"),
    )
    add("astype s->int", lambda: F.Series(["+5", "-0", "007", "9223372036854775807", None]).astype("int64"))
    add(
        "astype s->float",
        lambda: F.Series(["nan", "inf", "-inf", "1e5", ".5", "5.", "+1.5E-3", "NaN", "-0"]).astype("float64"),
    )
    add("astype s->bool", lambda: F.Series(["True", "False", None]).astype("bool"))
    add("err astype s->int space", lambda: F.Series([" 1"]).astype("int64"))
    add("err astype s->int underscore", lambda: F.Series(["1_0"]).astype("int64"))
    add("err astype s->int range", lambda: F.Series(["9223372036854775808"]).astype("int64"))
    add("err astype s->int float text", lambda: F.Series(["1.0"]).astype("int64"))
    add("err astype s->float empty", lambda: F.Series([""]).astype("float64"))
    add("err astype s->bool lower", lambda: F.Series(["true"]).astype("bool"))
    add("err astype bad dtype", lambda: F.Series([1]).astype("object"))
    add("df astype one", lambda: F.DataFrame({"a": [1, 2], "b": [True, None]}).astype("float64"))
    add("df astype dict", lambda: sdf().astype({"i": "float64", "b": "string"}))
    add("err df astype dict unknown", lambda: sdf().astype({"zz": "float64"}))
    add("err df astype string->int", lambda: sdf()[["i", "s"]].astype("int64"))
    add("big astype f->string", lambda: D.BF.astype("string"))
    add("big astype string roundtrip", lambda: D.BF.astype("string").astype("float64"))
    add("big astype i->f", lambda: D.HI.astype("float64"))
    add("big astype i->string->i", lambda: D.BI.astype("string").astype("int64"))
    add("big astype f->bool", lambda: D.BF.astype("bool"))
    add("big astype b->int", lambda: D.BB.astype("int64"))
    add("err big astype f->int nan", lambda: D.BF.astype("int64"))

    # ---------------------------------------------------------------- NA handling (§2, §12)
    for k, mk in small.items():
        add(f"isna {k}", lambda mk=mk: mk().isna())
        add(f"notna {k}", lambda mk=mk: mk().notna())
        add(f"dropna {k}", lambda mk=mk: mk().dropna())
        add(f"dropna {k} ignore_index", lambda mk=mk: mk().dropna(ignore_index=True))
    add("fillna f", lambda: sf().fillna(-1.5))
    add("fillna f int value", lambda: sf().fillna(3))
    add("fillna i", lambda: si().fillna(0))
    add("fillna i float lossless", lambda: si().fillna(2.0))
    add("fillna b", lambda: sb().fillna(False))
    add("fillna s", lambda: ss().fillna("?"))
    add("fillna NA no-op", lambda: si().fillna(NA))
    add("fillna np scalar", lambda: si().fillna(np_.int32(4)))
    add("err fillna i lossy", lambda: si().fillna(1.5))
    add("err fillna s int", lambda: ss().fillna(1))
    add("err fillna b int", lambda: sb().fillna(1))
    add("err fillna list", lambda: si().fillna([1]))
    add("df isna", lambda: sdf().isna())
    add("df notna", lambda: sdf_l().notna())
    add("df fillna dict", lambda: sdf().fillna({"i": -1, "s": "?"}))
    add("df fillna scalar", lambda: F.DataFrame({"a": [1, None], "b": [None, 2]}).fillna(0))
    add("err df fillna scalar mixed", lambda: sdf().fillna(0))
    add("err df fillna dict unknown", lambda: sdf().fillna({"zz": 1}))
    add("df dropna", lambda: sdf().dropna())
    add("df dropna all", lambda: F.DataFrame({"a": [1, None, None], "b": [None, None, "x"]}).dropna(how="all"))
    add("df dropna subset", lambda: sdf_l().dropna(subset=["f", "b"]))
    add("df dropna subset str", lambda: sdf().dropna(subset="s", ignore_index=True))
    add("df dropna subset empty", lambda: sdf().dropna(subset=[]))
    add("err df dropna how", lambda: sdf().dropna(how="some"))
    add("err df dropna subset unknown", lambda: sdf().dropna(subset=["zz"]))
    add("big isna", lambda: D.BF.isna())
    add("big notna", lambda: D.BS.notna())
    add("big fillna int", lambda: D.BI.fillna(0))
    add("big fillna string", lambda: D.BS.fillna("x"))
    add("big fillna float", lambda: D.BF.fillna(-0.0))
    add("big dropna", lambda: D.BF.dropna())
    add("big dropna ignore_index", lambda: D.BS.dropna(ignore_index=True))
    add("big df dropna", lambda: D.BDF.dropna())
    add("big df dropna all subset", lambda: D.BDF.dropna(how="all", subset=["v_f", "k_s"]))
    add("big df fillna dict", lambda: D.BDF.fillna({"v_i": 0, "v_s": "", "k_f": nan}))

    # ---------------------------------------------------------------- unique values and duplicates (§13)
    uniq_small = {
        "f": lambda: F.Series(
            [-0.0, 0.0, nan, float.fromhex("-0x1.8p+1023") * 0 + nan, None, 1.5, None, nan, 1.5], name="u"
        ),
        "nanpayload": lambda: F.Series(
            np_.frombuffer(
                np_.array([0x7FF8000000000001, 0x7FF8000000000002, 0xFFF8000000000000], dtype=np_.uint64).tobytes(),
                dtype=np_.float64,
            ).copy()
        ),
        "i": si,
        "b": sb,
        "s": ss,
        "empty": lambda: F.Series([], dtype="int64"),
        "allNA": lambda: F.Series([None, None], dtype="string"),
    }
    for k, mk in uniq_small.items():
        add(f"unique {k}", lambda mk=mk: _uv(mk()))
        add(f"nunique {k}", lambda mk=mk: mk().nunique())
        add(f"nunique {k} dropna=False", lambda mk=mk: mk().nunique(dropna=False))
        for keep in ["first", "last", "none"]:
            add(f"duplicated {k} {keep}", lambda mk=mk, keep=keep: mk().duplicated(keep=keep))
            add(f"drop_duplicates {k} {keep}", lambda mk=mk, keep=keep: mk().drop_duplicates(keep=keep))
        add(f"drop_duplicates {k} ignore_index", lambda mk=mk: mk().drop_duplicates(ignore_index=True))
        add(f"duplicated {k} default", lambda mk=mk: mk().duplicated())
        add(f"value_counts {k}", lambda mk=mk: mk().value_counts())
        add(f"value_counts {k} dropna=False", lambda mk=mk: mk().value_counts(dropna=False))
        add(f"value_counts {k} normalize", lambda mk=mk: mk().value_counts(normalize=True, dropna=False))
        add(f"value_counts {k} ascending", lambda mk=mk: mk().value_counts(**_kw(dict(ascending=True, dropna=False))))
        add(f"value_counts {k} sort=False", lambda mk=mk: mk().value_counts(sort=False, dropna=False))
    add("value_counts unnamed", lambda: F.Series([1, 1, 2]).value_counts())
    add("err value_counts name count", lambda: F.Series([1], name="count").value_counts())
    add("err value_counts name proportion", lambda: F.Series([1], name="proportion").value_counts(normalize=True))
    add("value_counts name proportion ok", lambda: F.Series([1], name="proportion").value_counts())
    add("err duplicated keep False", lambda: F.Series([1]).duplicated(keep=False))
    add("err duplicated keep bad", lambda: F.Series([1]).duplicated(keep="middle"))
    add("err nunique positional", lambda: F.Series([1]).nunique(False))
    add("df nunique", lambda: sdf().nunique())
    add("df nunique dropna=False", lambda: sdf().nunique(dropna=False))
    add(
        "df duplicated",
        lambda: F.DataFrame({"a": [1, 1, 2, 1, None, None], "b": ["x", "x", "x", "y", None, None]}).duplicated(),
    )
    for keep in ["first", "last", "none"]:
        add(f"df duplicated subset {keep}", lambda keep=keep: sdf().duplicated(["b", "s"], keep=keep))
        add(f"df drop_duplicates subset {keep}", lambda keep=keep: sdf_l().drop_duplicates("b", keep=keep))
    add("df drop_duplicates ignore_index", lambda: sdf_l().drop_duplicates(["b"], ignore_index=True))
    add(
        "df value_counts", lambda: F.DataFrame({"a": [1, 1, 2, 1, None], "b": ["x", "x", "x", "y", "y"]}).value_counts()
    )
    add("df value_counts subset", lambda: sdf().value_counts(["b", "s"], dropna=False))
    add("df value_counts subset str normalize", lambda: sdf().value_counts("b", normalize=True))
    add(
        "df value_counts ascending sort=False",
        lambda: sdf().value_counts(["b"], **_kw(dict(ascending=True, sort=False, dropna=False))),
    )
    add("err df value_counts count column", lambda: F.DataFrame({"count": [1]}).value_counts())
    add("err df duplicated unknown", lambda: sdf().duplicated(["zz"]))
    add("err df duplicated empty subset", lambda: sdf().duplicated([]))
    for nm, get in [
        ("f", lambda: D.BF),
        ("i", lambda: D.BI),
        ("s", lambda: D.BS),
        ("b", lambda: D.BB),
        ("huge f", lambda: D.HF),
        ("huge i", lambda: D.HI),
    ]:
        add(f"big unique {nm}", lambda get=get: _uv(get()))
        add(f"big nunique {nm}", lambda get=get: (get().nunique(), get().nunique(dropna=False)))
        for keep in ["first", "last", "none"]:
            add(f"big duplicated {nm} {keep}", lambda get=get, keep=keep: get().duplicated(keep=keep))
        add(f"big drop_duplicates {nm}", lambda get=get: get().drop_duplicates(keep="last"))
        add(f"big value_counts {nm}", lambda get=get: get().value_counts())
        add(
            f"big value_counts {nm} dropna=False sort=False",
            lambda get=get: get().value_counts(dropna=False, sort=False),
        )
        add(
            f"big value_counts {nm} normalize ascending",
            lambda get=get: get().value_counts(**_kw(dict(normalize=True, ascending=True))),
        )
    add("big df nunique", lambda: D.BDF.nunique())
    for keep in ["first", "last", "none"]:
        add(f"big df duplicated {keep}", lambda keep=keep: D.BDF.duplicated(["k_i", "k_s"], keep=keep))
    add("big df duplicated all", lambda: D.BDF.duplicated())
    add("big df drop_duplicates", lambda: D.BDF.drop_duplicates(["k_s", "k_f", "k_b"], ignore_index=True))
    add("big df value_counts", lambda: D.BDF.value_counts(["k_s", "k_b"]))
    add("big df value_counts dropna=False", lambda: D.BDF.value_counts(["k_f", "k_b"], dropna=False, sort=False))

    # ---------------------------------------------------------------- selection (§7, §14)
    def sl():
        return F.Series(list(SI), index=list(LBL), name="i")

    def sdup():
        return F.Series([10, 20, 30, 40], index=["a", "b", "a", "c"])

    def sint():
        return F.Series([1.5, 2.5, 3.5, 4.5], index=[10, 30, 20, 40])

    sel = [
        ("loc label", lambda: sl().loc["r4"]),
        ("[] label", lambda: sl()["r0"]),
        ("loc NA value", lambda: sl().loc["r1"]),
        ("loc list", lambda: sl().loc[["r3", "r0", "r3"]]),
        ("loc Index", lambda: sl().loc[F.Index(["r2", "r1"])]),
        ("loc numpy labels", lambda: sl().loc[np_.array(["r2", "r1"])]),
        ("loc slice", lambda: sl().loc["r2":"r6"]),
        ("loc slice step", lambda: sl().loc["r1":"r9":3]),
        ("loc slice reversed", lambda: sl().loc["r6":"r2":-1]),
        ("loc slice open", lambda: sl().loc[:"r3"]),
        ("loc slice open end", lambda: sl().loc["r12":]),
        ("loc slice empty", lambda: sl().loc["r6":"r2"]),
        ("loc mask", lambda: sl().loc[sl() > 2]),
        ("[] mask", lambda: sl()[sl().notna()]),
        ("loc dup list", lambda: sdup().loc[["a", "c"]]),
        ("loc int label", lambda: sint().loc[20]),
        ("loc np int label", lambda: sint().loc[np_.int64(40)]),
        ("loc int slice", lambda: sint().loc[30:40]),
        ("loc range label", lambda: si().loc[3]),
        ("loc range slice", lambda: si().loc[2:9]),
        ("loc range slice step", lambda: si().loc[1:13:4]),
        ("loc range list", lambda: si().loc[[4, 0]]),
        ("iloc int", lambda: sl().iloc[2]),
        ("iloc neg", lambda: sl().iloc[-1]),
        ("iloc NA", lambda: sl().iloc[1]),
        ("iloc slice", lambda: sl().iloc[2:8]),
        ("iloc slice step", lambda: sl().iloc[1:13:3]),
        ("iloc slice neg step", lambda: sl().iloc[::-2]),
        ("iloc slice out", lambda: sl().iloc[10:100]),
        ("iloc list", lambda: sl().iloc[[0, -1, 3, 3]]),
        ("iloc np int", lambda: sl().iloc[np_.int64(4)]),
        ("iloc numpy list", lambda: sl().iloc[np_.array([1, 2])]),
        ("iloc range keeps RangeIndex", lambda: si().iloc[3:9:2]),
        ("iloc range list Index", lambda: si().iloc[[3, 4]]),
        ("mask range Index", lambda: si()[si() > 0]),
        ("mask NA dropped", lambda: si()[F.Series(SB)]),
        ("mask reordered labels", lambda: sl()[F.Series([x > 5 for x in range(15)], index=list(reversed(LBL)))]),
        ("loc float series label", lambda: sf().loc[3]),
        ("loc string series", lambda: ss().loc[[0, 1, 2]]),
    ]
    for nm, fn in sel:
        add(f"sel {nm}", fn)
    errs = [
        ("loc missing", KeyError, lambda: sl().loc["zz"]),
        ("loc list missing", KeyError, lambda: sl().loc[["r0", "zz", "yy"]]),
        ("loc dup label", KeyError, lambda: sdup().loc["a"]),
        ("loc slice dup end", KeyError, lambda: sdup().loc["a":"c"]),
        ("loc slice missing end", KeyError, lambda: sl().loc["r1":"zz"]),
        ("loc wrong type", TypeError, lambda: sl().loc[0]),
        ("loc bool label", TypeError, lambda: sint().loc[True]),
        ("loc float label", TypeError, lambda: sint().loc[20.0]),
        ("loc tuple", TypeError, lambda: sl().loc[("r0", "r1")]),
        ("loc slice step 0", ValueError, lambda: sl().loc["r0":"r3":0]),
        ("loc range missing", KeyError, lambda: si().loc[15]),
        ("[] missing", KeyError, lambda: sl()["zz"]),
        ("iloc out", IndexError, lambda: sl().iloc[15]),
        ("iloc neg out", IndexError, lambda: sl().iloc[-16]),
        ("iloc list out", IndexError, lambda: sl().iloc[[0, 20]]),
        ("iloc str", TypeError, lambda: sl().iloc["r0"]),
        ("iloc float", TypeError, lambda: sl().iloc[1.0]),
        ("iloc list float", TypeError, lambda: sl().iloc[[1.0]]),
        ("mask not bool", TypeError, lambda: si()[si()]),
        ("mask labels differ", ValueError, lambda: sl()[F.Series([True] * 15)]),
        ("mask shorter", ValueError, lambda: si()[F.Series([True, False])]),
        ("mask dup labels", ValueError, lambda: sdup()[F.Series([True] * 4, index=["a", "b", "c", "d"])]),
    ]
    for nm, _, fn in errs:
        add(f"err sel {nm}", fn)

    add("df sel column", lambda: sdf_l()["s"])
    add("df sel columns", lambda: sdf_l()[["s", "f"]])
    add("df sel mask", lambda: sdf_l()[sdf_l()["i"] > 0])
    add("df sel mask NA", lambda: sdf()[F.Series(SB)])
    add("df loc row", lambda: sdf_l().loc["r2"])
    add("df loc row col", lambda: sdf_l().loc["r2", "s"])
    add("df loc row col NA", lambda: sdf_l().loc["r1", "i"])
    add("df loc rows col", lambda: sdf_l().loc[["r3", "r1"], "i"])
    add("df loc col slice", lambda: sdf_l().loc[:, "i":"s"])
    add("df loc col slice open", lambda: sdf_l().loc["r1":"r4", :"i"])
    add("df loc row cols", lambda: sdf_l().loc["r2", ["s", "f"]])
    add("df loc mask cols", lambda: sdf_l().loc[sdf_l()["b"], ["i"]])
    add("df loc slice", lambda: sdf_l().loc["r10":])
    add("df iloc row", lambda: sdf().iloc[0])
    add("df iloc row col", lambda: sdf().iloc[2, 0])
    add("df iloc neg", lambda: sdf().iloc[-1, -1])
    add("df iloc slice cols list", lambda: sdf_l().iloc[1:, [3, 0]])
    add("df iloc list col slice", lambda: sdf_l().iloc[[4, 2], 1:3])
    add("df iloc step", lambda: sdf().iloc[::4])
    add("df iloc range list", lambda: sdf().iloc[[1, 2]])
    add("df [] after iloc keeps range", lambda: sdf().iloc[2:5].index)
    add("df reindex", lambda: sdf_l().reindex(["r3", "zz", "r0"]))
    add("df reindex RangeIndex", lambda: sdf().reindex(F.RangeIndex(10, 20)))
    add("reindex series", lambda: sl().reindex(["r14", "nope", "r0"]))
    add("reindex series range", lambda: si().reindex(range(-2, 4)))
    add("reindex series Index", lambda: sl().reindex(F.Index(["r1"], name="nn")))
    add("err reindex dup", lambda: sdup().reindex(["a"]))
    add("err reindex wrong type", lambda: sl().reindex([1, 2]))
    add("err reindex float labels", lambda: sl().reindex([1.5]))
    add("err df [] missing", lambda: sdf()["zz"])
    add("err df [] int", lambda: sdf()[1])
    add("err df [] slice", lambda: sdf()[1:3])
    add("err df [] list missing", lambda: sdf()[["f", "zz"]])
    add("err df [] list dup", lambda: sdf()[["f", "f"]])
    add("err df loc col missing", lambda: sdf_l().loc["r2", "zz"])
    add("err df loc row missing", lambda: sdf_l().loc["zz"])
    add("err df loc col slice step", lambda: sdf_l().loc[:, "f":"s":2])
    add("err df iloc col out", lambda: sdf().iloc[0, 9])
    add("err df iloc row out", lambda: sdf().iloc[99])
    add("err df mask labels", lambda: sdf_l()[F.Series([True] * 15)])
    add("big sel mask", lambda: D.BDF[D.MASK])
    add("big sel mask NA", lambda: D.BDF[D.MASK_NA])
    add("big sel series mask", lambda: D.BS[D.MASK])
    add("big sel iloc take", lambda: D.BDF.iloc[D.PERM])
    add("big sel iloc slice step", lambda: D.BDF.iloc[5::3])
    add("big sel iloc series slice", lambda: D.BF.iloc[1000:250_000])
    add("big sel loc list", lambda: F.Series(_tl(D.BF), index=D.base["words"]).loc[["w1", "é7", ""]])
    add("big sel loc labels", lambda: D.ALIGN[0].loc[D.PERM[:200_000]])
    add("big sel loc string labels", lambda: D.ALIGN[2].loc[[f"k{j}" for j in D.PERM[:150_000]]])
    add("big sel loc slice", lambda: D.ALIGN[0].loc[D.PERM[0] : D.PERM[1]])
    add(
        "big sel mask aligned",
        lambda: D.ALIGN[0][F.Series((D.ALIGN[0] > 0).to_numpy(), index=_tl(D.ALIGN[0].index)[::-1])],
    )
    add("big reindex", lambda: D.ALIGN[0].reindex(list(range(-5, N, 2))))
    add("big df reindex", lambda: D.BDF.reindex(D.PERM))
    add("err big sel mask labels", lambda: D.BDF[F.Series(D.base["mask"], index=list(range(1, N + 1)))])

    # ---------------------------------------------------------------- assignment and Copy-on-Write (§14)
    def assign(target, fn):
        def run():
            obj = target()
            r = fn(obj)
            return obj if r is None else (obj, r)

        return run

    add("set loc label", assign(sl, lambda s: s.loc.__setitem__("r1", 20)))
    add("set [] label", assign(sl, lambda s: s.__setitem__("r0", -5)))
    add("set iloc neg NA", assign(sl, lambda s: s.iloc.__setitem__(-1, NA)))
    add("set loc list", assign(sl, lambda s: s.loc.__setitem__(["r0", "r2"], [100, 300])))
    add("set loc slice scalar lossless", assign(sl, lambda s: s.loc.__setitem__(slice("r2", "r5"), 7.0)))
    add(
        "set loc series aligned",
        assign(sl, lambda s: s.loc.__setitem__(["r4", "r0"], F.Series([1, 2], index=["r0", "r4"]))),
    )
    add("set loc mask", assign(sl, lambda s: s.loc.__setitem__(s > 2, 0)))
    add("set [] mask NA", assign(sl, lambda s: s.__setitem__(F.Series(SB, index=list(LBL)), NA)))
    add("set iloc slice list", assign(sl, lambda s: s.iloc.__setitem__(slice(0, 3), [9, None, 7])))
    add("set iloc step", assign(sl, lambda s: s.iloc.__setitem__(slice(None, None, 4), -1)))
    add("set iloc list", assign(sl, lambda s: s.iloc.__setitem__([0, -2], 5)))
    add("set iloc numpy scalar", assign(sl, lambda s: s.iloc.__setitem__(0, np_.int8(-3))))
    add("set string", assign(ss, lambda s: (s.iloc.__setitem__(1, "yy"), s.iloc.__setitem__([0, 2], NA))))
    add("set float", assign(sf, lambda s: s.iloc.__setitem__(slice(0, 4), [-0.0, nan, None, 5])))
    add("set float series cast", assign(sf, lambda s: s.iloc.__setitem__(slice(0, 2), F.Series([1, 2]))))
    add("set bool", assign(sb, lambda s: s.loc.__setitem__([1, 6], [False, True])))
    add("err set lossy", assign(sl, lambda s: s.loc.__setitem__("r0", 1.5)))
    add("err set string into int", assign(sl, lambda s: s.loc.__setitem__("r0", "x")))
    add("err set bool into int", assign(sl, lambda s: s.loc.__setitem__("r0", True)))
    add("err set int into bool", assign(sb, lambda s: s.iloc.__setitem__(0, 1)))
    add("err set new label", assign(sl, lambda s: s.loc.__setitem__("new", 1)))
    add("err set length", assign(sl, lambda s: s.iloc.__setitem__(slice(0, 2), [1, 2, 3])))
    add("err set list to one row", assign(sl, lambda s: s.iloc.__setitem__(0, [1])))
    add("err set series to one row", assign(sl, lambda s: s.iloc.__setitem__(0, F.Series([1]))))
    add("err set iloc out", assign(sl, lambda s: s.iloc.__setitem__(15, 1)))
    add(
        "set series missing label gives NA",
        assign(sl, lambda s: s.loc.__setitem__(["r0"], F.Series([1], index=["r1"]))),
    )

    def cow_col():
        df = sdf_l()
        col = df["i"]
        col.iloc[0] = 99
        return df, col

    def cow_head():
        s = sl()
        h = s.head(3)
        h.iloc[0] = -1
        s.iloc[1] = 50
        return s, h

    def cow_copy():
        s = sl()
        other = F.Series(s)
        other.iloc[0] = 0
        s2 = F.Series(s, name="renamed")
        s.loc["r2"] = 1
        return s, other, s2

    def cow_view():
        s = F.Series([1.5, 2.5, 3.5])
        v = s.to_numpy()
        s.iloc[0] = 7.0
        return s, v.copy(), v.flags.writeable

    def cow_df_copy():
        df = sdf()
        df2 = F.DataFrame(df)
        df2.loc[0, "i"] = -1
        df2["new"] = 1
        sub = df[["i"]]
        df.iloc[1, 1] = 4
        return df, df2, sub

    def cow_slice():
        s = sl()
        part = s.iloc[2:8]
        s.iloc[3] = 1000
        part.iloc[0] = -1000
        lp = s.loc["r3":"r6"]
        return s, part, lp

    def cow_groupby():
        df = F.DataFrame({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
        g = df.groupby("k")
        df["k"] = [5, 6, 7]
        df.loc[0, "v"] = 100.0
        return g.sum(), df

    def cow_mask_result():
        s = sl()
        m = s[s.notna()]
        m.iloc[0] = 1
        return s, m

    for nm, fn in [
        ("column", cow_col),
        ("head", cow_head),
        ("copy", cow_copy),
        ("to_numpy view", cow_view),
        ("df copy", cow_df_copy),
        ("slice", cow_slice),
        ("groupby", cow_groupby),
        ("mask", cow_mask_result),
    ]:
        add(f"cow {nm}", fn)

    def dset(fn, mk=sdf_l):
        return assign(mk, fn)

    add("df set new scalar", dset(lambda d: d.__setitem__("new", 1.5)))
    add("df set new string scalar", dset(lambda d: d.__setitem__("new", "k")))
    add("df set new NA scalar", dset(lambda d: d.__setitem__("new", None)))
    add("df set replace list", dset(lambda d: d.__setitem__("i", list(range(15)))))
    add("df set replace type change", dset(lambda d: d.__setitem__("i", ["x"] * 15)))
    add("df set series aligned", dset(lambda d: d.__setitem__("n", F.Series([1, 2], index=["r3", "r0"]))))
    add("df set numpy", dset(lambda d: d.__setitem__("n", np_.arange(15) * 0.5)))
    add("df set first column of empty", assign(lambda: F.DataFrame(), lambda d: d.__setitem__("a", [1, 2, 3])))
    add("df set loc", dset(lambda d: d.loc.__setitem__(("r2", "i"), 2)))
    add("df set loc lossless", dset(lambda d: d.loc.__setitem__(("r2", "i"), 3.0)))
    add("df set iloc NA", dset(lambda d: d.iloc.__setitem__((0, 1), NA)))
    add("df set loc mask", dset(lambda d: d.loc.__setitem__((d["b"].notna(), "s"), "z")))
    add("df set loc slice list", dset(lambda d: d.loc.__setitem__((slice("r1", "r3"), "f"), [1, 2, 3])))
    add(
        "df set iloc list series",
        dset(lambda d: d.iloc.__setitem__(([0, 1], 0), F.Series([5.5, 6.5], index=["r1", "r0"]))),
    )
    add("err df set loc new column", dset(lambda d: d.loc.__setitem__(("r2", "new"), 1)))
    add("err df set loc new row", dset(lambda d: d.loc.__setitem__(("zz", "i"), 1)))
    add("err df set loc two columns", dset(lambda d: d.loc.__setitem__(("r2", ["f", "i"]), 1)))
    add("err df set lossy", dset(lambda d: d.loc.__setitem__(("r0", "i"), 0.5)))
    add("err df set length", dset(lambda d: d.__setitem__("n", [1, 2])))
    add("err df set name int", dset(lambda d: d.__setitem__(3, [1] * 15)))
    add("err df set DataFrame", dset(lambda d: d.__setitem__("n", F.DataFrame({"a": [1]}))))
    add("err df set series dup labels", dset(lambda d: d.__setitem__("n", F.Series([1, 2], index=["r0", "r0"]))))

    def big_set():
        s = F.Series(D.BF)
        s[D.MASK] = NA
        s.iloc[D.PERM[:1000]] = -0.0
        s.iloc[100_000:200_000:7] = 1.25
        return s, D.BF.isna().sum()

    def big_df_set():
        df = F.DataFrame(D.BDF)
        df.loc[D.MASK_NA, "v_i"] = 0
        df["w"] = D.BF * 2
        df.iloc[D.PERM[:5000], 6] = None
        return df, D.BDF.shape

    def big_set_series():
        s = F.Series(D.ALIGN[0])
        s.loc[D.PERM[:100_000]] = F.Series(list(range(100_000)), index=D.PERM[:100_000][::-1], dtype="float64")
        return s

    add("big set", big_set)
    add("big df set", big_df_set)
    add("big set series aligned", big_set_series)

    # ---------------------------------------------------------------- reductions (§2, §15)
    red_small = {
        "f": sf,
        "f2": sf2,
        "i": si,
        "i2": si2,
        "b": sb,
        "s": ss,
        "f noNaN": lambda: F.Series([1.5, None, -0.0, 0.0, 2.25, -7.0]),
        "negzero": lambda: F.Series([-0.0, 0.0, None]),
        "zeroneg": lambda: F.Series([0.0, -0.0]),
        "i exact mean": lambda: F.Series([I64_MAX, I64_MAX, I64_MAX, 1]),
        "i wrap sum": lambda: F.Series([I64_MAX, 1, None]),
        "allNA f": lambda: F.Series([None, None]),
        "allNA i": lambda: F.Series([None], dtype="int64"),
        "allNA s": lambda: F.Series([None], dtype="string"),
        "allNA b": lambda: F.Series([None], dtype="bool"),
        "empty f": lambda: F.Series([]),
        "empty i": lambda: F.Series([], dtype="int64"),
        "one": lambda: F.Series([4.0, None]),
    }
    for k, mk in red_small.items():
        isstr = k in ("s", "allNA s")
        isbool = k in ("b", "allNA b")
        for op in REDUCTIONS:
            bad = (isstr and op != "min" and op != "max") or (isbool and op in ("std", "var"))
            p = "err " if bad else ""
            add(f"{p}reduce {op} {k}", lambda mk=mk, op=op: _red(mk(), op))
            add(f"{p}reduce {op} {k} skipna=False", lambda mk=mk, op=op: _red(mk(), op, skipna=False))
            if op in ("std", "var"):
                for dd in (0, 2, 5):
                    add(f"{p}reduce {op} {k} ddof={dd}", lambda mk=mk, op=op, dd=dd: _red(mk(), op, ddof=dd))
                    add(
                        f"{p}reduce {op} {k} ddof={dd} skipna=False",
                        lambda mk=mk, op=op, dd=dd: _red(mk(), op, ddof=dd, skipna=False),
                    )
        add(f"reduce count {k}", lambda mk=mk: mk().count())
    add("err reduce std ddof negative", lambda: _red(sf(), "std", ddof=-1))
    add("err reduce var ddof float", lambda: _red(sf(), "var", ddof=1.5))
    add("err reduce sum positional", lambda: sf().sum(False))
    add("err reduce sum axis", lambda: sf().sum(axis=0))
    add(
        "reduce result types",
        lambda: [type(v).__name__ for v in (si().sum(), si().mean(), sb().sum(), sb().min(), ss().max())],
    )
    for op in REDUCTIONS:
        add(f"df reduce {op}", lambda op=op: _red(F.DataFrame({"f": SF, "i": SI, "b2": SF2}), op))
        add(f"df reduce {op} skipna=False", lambda op=op: _red(F.DataFrame({"i": SI, "f": SF2}), op, skipna=False))
    add("df reduce std ddof", lambda: _red(F.DataFrame({"i": SI2, "f": SF2}), "std", ddof=0))
    add("df reduce min mixed", lambda: sdf().min())
    add("df reduce max mixed", lambda: sdf_l().max(skipna=False))
    add("df reduce count", lambda: sdf().count())
    add("df reduce sum bool", lambda: F.DataFrame({"b": SB}).sum())
    add("df reduce empty", lambda: F.DataFrame().sum())
    add("err df reduce mean string", lambda: sdf().mean())
    add("err df reduce std bool", lambda: F.DataFrame({"f": SF, "b": SB}).std())
    big_red = {
        "huge f": lambda: D.HF,
        "huge i": lambda: D.HI,
        "huge fNA": lambda: D.HFN,
        "f": lambda: D.BF,
        "f raw": lambda: D.BF_raw,
        "i": lambda: D.BI,
        "b": lambda: D.BB,
        "s": lambda: D.BS,
        "noNaN v": lambda: D.BDF_noNaN["v"],
        "mask": lambda: D.MASK,
    }
    for k, get in big_red.items():
        for op in REDUCTIONS:
            if k == "s" and op not in ("min", "max"):
                continue
            if k in ("b", "mask") and op in ("std", "var"):
                continue
            add(f"big reduce {op} {k}", lambda get=get, op=op: _red(get(), op))
            add(f"big reduce {op} {k} skipna=False", lambda get=get, op=op: _red(get(), op, skipna=False))
            if op in ("std", "var"):
                add(f"big reduce {op} {k} ddof=0", lambda get=get, op=op: _red(get(), op, ddof=0))
        add(f"big reduce count {k}", lambda get=get: get().count())
    add("err big reduce mean s", lambda: D.BS.mean())
    for op in REDUCTIONS:
        add(f"big df reduce {op}", lambda op=op: _red(D.BDF_noNaN[["v", "w"]], op))
    add("big df reduce min all", lambda: D.BDF.min())
    add("big df reduce count", lambda: D.BDF.count())
    add("big reduce sum of subset matches", lambda: D.BDF_noNaN[D.BDF_noNaN["k"] == 17]["v"].sum())

    # ---------------------------------------------------------------- groupby (§4, §16)
    def gdf():
        return F.DataFrame(
            {
                "k": [1, 2, 1, None, 2, 3, 1, None, 3, 2, 1, 2],
                "kf": [0.0, -0.0, nan, 1.5, nan, None, 0.0, 2.5, -0.0, 1.5, None, inf],
                "ks": ["b", "a", "b", None, "a", "c", "b", None, "c", "a", "b", "é"],
                "kb": [True, False, True, None, False, True, True, None, False, False, True, False],
                "v": [1.5, None, 2.5, 4.0, nan, 3.0, None, 7.0, -0.0, 0.25, 1e300, 1e300],
                "n": [1, 2, None, 4, 5, 6, I64_MAX, 8, 9, None, 11, 1],
                "s": ["x", "y", None, "z", "w", "v", "u", "t", None, "r", "q", "p"],
                "t": [True, None, False, True, False, None, True, True, False, True, True, None],
            }
        )

    gfuncs = [
        ("sum", {}),
        ("mean", {}),
        ("min", {}),
        ("max", {}),
        ("std", {}),
        ("var", {}),
        ("count", None),
        ("size", None),
        ("first", {}),
        ("last", {}),
        ("first", {"skipna": False}),
        ("last", {"skipna": False}),
        ("sum", {"skipna": False}),
        ("max", {"skipna": False}),
        ("std", {"ddof": 0}),
        ("var", {"ddof": 3}),
        ("mean", {"skipna": False}),
    ]

    def gcall(g, f, kw):
        return getattr(g, f)() if kw is None else getattr(g, f)(**_kw(kw))

    def gname(f, kw):
        return f + ("" if not kw else " " + ",".join(f"{a}={b}" for a, b in kw.items()))

    numeric = ["v", "n", "t"]
    for by in ["k", "kf", "ks", "kb", ["k", "ks"], ["kb", "kf", "ks"]]:
        bn = by if isinstance(by, str) else "+".join(by)
        for sort, dropna in [(True, True), (False, True), (True, False), (False, False)]:
            for f, kw in gfuncs:
                vals = None if f in ("min", "max", "count", "size", "first", "last") else numeric
                if vals is not None:
                    vals = [c for c in vals if not (f in ("std", "var") and c == "t")]
                    fn = lambda by=by, sort=sort, dropna=dropna, f=f, kw=kw, vals=vals: gcall(
                        gdf().groupby(by, sort=sort, dropna=dropna)[vals], f, kw
                    )
                else:
                    fn = lambda by=by, sort=sort, dropna=dropna, f=f, kw=kw: gcall(
                        gdf().groupby(by, sort=sort, dropna=dropna), f, kw
                    )
                add(f"groupby {bn} sort={sort} dropna={dropna} {gname(f, kw)}", fn)
    add("groupby agg str", lambda: gdf().groupby("k")[["v", "n"]].agg("sum"))
    add("groupby agg str all", lambda: gdf().groupby("ks").agg("first"))
    add("groupby agg dict", lambda: gdf().groupby("k").agg({"v": "mean", "n": "max", "s": "last"}))
    add(
        "groupby agg named",
        lambda: (
            gdf()
            .groupby(["ks"], sort=False, dropna=False)
            .agg(
                total=("v", "sum"),
                n=("n", "count"),
                m=("v", "mean"),
                lo=("s", "min"),
                sd=("n", "std"),
                vr=("v", "var"),
                f=("t", "first"),
                l=("t", "last"),
                hi=("n", "max"),
            )
        ),
    )
    add("groupby single column", lambda: gdf().groupby("k")["v"].sum())
    add("groupby select list", lambda: gdf().groupby("k")[["n", "v"]].max())
    add("groupby keys only", lambda: F.DataFrame({"a": [2, 1, 2], "b": ["x", "y", "x"]}).groupby(["a", "b"]).size())
    add("groupby keys only sum", lambda: F.DataFrame({"a": [2, 1, 2]}).groupby("a").sum())
    add("groupby empty frame", lambda: F.DataFrame({"a": [], "b": []}, dtype="int64").groupby("a").sum())
    add("groupby all NA keys", lambda: F.DataFrame({"a": [None, None], "b": [1, 2]}).groupby("a").sum())
    add("groupby string agg min", lambda: gdf().groupby("k")["s"].min())
    add("groupby bool agg", lambda: gdf().groupby("ks")["t"].agg("mean"))
    add("groupby repeated use", lambda: (lambda g: (g.sum(), g.count(), g["v"].mean()))(gdf().groupby("k")[["v", "n"]]))
    add("err groupby string mean", lambda: gdf().groupby("k").mean())
    add("err groupby string sum", lambda: gdf().groupby("k")["s"].sum())
    add("err groupby bool std", lambda: gdf().groupby("k")["t"].std())
    add("err groupby agg list", lambda: gdf().groupby("k").agg(["sum"]))
    add("err groupby agg dict list", lambda: gdf().groupby("k").agg({"v": ["sum"]}))
    add("err groupby agg unknown", lambda: gdf().groupby("k").agg("median"))
    add("err groupby agg nothing", lambda: gdf().groupby("k").agg())
    add("err groupby agg both", lambda: gdf().groupby("k").agg("sum", x=("v", "sum")))
    add("err groupby agg named bad spec", lambda: gdf().groupby("k").agg(x="v"))
    add("err groupby agg named unknown column", lambda: gdf().groupby("k").agg(x=("zz", "sum")))
    add("err groupby agg named key column", lambda: gdf().groupby("k").agg(x=("k", "sum")))
    add("err groupby agg result is key", lambda: gdf().groupby("k").agg(k=("v", "sum")))
    add("err groupby select key", lambda: gdf().groupby("k")["k"])
    add("err groupby select unknown", lambda: gdf().groupby("k")["zz"])
    add("err groupby select dup", lambda: gdf().groupby("k")[["v", "v"]])
    add("err groupby unknown key", lambda: gdf().groupby("zz"))
    add("err groupby empty by", lambda: gdf().groupby([]))
    add("err groupby dup by", lambda: gdf().groupby(["k", "k"]))
    add("err groupby positional sort", lambda: gdf().groupby("k", False))
    add("err groupby series key", lambda: gdf().groupby(F.Series([1] * 12)))

    bnum = ["v_f", "v_i", "v_b"]
    for by in ["k_i", "k_s", "k_f", "k_b", ["k_i", "k_s"], ["k_s", "k_b", "k_f"]]:
        bn = by if isinstance(by, str) else "+".join(by)
        for sort, dropna in [(True, True), (False, False), (True, False)]:
            for f, kw in gfuncs:
                if f in ("std", "var"):
                    vals = ["v_f", "v_i"]
                elif f in ("sum", "mean"):
                    vals = bnum
                else:
                    vals = None
                if vals is not None:
                    fn = lambda by=by, sort=sort, dropna=dropna, f=f, kw=kw, vals=vals: gcall(
                        D.BDF.groupby(by, sort=sort, dropna=dropna)[vals], f, kw
                    )
                else:
                    fn = lambda by=by, sort=sort, dropna=dropna, f=f, kw=kw: gcall(
                        D.BDF.groupby(by, sort=sort, dropna=dropna), f, kw
                    )
                add(f"big groupby {bn} sort={sort} dropna={dropna} {gname(f, kw)}", fn)
    for f in ["sum", "mean", "std", "var", "min", "max"]:
        add(f"big groupby noNaN {f}", lambda f=f: getattr(D.BDF_noNaN.groupby("k")[["v", "w"]], f)())
        add(
            f"big groupby noNaN multi {f}",
            lambda f=f: getattr(D.BDF_noNaN.groupby(["k2", "k"], sort=False)[["v", "w"]], f)(),
        )
    add(
        "big groupby agg named",
        lambda: D.BDF.groupby("k_s").agg(
            a=("v_f", "sum"), b=("v_i", "mean"), c=("v_s", "max"), d=("v_f", "std"), e=("v_b", "count")
        ),
    )
    add("big groupby agg dict", lambda: D.BDF.groupby(["k_b", "k_i"], dropna=False).agg({"v_s": "first", "v_f": "var"}))

    # ---------------------------------------------------------------- sorting (§17)
    for k, mk in [("f", sf), ("i", si), ("b", sb), ("s", ss), ("f2", sf2)]:
        for asc in (True, False):
            for nap in ("last", "first"):
                add(
                    f"sort_values {k} asc={asc} na={nap}",
                    lambda mk=mk, asc=asc, nap=nap: mk().sort_values(**_kw(dict(ascending=asc, na_position=nap))),
                )
        add(f"sort_values {k} ignore_index", lambda mk=mk: mk().sort_values(ignore_index=True))
    add("sort_values labelled", lambda: sl().sort_values(**_kw(dict(ascending=False))))
    add("sort_values empty", lambda: F.Series([], dtype="string").sort_values())
    for asc in (True, False):
        kw = dict(ascending=asc)
        add(
            f"sort_index str asc={asc}",
            lambda kw=kw: F.Series(list(range(15)), index=list(reversed(LBL))).sort_index(**_kw(kw)),
        )
        add(f"sort_index int asc={asc}", lambda kw=kw: sint().sort_index(**_kw(kw)))
        add(f"sort_index range asc={asc}", lambda kw=kw: sf().sort_index(**_kw(kw)))
        add(f"sort_index dup asc={asc}", lambda kw=kw: sdup().sort_index(**_kw(kw)))
        add(f"df sort_index asc={asc}", lambda kw=kw: sdf_l().iloc[[5, 0, 12, 3]].sort_index(**_kw(kw)))
    add("sort_index ignore_index", lambda: sint().sort_index(ignore_index=True))
    add("df sort_index ignore_index", lambda: sdf_l().sort_index(**_kw(dict(ascending=False, ignore_index=True))))
    for by, asc, nap in [
        ("f", True, "last"),
        ("s", False, "first"),
        (["b", "f"], [True, False], ["first", "last"]),
        (["b", "s", "i"], False, "first"),
        (["i"], [False], ["last"]),
        (["s", "b"], True, ["last", "first"]),
    ]:
        add(
            f"df sort_values {by} {asc} {nap}",
            lambda by=by, asc=asc, nap=nap: sdf_l().sort_values(by, **_kw(dict(ascending=asc, na_position=nap))),
        )
    add("df sort_values ignore_index", lambda: sdf_l().sort_values("i", ignore_index=True))
    add("err sort_values na_position", lambda: sf().sort_values(na_position="middle"))
    add("err sort_values ascending int", lambda: sf().sort_values(**_kw(dict(ascending=1))))
    add("err sort_values positional", lambda: sf().sort_values(False))
    add("err df sort_values unknown", lambda: sdf().sort_values("zz"))
    add("err df sort_values empty", lambda: sdf().sort_values([]))
    add("err df sort_values ascending length", lambda: sdf().sort_values(["f", "i"], **_kw(dict(ascending=[True]))))
    add("err df sort_values na length", lambda: sdf().sort_values(["f"], na_position=["first", "last"]))
    add("err sort_values kind", lambda: sf().sort_values(kind="stable"))
    for k, get in [
        ("f", lambda: D.BF),
        ("i", lambda: D.BI),
        ("s", lambda: D.BS),
        ("b", lambda: D.BB),
        ("huge f", lambda: D.HF),
        ("huge i", lambda: D.HI),
    ]:
        for asc, nap in [(True, "last"), (False, "first"), (False, "last")]:
            add(
                f"big sort_values {k} asc={asc} na={nap}",
                lambda get=get, asc=asc, nap=nap: get().sort_values(**_kw(dict(ascending=asc, na_position=nap))),
            )
    add("big sort_values ignore_index", lambda: D.BF.sort_values(ignore_index=True))
    add("big sort_index int", lambda: D.ALIGN[0].sort_index())
    add("big sort_index int desc", lambda: D.ALIGN[0].sort_index(**_kw(dict(ascending=False))))
    add("big sort_index string", lambda: D.ALIGN[2].sort_index())
    add(
        "big sort_index dup",
        lambda: F.Series(D.base["ints"], index=D.base["ints"]).sort_index(**_kw(dict(ascending=False))),
    )
    add(
        "big df sort_values multi",
        lambda: D.BDF.sort_values(["k_s", "v_f"], **_kw(dict(ascending=[True, False], na_position=["last", "first"]))),
    )
    add(
        "big df sort_values 4 keys",
        lambda: D.BDF.sort_values(["k_b", "k_f", "k_i", "v_s"], **_kw(dict(ascending=False))),
    )
    add("big df sort_values ignore_index", lambda: D.BDF.sort_values("v_i", ignore_index=True, na_position="first"))
    add("big df sort_index", lambda: D.BDF.iloc[D.PERM].sort_index(**_kw(dict(ascending=False))))

    # ---------------------------------------------------------------- merge (§5, §18)
    def ml():
        return F.DataFrame(
            {
                "k": [1, 2, 2, None, 3, 5, 1],
                "kf": [0.0, nan, -0.0, 1.5, nan, None, 2.5],
                "ks": ["a", "b", "b", None, "c", "e", "a"],
                "lv": [10, 20, 21, 30, 40, 50, 11],
                "x": ["l0", "l1", "l2", "l3", "l4", "l5", "l6"],
            }
        )

    def mr():
        return F.DataFrame(
            {
                "k": [2, 1, None, 4, 2, 6],
                "kf": [-0.0, nan, 1.5, None, 0.0, 9.0],
                "ks": ["b", "a", None, "d", "b", "f"],
                "rv": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                "x": ["r0", "r1", "r2", "r3", "r4", "r5"],
            }
        )

    for how in ["inner", "left", "right", "outer"]:
        for key in ["k", "kf", "ks", ["k", "ks"]]:
            kn = key if isinstance(key, str) else "+".join(key)
            add(f"merge {how} on {kn}", lambda how=how, key=key: F.merge(ml(), mr(), how=how, on=key))
            add(f"merge {how} on {kn} sort", lambda how=how, key=key: F.merge(ml(), mr(), how=how, on=key, sort=True))
        add(
            f"merge {how} left_on/right_on",
            lambda how=how: F.merge(ml(), mr().iloc[:, [0, 3, 4]], how=how, left_on="lv", right_on="k"),
        )
        add(
            f"merge {how} left_on/right_on sort",
            lambda how=how: F.merge(
                ml(),
                F.DataFrame({"kk": [21, 10, 99, 11], "z": [1, 2, 3, 4]}),
                how=how,
                left_on=["lv"],
                right_on=["kk"],
                sort=True,
            ),
        )
        add(f"merge {how} method", lambda how=how: ml().merge(mr(), how=how, on="ks", suffixes=("_L", "_R")))
    add("merge left_on/right_on same name", lambda: F.merge(ml(), mr(), left_on="k", right_on="k", how="outer"))
    add("merge on overlapping key names kept", lambda: F.merge(ml(), mr(), left_on=["ks"], right_on=["x"], how="outer"))
    add("merge empty", lambda: F.merge(ml().iloc[0:0], mr(), on="k", how="outer"))
    add("merge suffix empty string", lambda: F.merge(ml(), mr(), on="k", suffixes=("", "_r")))
    add("err merge no keys", lambda: F.merge(ml(), mr()))
    add("err merge on and left_on", lambda: F.merge(ml(), mr(), on="k", left_on="k", right_on="k"))
    add("err merge key count", lambda: F.merge(ml(), mr(), left_on=["k", "ks"], right_on=["k"]))
    add("err merge dtype mismatch", lambda: F.merge(ml(), mr(), left_on="k", right_on="kf"))
    add("err merge how cross", lambda: F.merge(ml(), mr(), how="cross", on="k"))
    add("err merge unknown key", lambda: F.merge(ml(), mr(), on="zz"))
    add(
        "err merge suffix collision",
        lambda: F.merge(F.DataFrame({"k": [1], "a": [1], "a_x": [2]}), F.DataFrame({"k": [1], "a": [3]}), on="k"),
    )
    add("err merge suffixes list", lambda: F.merge(ml(), mr(), on="k", suffixes=["_a", "_b"]))
    add("err merge series", lambda: F.merge(ml(), mr()["k"], on="k"))
    add("err merge on empty", lambda: F.merge(ml(), mr(), on=[]))
    add("err merge on int", lambda: F.merge(ml(), mr(), on=[1]))
    add("err merge positional how", lambda: ml().merge(mr(), "inner"))
    for how in ["inner", "left", "right", "outer"]:
        add(f"big merge {how} int", lambda how=how: F.merge(D.MERGE[0], D.MERGE[1], how=how, on="key"))
        add(f"big merge {how} multi", lambda how=how: F.merge(D.MERGE[0], D.MERGE[1], how=how, on=["key", "tag"]))
        add(f"big merge {how} float", lambda how=how: F.merge(D.MERGE[2], D.MERGE[3], how=how, on="fk"))
        add(f"big merge {how} string", lambda how=how: D.MERGE[4].merge(D.MERGE[5], how=how, on="s"))
        add(
            f"big merge {how} left_on sort",
            lambda how=how: F.merge(D.MERGE[0], D.MERGE[1], how=how, left_on="key", right_on="key", sort=True),
        )
    add(
        "big merge outer sort multi", lambda: F.merge(D.MERGE[0], D.MERGE[1], how="outer", on=["tag", "key"], sort=True)
    )

    # ---------------------------------------------------------------- concat (§18)
    add("concat dfs outer", lambda: F.concat([sdf_l().iloc[:3], F.DataFrame({"s": ["q"], "z": [1.5]}, index=["n0"])]))
    add("concat dfs inner", lambda: F.concat([sdf().iloc[:3], F.DataFrame({"s": ["q"], "i": [7]})], join="inner"))
    add("concat dfs ignore_index", lambda: F.concat([sdf().iloc[2:4], sdf().iloc[:2]], ignore_index=True))
    add("concat dfs labels dup", lambda: F.concat([sdf().iloc[2:4], sdf().iloc[:3]]))
    add(
        "concat dfs index names",
        lambda: F.concat(
            [
                F.DataFrame({"a": [1]}, index=F.Index(["x"], name="n")),
                F.DataFrame({"a": [2]}, index=F.Index(["y"], name="n")),
            ]
        ),
    )
    add(
        "concat dfs index names differ",
        lambda: F.concat(
            [
                F.DataFrame({"a": [1]}, index=F.Index(["x"], name="n")),
                F.DataFrame({"a": [2]}, index=F.Index(["y"], name="m")),
            ]
        ),
    )
    add("concat series", lambda: F.concat([sf(), sf2()]))
    add("concat series same name", lambda: F.concat([sf(), sf().iloc[:2]]))
    add("concat series labelled", lambda: F.concat([sl(), sl().iloc[::-3]]))
    add("concat series ignore_index", lambda: F.concat((ss(), ss2()), ignore_index=True))
    add("concat one", lambda: F.concat([si()]))
    add("concat generator", lambda: F.concat(x for x in [si(), si2()]))
    add("concat empty frames", lambda: F.concat([F.DataFrame({"a": []}), F.DataFrame({"a": [1.0]})]))
    add("err concat empty", lambda: F.concat([]))
    add("err concat mix", lambda: F.concat([sdf(), sf()]))
    add("err concat series dtypes", lambda: F.concat([sf(), si()]))
    add("err concat df dtypes", lambda: F.concat([F.DataFrame({"a": [1]}), F.DataFrame({"a": [1.0]})]))
    add("err concat labels int string", lambda: F.concat([sf(), sl()]))
    add("err concat join", lambda: F.concat([sf()], join="left"))
    add("err concat axis", lambda: F.concat([sf()], axis=1))
    add("err concat positional", lambda: F.concat([sf()], "outer"))
    add("big concat dfs", lambda: F.concat([D.BDF, D.BDF.iloc[::-5], F.DataFrame({"k_i": [1], "zz": ["q"]})]))
    add(
        "big concat dfs inner ignore",
        lambda: F.concat([D.BDF.iloc[D.PERM], D.BDF[["v_s", "k_i"]]], join="inner", ignore_index=True),
    )
    add("big concat series", lambda: F.concat([D.BS, D.BS.iloc[1000:], D.BS[D.MASK]]))
    add("big concat series ignore", lambda: F.concat([D.HF, D.BF_raw], ignore_index=True))

    # ---------------------------------------------------------------- read_csv (§19)
    def rc(text, **kw):
        return lambda: F.read_csv(_csv_path(text), **kw)

    csv_cases = [
        ("quoting bom crlf", b'\xef\xbb\xbfk,s\r\n1,"he said ""hi"""\r\n2,""\r\n3,"a,b\nc"\r\n4,\r\n5,"12"', {}),
        ("lone cr", b"a,b\r\nx\r,y\r\nz,w\r", {}),
        ("quoted numbers", 'a\n"12"\n3\n', {}),
        ("na tokens default", "a,b\n1.5,x\nNaN,NA\n,null\n", {}),
        ("na_values", "a,b\n1.5,x\nNaN,NA\n,null\n", {"na_values": ["NA", "null"]}),
        ("na_values numeric", "a,b\n1,-999\n-999,2\n", {"na_values": ["-999"]}),
        ("blank line one column", "a\n1\n\n3\n", {}),
        (
            "inference",
            "i,f,b,s,big,inexact,space,under,allna,e\n1,1.5,True,x,1,9007199254740993,1,1,,1e5\n"
            "-2,nan,False,2,18446744073709551616,0.5, 1,1_0,,-inf\n+3,1e400,,,3,2,3,3,,inf\n",
            {},
        ),
        ("float specials", "x\nnan\nNaN\n-inf\nInfinity\n-0.0\n.5\n5.\n1E-5\n+7\n", {}),
        ("int bounds", "x\n9223372036854775807\n-9223372036854775808\n", {}),
        ("int over", "x\n9223372036854775808\n1\n", {}),
        ("bool case", "x\nTrue\nfalse\n", {}),
        ("unicode", "이름,값\n가,1\n😀,2\n", {}),
        ("quoted empty vs NA", 'a,b\n"",\n,""\n', {}),
        ("header only", "a,b\n", {}),
        ("no trailing newline", "a,b\n1,2", {}),
        ("crlf", "a,b\r\n1,x\r\n2,y\r\n", {}),
        ("quoted header", '"a,1","b""q"\n1,2\n', {}),
        ("sep semicolon", "k;x;y\nb;1;2.5\na;2;3.5\nc;3;4.5\n", {"sep": ";"}),
        ("sep tab", "a\tb\n1\t2\n", {"sep": "\t"}),
        (
            "options",
            "k;x;y\nb;1;2.5\na;2;3.5\nc;3;4.5\n",
            {"sep": ";", "usecols": ["y", "k"], "index_col": "k", "nrows": 2, "dtype": {"y": "string"}},
        ),
        ("dtype all string", "k;x;y\nb;1;2.5\n", {"sep": ";", "dtype": "string"}),
        ("dtype float for ints", "a,b\n1,2\n3,4\n", {"dtype": {"a": "float64"}}),
        ("dtype bool", "a\nTrue\n\nFalse\n", {"dtype": {"a": "bool"}}),
        ("header False names", "1,2\n3,4\n", {"header": False, "names": ["a", "b"]}),
        ("names rename", "1,2\n3,4\n", {"names": ["p", "q"]}),
        ("nrows 0", "a,b\n1,2\n", {"nrows": 0}),
        ("nrows big", "a,b\n1,2\n", {"nrows": 10}),
        ("index_col int", "id,v\n3,a\n1,b\n", {"index_col": "id"}),
        ("usecols order", "a,b,c\n1,2,3\n", {"usecols": ["c", "a"]}),
        ("usecols empty", "a,b,c\n1,2,3\n", {"usecols": []}),
        ("all NA column", "a,b\n1,\n2,\n", {}),
        ("one column NA lines", "a\n\n\n", {}),
    ]
    for nm, text, kw in csv_cases:
        add(f"csv {nm}", rc(text, **kw))
    add("csv str path", lambda: F.read_csv(str(_csv_path("a\n1\n"))))
    csv_errs = [
        ("fields short", "a,b\n1\n", {}),
        ("fields long", "a,b\n1,2,3\n", {}),
        ("text after quote", 'a\n"x"y\n', {}),
        ("quote in unquoted", 'a\nx"y\n', {}),
        ("never closed", 'a\n"never closed\n', {}),
        ("dup header", "a,a\n1,2\n", {}),
        ("blank line two columns", "a,b\n1,2\n\n3,4\n", {}),
        ("quote then cr", b'a,b\n"q"\r,z\n', {}),
        ("empty file", "", {}),
        ("invalid utf8", b"a\n\xff\xfe\n", {}),
        ("header False no names", "1,2\n", {"header": False}),
        ("sep two chars", "a\n", {"sep": ";;"}),
        ("sep quote", "a\n", {"sep": '"'}),
        ("usecols unknown", "a\n1\n", {"usecols": ["zz"]}),
        ("usecols repeat", "a\n1\n", {"usecols": ["a", "a"]}),
        ("usecols str", "a\n1\n", {"usecols": "a"}),
        ("dtype parse", "a\n1\nx\n", {"dtype": {"a": "int64"}}),
        ("dtype bool parse", "a\n1\n", {"dtype": {"a": "bool"}}),
        ("dtype unknown column", "a\n1\n", {"dtype": {"zz": "int64"}}),
        ("dtype bad", "a\n1\n", {"dtype": "int32"}),
        ("index_col float", "a,b\n1.5,2\n", {"index_col": "a"}),
        ("index_col NA", "a,b\n,2\n", {"index_col": "a"}),
        ("index_col unknown", "a,b\n1,2\n", {"index_col": "zz"}),
        ("nrows negative", "a\n1\n", {"nrows": -1}),
        ("nrows bool", "a\n1\n", {"nrows": True}),
        ("names count", "a,b\n1,2\n", {"names": ["x"]}),
        ("header not bool", "a\n1\n", {"header": 1}),
        ("na_values str", "a\n1\n", {"na_values": "NA"}),
        ("names dup", "1,2\n", {"header": False, "names": ["a", "a"]}),
        ("string invalid utf8 in quotes", b'a\n"\xc3"\n', {}),
    ]
    for nm, text, kw in csv_errs:
        add(f"err csv {nm}", rc(text, **kw))
    add("err csv positional sep", lambda: F.read_csv(_csv_path("a;b\n1;2\n"), ";"))
    add("err csv missing file", lambda: F.read_csv("/nonexistent-stardust-golden/x.csv"))

    add("big csv", lambda: F.read_csv(_csv_path(D.BIG_CSV)))
    add("big csv crlf bom", lambda: F.read_csv(_csv_path(b"\xef\xbb\xbf" + D.BIG_CSV.replace("\n", "\r\n").encode())))
    add(
        "big csv usecols index_col",
        lambda: F.read_csv(_csv_path(D.BIG_CSV), usecols=["name", "x", "id"], index_col="id"),
    )
    add("big csv nrows", lambda: F.read_csv(_csv_path(D.BIG_CSV), nrows=100_000))
    add(
        "big csv dtype",
        lambda: F.read_csv(_csv_path(D.BIG_CSV), dtype={"qint": "float64", "flag": "string", "id": "string"}),
    )
    add("big csv na_values", lambda: F.read_csv(_csv_path(D.BIG_CSV), na_values=["NaN", "True", "n7"]))
    add(
        "big csv header False",
        lambda: F.read_csv(
            _csv_path(D.BIG_CSV.split("\n", 1)[1]), header=False, names=["a", "b", "c", "d", "e", "f", "g", "h"]
        ),
    )

    def big_err(mutate):
        def run():
            lines = D.BIG_CSV.split("\n")
            mutate(lines)
            return F.read_csv(_csv_path("\n".join(lines)))

        return run

    def two_bad(lines):
        lines[140_000] += ",extra"
        lines[100_000] = lines[100_000].rsplit(",", 1)[0]

    def bad_quote(lines):
        lines[130_000] = lines[130_000].replace(",", ',x"y', 1)
        lines[149_000] += ",1"

    def unclosed(lines):
        lines[120_000] = '"' + lines[120_000]

    def parse_late(lines):
        lines[0] += ",late"
        for j in range(1, len(lines) - 1):
            lines[j] += ",1"
        lines[110_000] = lines[110_000][:-1] + "x"
        lines[60_000] = lines[60_000][:-1] + "y"

    add("err big csv two bad rows", big_err(two_bad))
    add("err big csv bad quote", big_err(bad_quote))
    add("err big csv unclosed", big_err(unclosed))

    def parse_late_run():
        lines = D.BIG_CSV.split("\n")
        parse_late(lines)
        return F.read_csv(_csv_path("\n".join(lines)), dtype={"late": "int64"})

    add("err big csv dtype late", parse_late_run)

    def infer_late():  # the inference of a column turns on its last value
        lines = D.BIG_CSV.split("\n")
        lines[0] += ",late"
        for j in range(1, len(lines) - 1):
            lines[j] += f",{j}"
        lines[-2] = lines[-2][:-1] + "1.5"
        return F.read_csv(_csv_path("\n".join(lines)), usecols=["late"])

    add("big csv inference late", infer_late)

    # ---------------------------------------------------------------- repr (§9)
    def wide(ncols, n=3):
        return F.DataFrame({f"c{j}": [j * 1.5 if j % 2 else j for _ in range(n)] for j in range(ncols)})

    reprs = {
        "series f": sf,
        "series f2": sf2,
        "series i": si,
        "series b": sb,
        "series s": ss,
        "series labelled": sl,
        "series named index": lambda: F.Series([1, -2], index=F.Index(["a", "bb"], name="ix"), name="v"),
        "series unnamed": lambda: F.Series([1, None, -1]),
        "series empty": lambda: F.Series([]),
        "series empty string": lambda: F.Series([], dtype="string"),
        "series long": lambda: F.Series(range(100)),
        "series 60": lambda: F.Series(range(60)),
        "series 61": lambda: F.Series(range(61)),
        "series exp": lambda: F.Series([1e-10, 1.0, 1e10]),
        "series exp big": lambda: F.Series([1e17, 2.5]),
        "series small floats": lambda: F.Series([0.000123456, 12.5, None, nan]),
        "series ints float": lambda: F.Series([1.0, 2.0, -0.0]),
        "series int64 bounds": lambda: F.Series([I64_MIN, I64_MAX, None]),
        "series inf": lambda: F.Series([inf, -inf, nan, 1.0]),
        "df": sdf,
        "df labelled": sdf_l,
        "df empty": lambda: F.DataFrame(),
        "df no rows": lambda: F.DataFrame(columns=["a", "b"]),
        "df no columns": lambda: F.DataFrame(index=[1, 2]),
        "df long": lambda: F.DataFrame({"a": list(range(100)), "b": [x / 3 for x in range(100)], "c": ["s"] * 100}),
        "df wide": lambda: wide(25),
        "df wide 20": lambda: wide(20),
        "df wide 21": lambda: wide(21),
        "df wide long": lambda: wide(30, 80),
        "df named index": lambda: F.DataFrame({"a": [1]}, index=F.Index([7], name="k")),
        "df mixed exp": lambda: F.DataFrame({"a": [1e-8, 1.0], "b": [1, 2], "c": [True, None]}),
        "groupby result": lambda: gdf().groupby("ks", dropna=False).agg(m=("v", "mean"), c=("n", "count")),
        "value_counts": lambda: ss().value_counts(dropna=False),
        "index str": lambda: F.Index(["a", "b"]),
        "index int named": lambda: F.Index([1, 2, 3], name="ix"),
        "index long": lambda: F.Index(list(range(200))),
        "index long str": lambda: F.Index([f"x{j}" for j in range(150)], name="n"),
        "index empty": lambda: F.Index([]),
        "rangeindex": lambda: F.RangeIndex(5),
        "rangeindex named": lambda: F.RangeIndex(2, 20, 3, name="r"),
        "columns": lambda: sdf().columns,
        "dtypes": lambda: sdf().dtypes,
        "nunique": lambda: sdf().nunique(),
        "df reduce": lambda: F.DataFrame({"f": SF, "i": SI}).sum(),
        "big series": lambda: D.BF,
        "big series string": lambda: D.BS,
        "big df": lambda: D.BDF,
        "big index": lambda: D.ALIGN[0].index,
        "huge series": lambda: D.HF,
    }
    for nm, mk in reprs.items():
        add(f"repr {nm}", lambda mk=mk: repr(mk()))
    add("str series", lambda: str(sf()))
    add("str df", lambda: str(sdf()))

    # ---------------------------------------------------------------- signatures (§11, C2)
    add("err sig unknown keyword", lambda: si().fillna(0, extra=1))  # value is positional-only in M6 (§9.3)
    add("sig keyword head", lambda: si().head(n=2))
    add("sig keyword astype", lambda: si().astype(dtype="float64"))
    add("sig keyword fillna", lambda: si().fillna(value=0))
    add("sig keyword reindex", lambda: si().reindex(labels=[0, 1]))
    add("sig keyword df sort_values", lambda: sdf().sort_values(by="f"))
    add("sig keyword groupby", lambda: sdf().groupby(by="b").size())
    add("sig keyword duplicated", lambda: sdf().duplicated(subset=["b"]))
    add("sig keyword merge", lambda: ml().merge(right=mr(), on="k"))
    add("sig keyword concat", lambda: F.concat(objs=[si()]))
    add("sig keyword Index", lambda: F.Index(data=[1]))
    add("sig keyword Series", lambda: F.Series(data=[1]))
    add("sig keyword equals", lambda: F.Index([1]).equals(other=F.Index([1])))
    add("err sig isnull", lambda: si().isnull())
    add("err sig tolist", lambda: si().to_list() if _NEW[0] else si().tolist())  # the alias stays absent (§9.4)
    add("err sig add method", lambda: si().add(1))
    add("err sig series groupby", lambda: si().groupby([1]))
    add("err sig dropna positional", lambda: sdf().dropna("any"))
    add("err sig agg func keyword", lambda: gdf().groupby("k").agg(func="sum"))

    # ---------------------------------------------------------------- threads (§20)
    # The thread count is library-wide: stardust_pandas.set_num_threads became
    # sd.set_num_threads (ROADMAP M5).
    T = L._threads

    def threads_roundtrip():
        n = T.get_num_threads()
        T.set_num_threads(n)
        return isinstance(n, int) and n >= 1 and T.get_num_threads() == n

    def threads_bad(arg, **kw):
        def run():
            n = T.get_num_threads()
            try:
                return T.set_num_threads(arg, **kw) if not kw else T.set_num_threads(**kw)
            finally:
                T.set_num_threads(n)

        return run

    add("threads roundtrip", threads_roundtrip)
    add("err threads zero", threads_bad(0))
    add("err threads negative", threads_bad(-1))
    add("err threads float", threads_bad(2.0))
    add("err threads keyword", threads_bad(None, n=2))
    add("err threads get arg", lambda: T.get_num_threads(1))

    yield from out
