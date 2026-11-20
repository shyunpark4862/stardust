"""The whole array area against numpy: every public function at 3, 10^3 .. 10^7
elements, in several layouts (contiguous, strided, transposed, broadcast) and dtypes.

numpy runs as it is (one thread), except where its BLAS does the work. Each case is a statement written
once with `M` for the library (M = sd on stardust's side, M = np on numpy's), or two
statements where the calls differ; the statement is compiled into a function whose
globals hold that side's arrays, so the timed call is the statement itself.

Fresh inputs. A branchy kernel timed on one input over and over lets the branch
predictor learn that input (numpy's argsort of 10^3 elements: 7.6 µs on one input,
29.3 µs on fresh ones). So every branchy computation -- sort, argsort, median,
quantile, nanmedian, nanquantile, unique_*, searchsorted, nonzero, boolean-mask
selection (a[mask], a[a > c]) and mask assignment -- is timed on harness.rotate over
K = harness.fresh_count(n) inputs of the same distribution, each from its own seed
(K from the case's largest array, so about 4·10^6 elements per array), on both sides
the same way. Elementwise operations, the plain and NaN-skipping reductions (the
NaN-skipping ones select with masks, no branch per element, on both sides), scans,
gathers and scatters keep one input.

Threads. sd's 1-thread rows run with Accelerate on one thread too, so the BLAS cases
(matmul, cov, corrcoef, tensordot) differ between 1 and 10 threads; there numpy "follows"
(its BLAS at sd's thread count, in the same worker), so each thread count meets its own. Cases with no parallel path
(the fills of the creation functions, views, Python conversions, scatters, 10^3 and
10^4 elements below the 2^15 threshold, tiny arrays) run on one thread only, except
s3 + t3s and a + b at 10^3, kept at 10 threads to catch the pool being used below the
threshold. The size group adds 2^14 and 2^16 around the elementwise threshold and
2^18 past the single-lane parallel sort (2 · 2^16).

Comparators. sd sorts stably by default, so numpy's sort and argsort run with
stable=True; "numpy default (unstable)" is numpy's default for a few of them. numpy's
unique_values hashes and leaves the values unsorted; sd sorts them, so the comparator
is np.unique (on this machine both take the same time, so no hash column).

R joins where the accuracy measurement compares with R (bench/accuracy/array.py:
R_ELEMENTWISE and the reductions of _red_values) and the case does the same work: the
float64 element-wise functions and complex abs on contiguous arrays, sum (also over
axis 0), mean, prod and matmul (R_STMTS below). Not var and std (sd's default
correction is 0, R's var and sd divide by n - 1), not the row-wise vecdot of two
matrices (the accuracy side's vecdot is the dot product of two vectors), not strided,
transposed-view or out= inputs (R has no views: its copy would be contiguous), not
batched or integer matmul. R gets plain float64 copies of the numpy side's arrays
under "r_<name>" keys; complex numbers are built in R's setup.

The arrays:
    f, g     10^6 float64 in [0.5, 1.5)          fs   10^6 standard normal
    u        10^6 in (-0.99, 0.99)               fh   10^6 in [1.5, 2.5)
    fn       fs with 10% NaN                     zh   half ±0.0, half normal
    i        10^6 int64 in [-1000, 1000)         j    10^6 int64 in [1, 1000)
    sh       shifts in [0, 63)                   b, b2  10^6 bool, half True
    s        10^6 bool, 1% True                  bz, bo  10^6 bool, all False / all True
    c, cc    10^6 complex128                     iv   10^6 int64 in [0, 1000)
    ivw      10^6 int64 in ±10^12                srt  sorted 10^6 float64 in [0, 1)
    kn, keys, k1k  10^6, 10^5, 10^3 keys in [0, 1)
    f2, g2   (1000, 1000) float64                i2   (1000, 1000) int64
    ft       f2.T (transposed view)              fst  big2[::2] (stride 2, 10^6)
    v        1000 float64                        t3   (100, 100, 100)
    p8       (125000, 8) float64 (short lanes)   fm, gm   10^4 float64
    s3, t3s, i3, sm  tiny (3 or 3 x 4)
    x, fw, gz, y  written in place (copies of f, zeros)
    np_*     numpy arrays given to both sides (conversions)

Every array is drawn from its own seeded generator, so a case builds only the arrays it
uses."""

from __future__ import annotations

import ast
import math
import zlib

from ..harness import fresh_count, rotate
from ..registry import Case, Check, Ref, RRef

N = 1_000_000
MID = 10_000
# the size group: data key suffix -> (size label, elements)
SIZES = {
    "1e3": ("10^3", 1_000),
    "p14": ("2^14", 1 << 14),
    "p16": ("2^16", 1 << 16),
    "1e5": ("10^5", 100_000),
    "p18": ("2^18", 1 << 18),
    "1e7": ("10^7", 10_000_000),
}

BOTH = (1, 10)
T1 = (1,)  # no parallel path


# ----------------------------------------------------------------- data


def _builders(np):
    """key -> build(rng, get): the numpy data, each from its own generator."""
    nan = np.nan
    B = {
        "f": lambda r, get: r.random(N) + 0.5,  # positive: log, sqrt
        "g": lambda r, get: r.random(N) + 0.5,
        "fs": lambda r, get: r.standard_normal(N),  # both signs
        "u": lambda r, get: r.random(N) * 1.98 - 0.99,  # (-1, 1): asin, acos, atanh
        "fh": lambda r, get: r.random(N) + 1.5,  # > 1: acosh
        "fn": lambda r, get: np.where(r.random(N) < 0.1, nan, get("fs")),  # 10% NaN
        "i": lambda r, get: r.integers(-1000, 1000, N),
        "j": lambda r, get: r.integers(1, 1000, N),
        "sh": lambda r, get: r.integers(0, 63, N),
        "b": lambda r, get: r.random(N) > 0.5,
        "b2": lambda r, get: r.random(N) > 0.5,
        "s": lambda r, get: r.random(N) < 0.01,  # sparse mask
        "bz": lambda r, get: np.zeros(N, dtype=bool),
        "bo": lambda r, get: np.ones(N, dtype=bool),
        "c": lambda r, get: get("f") + 1j * r.random(N),
        "cc": lambda r, get: r.random(N) + 1j * r.random(N),
        "f2": lambda r, get: r.random((1000, 1000)),
        "g2": lambda r, get: r.random((1000, 1000)),
        "i2": lambda r, get: r.integers(-1000, 1000, (1000, 1000)),
        "v": lambda r, get: r.random(1000),
        "t3": lambda r, get: r.random((100, 100, 100)),
        "p8": lambda r, get: r.random((125_000, 8)),
        "big2": lambda r, get: r.random(2 * N),  # strided views come from here
        "m512": lambda r, get: r.random((512, 512)),
        "m1k": lambda r, get: r.random((1000, 1000)),
        "im": lambda r, get: r.integers(-100, 100, (200, 200)),
        "bm": lambda r, get: r.random((1000, 8, 8)),
        "obs": lambda r, get: r.random((20, 100_000)),
        "iv": lambda r, get: r.integers(0, 1000, N),
        "ivw": lambda r, get: r.integers(-(10**12), 10**12, N),
        "idx": lambda r, get: r.integers(0, N, 100_000),
        "idxn": lambda r, get: r.integers(0, N, N),
        "rows": lambda r, get: r.integers(0, 1000, 1000),
        "idx2": lambda r, get: r.integers(0, 1000, (1000, 1000)),
        "srt": lambda r, get: np.sort(r.random(N)),
        "kn": lambda r, get: r.random(N),
        "keys": lambda r, get: r.random(100_000),
        "k1k": lambda r, get: r.random(1000),
        "q5": lambda r, get: np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
        "zh": lambda r, get: (
            np.where(r.random(N) < 0.5, 0.0, r.standard_normal(N)) * np.where(r.random(N) < 0.5, 1, -1)
        ),
        "fm": lambda r, get: r.random(MID) + 0.5,
        "gm": lambda r, get: r.random(MID) + 0.5,
        "s3": lambda r, get: np.array([1.0, 2.0, 3.0]),
        "t3s": lambda r, get: np.array([4.0, 5.0, 6.0]),
        "i3": lambda r, get: np.array([1, 2, 3]),
        "sm": lambda r, get: np.arange(12.0).reshape(3, 4),
        # only for the conversions
        "m2k": lambda r, get: r.random((2000, 2000)),
        "be2k": lambda r, get: r.random((2000, 1000)),
    }
    for nm, (_, k) in SIZES.items():
        B["a" + nm] = lambda r, get, k=k: r.random(k) + 0.5
        B["b" + nm] = lambda r, get, k=k: r.random(k) + 0.5
        B["i" + nm] = lambda r, get, k=k: r.integers(-1000, 1000, k)
        B["j" + nm] = lambda r, get, k=k: r.integers(-1000, 1000, k)
    return B


# numpy arrays (or a list) handed to both sides as they are: the conversions
_CONV = {
    "np_f": lambda np, get: get("f"),
    "np_fst": lambda np, get: get("big2")[::2],
    "np_ft": lambda np, get: get("f2").T,
    "np_f32": lambda np, get: get("f").astype(np.float32),
    "np_f32t": lambda np, get: get("m2k").astype(np.float32)[:1000, :1000].T,
    "np_f32r": lambda np, get: get("m2k").astype(np.float32)[:1000, 1:1001],
    "np_be": lambda np, get: get("f").astype(">f8"),
    "np_be2": lambda np, get: get("be2k").astype(">f8")[::2],
    "np_i32": lambda np, get: get("i").astype(np.int32),
    "np_list3": lambda np, get: [1.0, 2.0, 3.0],
    "np_list": lambda np, get: get("f")[:10_000].tolist(),
}

# per side, from that side's arrays (lib is np or sd; side(key) is that side's array)
_DERIVED = {
    "fst": lambda lib, get, side: side("big2")[::2],  # strided, 10^6
    "ft": lambda lib, get, side: side("f2").T,  # transposed
    "x": lambda lib, get, side: lib.asarray(get("f"), copy=True),  # written in place
    "fw": lambda lib, get, side: lib.asarray(get("f"), copy=True),
    "gz": lambda lib, get, side: lib.zeros(N),
    "y": lambda lib, get, side: lib.zeros(N),  # out=
}


def _names(*stmts) -> set[str]:
    out = set()
    for s in stmts:
        if s:
            out |= {n.id for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Name)}
    return out


def _make_setup(names, alias, fresh):
    """A setup building only the arrays `names` refer to (through `alias`: name -> key).

    It returns {"np": [namespace, ...], "sd": [namespace, ...]}: one namespace per input,
    one unless `fresh`, then K = fresh_count(largest array) of them, input k drawn with
    the seed [0, crc32(key), k] (input 0 with [0, crc32(key)])."""

    def setup():
        import numpy as np

        import stardust as sd

        builders = _builders(np)
        raw: dict = {}

        def getter(k):
            def get(key):
                if (key, k) not in raw:
                    seed = [0, zlib.crc32(key.encode())] + ([k] if k else [])
                    raw[key, k] = builders[key](np.random.default_rng(seed), getter(k))
                return raw[key, k]

            return get

        per: dict = {}

        def side(lib_name, k):
            def at(key):
                if (lib_name, key, k) not in per:
                    v = getter(k)(key)
                    per[lib_name, key, k] = v.copy() if lib_name == "np" else sd.asarray(v, copy=True)
                return per[lib_name, key, k]

            return at

        pairs = [(name, alias.get(name, name)) for name in sorted(names)]
        conv = {name: _CONV[key](np, getter(0)) for name, key in pairs if key in _CONV}

        def namespaces(k):
            npn = {"M": np, "np": np, "sd": sd, "N": N, **conv}
            sdn = {"M": sd, "np": np, "sd": sd, "N": N, **conv}
            get = getter(k)
            for name, key in pairs:
                if key in _DERIVED:
                    npn[name] = _DERIVED[key](np, get, side("np", k))
                    sdn[name] = _DERIVED[key](sd, get, side("sd", k))
                elif key in builders:
                    npn[name] = side("np", k)(key)
                    sdn[name] = side("sd", k)(key)
                # other names (builtins, functions) resolve as usual
            return npn, sdn

        out = [namespaces(0)]
        if fresh:
            biggest = max((a.size for a in out[0][0].values() if isinstance(a, np.ndarray)), default=1)
            out += [namespaces(k) for k in range(1, fresh_count(biggest))]
        return {"np": [n for n, _ in out], "sd": [s for _, s in out]}

    return setup


def _compile(stmt: str, ns: dict):
    """A zero-argument function running stmt with ns as its globals."""
    try:
        ast.parse(stmt, mode="eval")
        src = f"def _f():\n    return {stmt}\n"
    except SyntaxError:
        src = f"def _f():\n    {stmt}\n"
    loc: dict = {}
    exec(compile(src, f"<array case: {stmt}>", "exec"), ns, loc)  # noqa: S102
    return loc["_f"]


def _n(size: str) -> int | None:
    if " x " in size:  # B x (m, n)
        b, shape = size.split(" x ", 1)
        return _n(b) * _n(shape)
    if size.startswith("10^"):
        return 10 ** int(size[3:])
    if size.startswith("2^"):
        return 2 ** int(size[2:])
    if size.startswith("("):
        return math.prod(int(p) for p in size.strip("()").split(",") if p.strip())
    if size.isdigit():
        return int(size)
    return None


def _strip(stmt: str) -> str:
    return stmt.replace("M.", "").replace("sd.", "")


def _stable(stmt: str) -> str:
    """numpy's stable sort for an sd sort statement: M.sort(x, ...) -> np.sort(x, ..., stable=True)."""
    return stmt.replace("M.", "np.")[:-1] + ", stable=True)"


# ---------------------------------------------------------------- cases
# _add(group, sd statement, size, np=numpy statement (default: the same text, M = np),
#      name=..., note=..., quick=..., alias={name in the statement: data key},
#      extra=((label, numpy statement), ...) further numpy comparators,
#      fresh=True: rotate over fresh inputs (branchy kernels), threads=sd's thread counts,
#      blas=True: numpy's BLAS follows sd's thread count)

G_CREATE = "creation"
G_CONVERT = "conversion"
G_MANIP = "reshape"
G_BIN_F = "binary-f64"
G_BIN_B = "binary-bool"
G_BIN_I = "binary-i64"
G_BIN_C = "binary-c128"
G_BIN_MIX = "binary-mixed"
G_BIN_LAYOUT = "binary-layout"
G_BIN_MID = "binary-1e4"
G_UN_F = "unary-f64"
G_UN_C = "unary-c128"
G_UN_I = "unary-i64"
G_UN_B = "unary-bool"
G_SELECT = "where"
G_RED_F = "reduce-f64"
G_RED_I = "reduce-i64"
G_RED_C = "reduce-c128"
G_RED_B = "reduce-bool"
G_RED_AX = "reduce-axis"
G_RED_MID = "reduce-1e4"
G_SCAN = "cumulative"
G_NAN = "nan-skipping"
G_COUNT = "counting"
G_STATS = "statistics"
G_SORT = "sorting"
G_UNIQUE = "unique"
G_SEARCH = "search-take"
G_INDEX = "indexing"
G_ASSIGN = "assignment"
G_MATMUL = "matmul"
G_SIZE = "by-size"
G_TINY = "small-arrays"

M6 = "10^6"
SQ = "(1000, 1000)"
T3 = "(100, 100, 100)"
P8 = "(125000, 8)"

UNSTABLE = "numpy default (unstable)"


def _specs() -> list[dict]:
    S: list[dict] = []

    def add(
        group,
        stmt,
        size,
        np=None,
        name=None,
        note="",
        quick=False,
        alias=None,
        extra=(),
        fresh=False,
        threads=BOTH,
        blas=False,
    ):
        S.append(
            {
                "group": group,
                "stmt": stmt,
                "size": size,
                "np": np,
                "name": name or _strip(stmt),
                "note": note,
                "quick": quick,
                "alias": alias or {},
                "extra": extra,
                "fresh": fresh,
                "threads": threads,
                "blas": blas,  # numpy's work is mostly BLAS: it follows sd's thread count
            }
        )

    # creation (the fills are serial loops)
    for s in [
        "zeros(N)",
        "ones(N)",
        "empty(N)",
        "full(N, 2.5)",
        "zeros_like(f)",
        "ones_like(f)",
        "empty_like(f)",
        "full_like(f, 2.5)",
        "arange(N)",
    ]:
        add(G_CREATE, "M." + s, M6, threads=T1)
    add(G_CREATE, "M.linspace(0.0, 1.0, N)", M6, threads=T1)
    add(G_CREATE, "M.eye(1000)", SQ, threads=T1)
    add(G_CREATE, "M.meshgrid(v, v)", SQ, note="1000개짜리 둘에서", threads=T1)
    add(G_CREATE, "M.tril(f2)", SQ)
    add(G_CREATE, "M.triu(f2)", SQ)
    add(G_CREATE, "M.zeros((1000, 1000))", SQ, threads=T1)
    add(G_CREATE, "M.arange(N, dtype=M.float64)", M6, threads=T1)
    add(G_CREATE, "M.asarray(f, copy=True)", M6)

    # conversion from numpy (sd.asarray(x) against np.asarray(x, dtype=float64): numpy keeps
    # the memory order, stardust makes C order)
    def conv(stmt, np, size, note, **kw):
        add(G_CONVERT, stmt, size, np=np, name=stmt, note=note, **kw)

    conv("sd.asarray(np_f)", "np.asarray(np_f)", M6, "f64, 복사 없음", threads=T1)
    conv("sd.asarray(np_fst)", "np.asarray(np_fst)", M6, "f64 stride 2")
    conv("sd.asarray(np_ft)", "np.asarray(np_ft)", SQ, "f64 전치")
    conv("sd.asarray(np_f32)", "np.asarray(np_f32, dtype=np.float64)", M6, "f32 → f64", quick=True)
    conv(
        "sd.asarray(np_f32t)",
        "np.asarray(np_f32t, dtype=np.float64)",
        SQ,
        "f32 전치 → f64; numpy는 F 순서를 지키고, 'numpy C order'는 C 순서로 만든다",
        extra=(("numpy C order", "np.ascontiguousarray(np_f32t, dtype=np.float64)"),),
    )
    conv("sd.asarray(np_f32r)", "np.asarray(np_f32r, dtype=np.float64)", SQ, "f32 행의 일부 m[:, 1:1001] → f64")
    conv("sd.asarray(np_be)", "np.asarray(np_be, dtype=np.float64)", M6, ">f8 → f64")
    conv("sd.asarray(np_be2)", "np.asarray(np_be2, dtype=np.float64)", SQ, ">f8 2-D 한 줄 걸러 → f64")
    conv("sd.asarray(np_i32)", "np.asarray(np_i32, dtype=np.int64)", M6, "i32 → i64")
    conv("sd.asarray(np_list3)", "np.asarray(np_list3)", "3", "파이썬 리스트", threads=T1)
    conv("sd.asarray(np_list)", "np.asarray(np_list)", "10^4", "파이썬 리스트", threads=T1)
    conv("sd.from_dlpack(np_f)", "np.from_dlpack(np_f)", M6, "f64", threads=T1)
    conv(
        "np.asarray(f)",
        "np.asarray(f)",
        M6,
        "sd 배열 → numpy; numpy 쪽은 받은 배열을 그대로 돌려줄 뿐이라 비율은 뜻이 없다",
        threads=T1,
    )
    add(G_CONVERT, "fm.tolist()", "10^4", note="파이썬 리스트로", threads=T1)
    add(G_CONVERT, "list(fm)", "10^4", note="원소마다 스칼라", threads=T1)

    # manipulation
    for s, size, note, th in [
        ("reshape(f, (1000, 1000))", M6, "뷰", T1),
        ("reshape(ft, (N,))", SQ, "전치라 복사", BOTH),
        ("ravel(ft)", SQ, "전치라 복사", BOTH),
        ("permute_dims(t3, (2, 0, 1))", T3, "뷰", T1),
        ("matrix_transpose(f2)", SQ, "뷰", T1),
        ("moveaxis(t3, 0, -1)", T3, "뷰", T1),
        ("squeeze(M.reshape(f, (1, N)), axis=0)", M6, "뷰", T1),
        ("expand_dims(f, axis=0)", M6, "뷰", T1),
        ("broadcast_to(v, (1000, 1000))", SQ, "뷰", T1),
        ("broadcast_arrays(f2, v)", SQ, "뷰", T1),
        ("concat([f, g])", M6, "", BOTH),
        ("concat([f2, g2], axis=1)", SQ, "", BOTH),
        ("stack([f, g])", M6, "", BOTH),
        ("flip(f)", M6, "뷰", T1),
        ("roll(f, 1)", M6, "", BOTH),
        ("roll(f2, 1, axis=1)", SQ, "", BOTH),
        ("repeat(f, 2)", M6, "", BOTH),
        ("tile(v, (1000,))", "10^3", "결과 10^6", BOTH),
        ("unstack(f2)", SQ, "뷰", T1),
        ("astype(f, M.int64)", M6, "", BOTH),
        ("astype(i, M.float64)", M6, "", BOTH),
        ("astype(f, M.complex128)", M6, "", BOTH),
        ("astype(b, M.float64)", M6, "", BOTH),
        ("astype(ft, M.float64)", SQ, "전치", BOTH),
    ]:
        add(G_MANIP, "M." + s, size, note=note, quick=(s == "concat([f2, g2], axis=1)"), threads=th)
    add(G_MANIP, "M.diff(f)", M6)

    # elementwise: binary
    for op in [
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "remainder",
        "pow",
        "atan2",
        "maximum",
        "minimum",
        "copysign",
        "hypot",
        "logaddexp",
        "nextafter",
        "equal",
        "less",
        "greater_equal",
    ]:
        add(G_BIN_F, f"M.{op}(f, g)", M6, quick=(op == "add"))
    for op in ["logical_and", "logical_or", "logical_xor"]:
        add(G_BIN_B, f"M.{op}(b, b2)", M6)
    for op in [
        "add",
        "multiply",
        "floor_divide",
        "remainder",
        "maximum",
        "equal",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    ]:
        add(G_BIN_I, f"M.{op}(i, j)", M6)
    add(G_BIN_I, "M.bitwise_left_shift(i, sh)", M6)
    add(G_BIN_I, "M.bitwise_right_shift(i, sh)", M6)
    add(G_BIN_I, "M.pow(i, 2)", M6)
    for op in ["add", "multiply", "divide", "equal", "pow"]:
        add(G_BIN_C, f"M.{op}(c, cc)", M6)
    add(G_BIN_MIX, "f * i", M6, note="f64 × i64 (형 변환)")
    add(G_BIN_MIX, "f * i + g", M6)
    add(G_BIN_MIX, "f + 1.0", M6, note="스칼라")
    add(G_BIN_MIX, "i + 1", M6, note="스칼라")
    add(G_BIN_MIX, "c * f", M6, note="c128 × f64")
    add(G_BIN_LAYOUT, "f2 + v", SQ, note="행 브로드캐스트")
    add(G_BIN_LAYOUT, "f2 + M.reshape(v, (1000, 1))", SQ, note="열 브로드캐스트")
    add(G_BIN_LAYOUT, "ft + 1.0", SQ, note="전치")
    add(G_BIN_LAYOUT, "ft + f2", SQ, note="전치 + C 순서", quick=True)
    add(G_BIN_LAYOUT, "fst + 1.0", M6, note="stride 2")
    add(G_BIN_LAYOUT, "fst + f", M6, note="stride 2 + 연속")
    add(G_BIN_LAYOUT, "M.flip(f) + g", M6, note="음의 stride + 연속")
    add(G_BIN_MID, "fm + gm", "10^4", threads=T1)
    add(G_BIN_MID, "fm * 2.0", "10^4", threads=T1)

    # elementwise: unary
    for op in [
        "negative",
        "positive",
        "abs",
        "square",
        "sqrt",
        "sign",
        "exp",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "floor",
        "ceil",
        "trunc",
        "round",
        "isnan",
        "isinf",
        "isfinite",
        "signbit",
        "reciprocal",
    ]:
        add(G_UN_F, f"M.{op}(f)", M6, quick=(op == "exp"))
    for op in ["asin", "acos", "atanh"]:
        add(G_UN_F, f"M.{op}(u)", M6, note="(-1, 1)")
    add(G_UN_F, "M.acosh(fh)", M6, note="> 1")
    add(G_UN_F, "M.sign(fs)", M6, note="양수·음수 섞임")
    add(G_UN_F, "M.exp(f, out=y)", M6, note="있는 배열에 씀")
    add(G_UN_F, "M.exp(fst)", M6, note="stride 2")
    add(G_UN_F, "M.sqrt(ft)", SQ, note="전치")
    add(G_UN_F, "M.exp(fm)", "10^4", threads=T1)
    for op in ["abs", "exp", "sqrt", "log", "sin", "conj", "real", "imag", "square", "isnan"]:
        add(G_UN_C, f"M.{op}(c)", M6)
    for op in ["abs", "negative", "square", "sign", "bitwise_invert"]:
        add(G_UN_I, f"M.{op}(i)", M6)
    add(G_UN_B, "M.logical_not(b)", M6)
    add(G_SELECT, "M.clip(f, min=0.6, max=1.2)", M6, np="np.clip(f, 0.6, 1.2)")
    add(G_SELECT, "M.clip(i, min=-10, max=10)", M6, np="np.clip(i, -10, 10)")
    add(G_SELECT, "M.where(b, f, g)", M6, quick=True)
    add(G_SELECT, "M.where(b, f, 0.0)", M6)

    # reductions
    for op in ["sum", "prod", "min", "max", "mean", "var", "std", "argmin", "argmax"]:
        add(G_RED_F, f"M.{op}(f)", M6, quick=(op == "sum"))
    for op in ["sum", "min", "max", "mean", "var", "argmax"]:
        add(G_RED_I, f"M.{op}(i)", M6)
    for op in ["sum", "prod", "mean", "var"]:
        add(G_RED_C, f"M.{op}(c)", M6)
    add(G_RED_B, "M.any(b)", M6, note="첫 원소에서 끝남")
    add(G_RED_B, "M.any(bz)", M6, note="끝까지 훑음")
    add(G_RED_B, "M.all(b)", M6, note="첫 원소에서 끝남")
    add(G_RED_B, "M.all(bo)", M6, note="끝까지 훑음")
    add(G_RED_B, "M.sum(b)", M6)
    for op in ["sum", "mean", "max", "var", "argmax"]:
        add(G_RED_AX, f"M.{op}(f2, axis=0)", SQ, quick=(op == "sum"))
        add(G_RED_AX, f"M.{op}(f2, axis=1)", SQ)
    add(G_RED_AX, "M.sum(t3, axis=(0, 2))", T3)
    add(G_RED_AX, "M.sum(ft)", SQ, note="전치")
    add(G_RED_AX, "M.sum(ft, axis=0)", SQ, note="전치")
    add(G_RED_AX, "M.sum(fst)", M6, note="stride 2")
    add(G_RED_AX, "M.sum(i2, axis=0)", SQ)
    add(G_RED_MID, "M.sum(fm)", "10^4", threads=T1)
    add(G_RED_MID, "M.var(fm)", "10^4", threads=T1)
    add(G_SCAN, "M.cumulative_sum(f)", M6, quick=True)
    add(G_SCAN, "M.cumulative_prod(g)", M6)
    add(G_SCAN, "M.cumulative_sum(i)", M6)
    add(G_SCAN, "M.cumulative_sum(f2, axis=0)", SQ)
    add(G_SCAN, "M.cumulative_sum(f2, axis=1)", SQ)
    add(G_SCAN, "M.cumulative_sum(c)", M6)
    for op in ["nansum", "nanprod", "nanmin", "nanmax", "nanmean", "nanvar", "nanstd", "nanargmin", "nanargmax"]:
        add(G_NAN, f"M.{op}(fn)", M6, note="NaN 10%", quick=(op == "nanmean"))
    add(G_NAN, "sd.nancumulative_sum(fn)", M6, np="np.nancumsum(fn)", note="NaN 10%")
    add(G_NAN, "sd.nancumulative_prod(fn)", M6, np="np.nancumprod(fn)", note="NaN 10%")
    add(G_NAN, "M.nansum(M.reshape(fn, (1000, 1000)), axis=0)", SQ, note="NaN 10%")
    add(G_COUNT, "M.count_nonzero(b)", M6)
    add(G_COUNT, "M.count_nonzero(fs)", M6)
    add(G_COUNT, "M.count_nonzero(f2, axis=0)", SQ)

    # statistics (the order statistics on fresh inputs)
    add(G_STATS, "M.quantile(fs, q5)", M6, note="확률 5개", fresh=True)
    add(G_STATS, "M.quantile(fs, 0.5)", M6, fresh=True)
    add(G_STATS, "M.median(fs)", M6, quick=True, fresh=True)
    add(G_STATS, "M.median(f2, axis=0)", SQ, fresh=True)
    add(G_STATS, "M.median(f2, axis=1)", SQ, fresh=True)
    add(G_STATS, "M.nanquantile(fn, q5)", M6, note="NaN 10%, 확률 5개", fresh=True)
    add(G_STATS, "M.nanmedian(fn)", M6, note="NaN 10%", fresh=True)
    add(G_STATS, "sd.cov(m1k, correction=1)", SQ, np="np.cov(m1k)", blas=True)
    add(G_STATS, "sd.cov(obs, correction=1)", "(20, 100000)", np="np.cov(obs)", blas=True)
    add(G_STATS, "M.corrcoef(m1k)", SQ, blas=True)
    add(G_STATS, "M.bincount(iv)", M6, note="값 0..999")
    add(G_STATS, "M.bincount(iv, weights=f)", M6, note="값 0..999")
    add(G_STATS, "M.histogram(fs, bins=100)", M6)
    add(G_STATS, "M.histogram(fs, bins=100, range=(-3.0, 3.0))", M6)

    # sorting (fresh inputs; numpy sorts stably, as sd does by default)
    def srt(stmt, size, note="", unstable=False, **kw):
        extra = ((UNSTABLE, stmt.replace("M.", "np.")),) if unstable else ()
        add(G_SORT, stmt, size, np=_stable(stmt), note=note, extra=extra, fresh=True, **kw)

    srt("M.sort(fs)", M6, unstable=True, quick=True)
    srt("M.sort(zh)", M6, note="절반이 ±0")
    srt("M.sort(i)", M6)
    srt("M.sort(ivw)", M6, note="값 ±10^12")
    srt("M.sort(fn)", M6, note="NaN 10%")
    srt("M.sort(f2, axis=1)", SQ)
    srt("M.sort(f2, axis=0)", SQ)
    srt("M.sort(p8, axis=1)", P8, note="길이 8인 줄 125000개")
    add(G_SORT, "sd.sort(fs, descending=True)", M6, np="np.sort(fs, stable=True)[::-1].copy()", fresh=True)
    srt("M.argsort(fs)", M6, unstable=True)
    srt("M.argsort(i)", M6, unstable=True)
    srt("M.argsort(fn)", M6, note="NaN 10%")
    srt("M.argsort(f2, axis=1)", SQ)
    srt("M.argsort(p8, axis=1)", P8, note="길이 8인 줄 125000개")
    srt("M.sort(fm)", "10^4", threads=T1)
    srt("M.argsort(fm)", "10^4", threads=T1)

    # unique values (fresh inputs; numpy's unique_values hashes and does not sort)
    add(
        G_UNIQUE,
        "M.unique_values(iv)",
        M6,
        np="np.unique(iv)",
        note="값 1000가지",
        quick=True,
        fresh=True,
    )
    add(G_UNIQUE, "M.unique_counts(iv)", M6, note="값 1000가지", fresh=True)
    add(G_UNIQUE, "M.unique_inverse(iv)", M6, note="값 1000가지", fresh=True)
    add(G_UNIQUE, "M.unique_all(iv)", M6, note="값 1000가지", fresh=True)
    add(
        G_UNIQUE,
        "M.unique_values(fs)",
        M6,
        np="np.unique(fs)",
        note="모두 다름",
        fresh=True,
    )

    # searching (searchsorted and nonzero on fresh inputs) and taking
    add(G_SEARCH, "M.searchsorted(srt, keys)", "10^5", note="정렬된 10^6에서", fresh=True)
    add(G_SEARCH, "M.searchsorted(srt, kn)", M6, note="정렬된 10^6에서", fresh=True)
    add(G_SEARCH, "M.searchsorted(srt, k1k)", "10^3", note="정렬된 10^6에서", fresh=True, threads=T1)
    add(G_SEARCH, "M.searchsorted(srt, 0.5)", "1", note="스칼라 하나, 정렬된 10^6에서", fresh=True, threads=T1)
    add(G_SEARCH, "M.nonzero(b)", M6, note="절반이 참", fresh=True)
    add(G_SEARCH, "M.nonzero(s)", M6, note="1%가 참", fresh=True)
    add(G_SEARCH, "M.nonzero(f2 > 0.5)", SQ, fresh=True)
    add(G_SEARCH, "M.take(f, idx)", "10^5", note="10^6에서", quick=True)
    add(G_SEARCH, "M.take(f2, rows, axis=0)", SQ, note="행 1000개")
    add(G_SEARCH, "M.take_along_axis(f2, idx2, axis=1)", SQ)

    # indexing (operators; the masks on fresh inputs)
    add(G_INDEX, "f[1000:2000]", M6, note="뷰", threads=T1)
    add(G_INDEX, "f[::2]", M6, note="뷰", threads=T1)
    add(G_INDEX, "f2[5]", SQ, note="행 뷰", threads=T1)
    add(G_INDEX, "f2[:, 5]", SQ, note="열", threads=T1)
    add(G_INDEX, "f2[10, 20]", SQ, note="스칼라", threads=T1)
    add(G_INDEX, "f[idx]", "10^5", note="10^6에서 모으기")
    add(G_INDEX, "f[idxn]", M6, note="모으기")
    add(G_INDEX, "f[b]", M6, note="마스크, 절반이 참", quick=True, fresh=True)
    add(G_INDEX, "f[s]", M6, note="마스크, 1%가 참", fresh=True)
    add(G_INDEX, "f2[rows]", SQ, note="행 1000개")
    add(G_INDEX, "f2[f2 > 0.5]", SQ, note="2-D 마스크", fresh=True)
    add(G_INDEX, "s3[1]", "3", note="스칼라", threads=T1)
    add(G_ASSIGN, "gz[idx] = 1.0", "10^5", note="10^6에 흩뿌리기", threads=T1)
    add(G_ASSIGN, "gz[idxn] = 1.0", M6, note="흩뿌리기", threads=T1)
    add(G_ASSIGN, "fw[b] = 0.0", M6, note="마스크, 절반이 참", quick=True, fresh=True)
    add(G_ASSIGN, "fw[s] = 0.0", M6, note="마스크, 1%가 참", fresh=True)
    add(G_ASSIGN, "fw[:] = 1.0", M6, note="채우기")
    add(G_ASSIGN, "fw[:] = g", M6, note="복사")
    add(G_ASSIGN, "fw[::2] = 1.0", M6, note="stride 2 채우기")
    # x += g would make x a local name of the compiled function: call the method
    add(G_ASSIGN, "x.__iadd__(g)", M6, name="x += g", note="제자리")
    add(G_ASSIGN, "x[1:] += x[:-1]", M6, note="겹침")
    add(G_ASSIGN, "x.__imul__(1.0000001)", M6, name="x *= 1.0000001", note="제자리")

    # linear algebra in the array area
    add(G_MATMUL, "M.matmul(m512, m512)", "(512, 512)", blas=True)
    add(G_MATMUL, "M.matmul(m1k, m1k)", SQ, quick=True, blas=True)
    add(G_MATMUL, "M.matmul(f2, v)", SQ, note="gemv", blas=True)
    add(G_MATMUL, "M.matmul(v, v)", "10^3", note="dot", blas=True)
    add(G_MATMUL, "M.matmul(bm, bm)", "10^3 x (8, 8)", note="8×8 1000개", blas=True)
    add(G_MATMUL, "M.matmul(im, im)", "(200, 200)", note="정수")
    add(G_MATMUL, "M.matmul(ft, f2)", SQ, note="전치", blas=True)
    add(G_MATMUL, "M.vecdot(f2, g2)", SQ)
    add(G_MATMUL, "M.tensordot(t3, t3, axes=2)", T3, blas=True)

    # the same operations across sizes (cache-resident to memory-bound; 10^7 is 80 MB),
    # a, b (float64 in [0.5, 1.5)) and i, j (int64 in [-1000, 1000)) standing for that
    # size's arrays. 2^14 and 2^16 sit around the elementwise threshold (2^15), 2^18
    # past the single-lane parallel sort (2^17).
    near = ("p14", "p16")
    past = ("p18",)
    for s, np_stmt, more, fresh, extra in [
        ("a + b", None, near, False, ()),
        ("a * 2.0", None, (), False, ()),
        ("M.exp(a)", None, near, False, ()),
        ("M.sin(a)", None, (), False, ()),
        ("M.sum(a)", None, near, False, ()),
        ("M.max(a)", None, (), False, ()),
        ("M.var(a)", None, (), False, ()),
        ("M.cumulative_sum(a)", None, near, False, ()),
        ("a < b", None, (), False, ()),
        ("a[a > 1.0]", None, near, True, ()),
        ("M.where(a > 1.0, a, b)", None, near, False, ()),
        ("M.sort(a)", _stable("M.sort(a)"), past, True, ()),
        ("M.argsort(a)", _stable("M.argsort(a)"), past, True, ()),
        ("M.astype(a, M.int64)", None, (), False, ()),
        ("M.asarray(a, copy=True)", None, (), False, ()),
        ("i + j", None, (), False, ()),
        ("M.sum(i)", None, (), False, ()),
        ("M.sort(i)", _stable("M.sort(i)"), past, True, ()),
        ("M.median(a)", None, past, True, ()),
        ("M.unique_values(i)", "np.unique(i)", (), True, ()),
    ]:
        nms = sorted(("1e3", "1e5", "1e7", *more), key=lambda nm: SIZES[nm][1])
        for nm in nms:
            alias = {k: k + nm for k in ("a", "b", "i", "j")}
            add(
                G_SIZE,
                s,
                SIZES[nm][0],
                np=np_stmt,
                alias=alias,
                extra=extra,
                fresh=fresh,
                quick=(s, nm) in {("a + b", "1e7"), ("M.exp(a)", "p16"), ("M.sort(a)", "p18")},
                threads=T1 if nm == "1e3" and s != "a + b" else BOTH,
            )

    # per-call overhead (tiny arrays); s3 + t3s also at 10 threads (no pool below the threshold)
    for s, size in [
        ("M.zeros(3)", "3"),
        ("M.empty(3)", "3"),
        ("M.asarray(s3)", "3"),
        ("s3 + t3s", "3"),
        ("s3 * 2.0", "3"),
        ("i3 + 1", "3"),
        ("M.sqrt(s3)", "3"),
        ("M.exp(s3)", "3"),
        ("s3 < t3s", "3"),
        ("M.sum(s3)", "3"),
        ("M.max(s3)", "3"),
        ("M.mean(s3)", "3"),
        ("M.sum(sm, axis=0)", "(3, 4)"),
        ("M.sort(s3)", "3"),
        ("M.reshape(sm, (4, 3))", "(3, 4)"),
        ("sm.T", "(3, 4)"),
        ("sm[1]", "(3, 4)"),
        ("sm[1:, ::2]", "(3, 4)"),
        ("M.concat([s3, t3s])", "3"),
        ("M.where(s3 > 1.5, s3, t3s)", "3"),
        ("M.matmul(sm.T, sm)", "(3, 4)"),
        ("M.result_type(f, i)", ""),
        ("M.can_cast(M.int64, M.float64)", ""),
        ("M.finfo(M.float64)", ""),
        ("s3.shape", "3"),
        ("len(s3)", "3"),
        ("float(s3[0])", "3"),
    ]:
        add(
            G_TINY,
            s,
            size,
            quick=(s == "s3 + t3s"),
            note={"M.asarray(s3)": "이미 배열(복사 없음)", "M.result_type(f, i)": "dtype만 봄"}.get(s, ""),
            threads=BOTH if s == "s3 + t3s" else T1,
        )
    return S


# R's version of a statement: R expression, or (expression, {R variable: numpy-side name or
# name.real / name.imag}, R setup). The variables default to the statement's array names.
_R_UNARY = ("exp", "log", "log2", "log10", "sin", "atan", "sinh", "cosh", "tanh", "asinh")
R_STMTS: dict = {
    **{f"M.{op}(f)": f"{op}(f)" for op in _R_UNARY},
    **{f"M.{op}(u)": f"{op}(u)" for op in ("asin", "acos", "atanh")},
    "M.acosh(fh)": "acosh(fh)",
    "M.exp(fm)": "exp(fm)",
    "M.pow(f, g)": "f^g",
    "M.atan2(f, g)": "atan2(f, g)",
    "M.abs(c)": ("Mod(z)", {"zre": "c.real", "zim": "c.imag"}, "z <- complex(real = zre, imaginary = zim)"),
    "M.sum(f)": "sum(f)",
    "M.prod(f)": "prod(f)",
    "M.mean(f)": "mean(f)",
    "M.sum(f2, axis=0)": "colSums(f2)",
    "M.sum(fm)": "sum(fm)",
    "M.matmul(m512, m512)": "m512 %*% m512",
    "M.matmul(m1k, m1k)": "m1k %*% m1k",
    "M.matmul(f2, v)": "f2 %*% v",
    "M.matmul(v, v)": "v %*% v",
    "M.matmul(ft, f2)": ("crossprod(f2, f2)", {"f2": "f2"}, ""),  # t(f2) %*% f2 without a transposed copy
    "M.exp(a)": "exp(a)",
    "M.sin(a)": "sin(a)",
    "M.sum(a)": "sum(a)",
    "M.exp(s3)": "exp(s3)",
    "M.sum(s3)": "sum(s3)",
    "M.mean(s3)": "mean(s3)",
    "M.sum(sm, axis=0)": "colSums(sm)",
    "M.matmul(sm.T, sm)": ("crossprod(sm, sm)", {"sm": "sm"}, ""),
}


def _r_ref(stmt):
    """The R comparator of an sd statement (None if R_STMTS has none) and its {R variable:
    source} map."""
    spec = R_STMTS.get(stmt)
    if spec is None:
        return None, {}
    expr, rvars, rsetup = (spec, None, "") if isinstance(spec, str) else spec
    if rvars is None:
        rvars = {nm: nm for nm in sorted(_names(stmt) - {"M", "sd", "np"})}
    return RRef("R", expr, data={f"r_{v}": v for v in rvars}, setup=rsetup), rvars


def _with_r(setup, rvars):
    """setup, plus the float64 arrays R reads ("r_<variable>") from the numpy side's first input."""

    def wrapped():
        import numpy as np

        d = setup()
        ns = d["np"][0]
        for var, src in rvars.items():
            name, _, part = src.partition(".")
            v = ns[name]
            d[f"r_{var}"] = np.ascontiguousarray(getattr(v, part) if part else v, dtype=np.float64)
        return d

    return wrapped


def _factory(stmt, lib):
    """Given the data, the timed callable: stmt compiled once per input, in turn."""
    return lambda d: rotate([_compile(stmt, ns) for ns in d[lib]])


# The section title of each group in bench/array/speed.md (the group is the English key in
# the case ids).
TITLES = {
    "creation": "생성",
    "conversion": "변환",
    "reshape": "모양 바꾸기",
    "binary-f64": "이항 f64",
    "binary-bool": "이항 bool",
    "binary-i64": "이항 i64",
    "binary-c128": "이항 c128",
    "binary-mixed": "이항 dtype 섞기·스칼라",
    "binary-layout": "이항 메모리 배치",
    "binary-1e4": "이항 10^4",
    "unary-f64": "단항 f64",
    "unary-c128": "단항 c128",
    "unary-i64": "단항 i64",
    "unary-bool": "단항 bool",
    "where": "골라 넣기",
    "reduce-f64": "리덕션 f64",
    "reduce-i64": "리덕션 i64",
    "reduce-c128": "리덕션 c128",
    "reduce-bool": "리덕션 bool",
    "reduce-axis": "축 리덕션",
    "reduce-1e4": "리덕션 10^4",
    "cumulative": "누적",
    "nan-skipping": "NaN 건너뛰기",
    "counting": "개수 세기",
    "statistics": "통계",
    "sorting": "정렬",
    "unique": "고유값",
    "search-take": "찾기·뽑기",
    "indexing": "인덱싱",
    "assignment": "대입",
    "matmul": "행렬곱",
    "by-size": "크기별",
    "small-arrays": "작은 배열",
}


# The cases whose results the smoke run's check does not compare, and why.
NO_CHECK = {
    "M.result_type(f, i)": "the result is a dtype object",
    "M.finfo(M.float64)": "the result is an finfo object",
}
ASSIGN_NO_CHECK = "an assignment: no result, and the target changes with every call"


def cases() -> list[Case]:
    out = []
    used = set()
    for s in _specs():
        np_stmt = s["np"] if s["np"] is not None else s["stmt"]
        extra_stmts = [e[1] for e in s["extra"]]
        names = _names(s["stmt"], np_stmt, *extra_stmts)
        refs = [Ref("numpy", _factory(np_stmt, "np"), threads="follow" if s["blas"] else "fixed")]
        refs += [Ref(label, _factory(st, "np"), lib="numpy") for label, st in s["extra"]]
        setup = _make_setup(frozenset(names), dict(s["alias"]), s["fresh"])
        rref, rvars = _r_ref(s["stmt"])
        if rref is not None:
            if s["fresh"]:
                raise ValueError(f"{s['stmt']}: R times one input, the case rotates fresh ones")
            refs.append(rref)
            setup = _with_r(setup, rvars)
            used.add(s["stmt"])
        out.append(
            Case(
                area="array",
                group=s["group"],
                name=s["name"],
                size=s["size"],
                n=_n(s["size"]),
                setup=setup,
                sd=_factory(s["stmt"], "sd"),
                refs=tuple(refs),
                threads=s["threads"],
                quick=s["quick"],
                note=s["note"],
                check=Check(skip=ASSIGN_NO_CHECK if s["group"] == G_ASSIGN else NO_CHECK.get(s["stmt"], "")),
            )
        )
    if set(R_STMTS) - used:
        raise ValueError(f"R_STMTS without a case: {sorted(set(R_STMTS) - used)}")
    return out
