"""Golden outputs: the results of the public API at fixed inputs, stored exactly, so a change
that must not change results (a move, a refactor, the formatter) can be checked bit for bit.

    .venv/bin/python tools/golden.py record [AREA ...]      # run the cases, write tests/golden/data/<area>.json
    .venv/bin/python tools/golden.py check  [AREA ...]      # run the cases, list every case that differs
    .venv/bin/python tools/golden.py list   [AREA ...]      # case counts per area (no run)

Options (bench/cli.py has the conventions every tool shares): --match PATTERN (only the cases
whose name matches: a glob, or a substring without wildcards; repeatable; check reports only
those, record replaces only those and drops the matching names that are no longer cases),
--jobs N (worker processes, default 8, at most 10), --out FILE (the golden file of the one area
given, in place of tests/golden/data/<area>.json), --threads 1,3,10, -v (every difference, not the
first few per case), --slow N (print the N slowest cases), --any-nan (check: NaNs that differ
only in sign or payload are equal; such differences are named as such either way).
Environment: GOLDEN_TIMEOUT (seconds a case may take before its worker is stopped and the case
named, default 120), GOLDEN_TRACE=1 (workers print each case name as it starts).

The cases live in tests/golden/cases_<area>.py: ``cases(L)`` yields (name, thunk); thunk() returns
a result or raises (an error is a result too: its class and message are stored). An optional
``encode(x, L, enc)`` gives the exact form of an area's own objects (enc encodes the parts).
L is tests/golden/_lib.py's context: L.sd, L.np, L.rng(seed), L.arr(numpy array). Optional
module constants: EXCLUDED = {public name: reason} for a name of the area's __all__ no case
calls, THREAD_DEPENDENT = {case name: reason} for a case whose result may depend on the thread
count (tests/golden/test_golden.py checks both lists).

Every area runs once per thread count, each in its own process (PYTHONHASHSEED=0, GOLDEN_TMP
the run's temporary folder, removed at the end), and the results must agree bit for bit across
thread counts: record stores no result that depends on the thread count and exits 1 naming the
case, unless the case is in THREAD_DEPENDENT (it is then stored per thread count).
Results are encoded exactly, and compared as normalized JSON text, so a type is part of the
result (True is not 1, a numpy scalar is not a Python one):
  bool, int, str and None as themselves; float -> float.hex (NaN keeps its bits); complex -> two
  of those; numpy scalars -> their dtype and value; other subclasses of the built-in types ->
  their class and value; exceptions and warnings -> class (module.qualname) and message;
  arrays with at most SMALL elements -> every element; larger arrays -> dtype, shape, strides,
  the SHA-256 of their bytes, a hash per chunk of elements (so a difference is placed and
  counted) and a fixed sample of elements; objects of stardust classes -> every public
  attribute and every method callable without arguments.
Exit status (bench/cli.py): 0 nothing differs; 1 a difference, a missing or old-format golden
file, or (record) a thread-dependent result; 2 a usage error or a failed worker.

The terminal shows a header block, a progress bar over the cases of every area and thread count
(the workers that finish are named above it), a summary block and tables (bench/console.py);
the same lines, time-stamped, go to build/logs/golden/<stamp>.log.
"""

import argparse
import faulthandler
import hashlib
import importlib
import inspect
import json
import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench import areas as AREAS_MOD  # noqa: E402
from bench import cli  # noqa: E402
from bench import console as C  # noqa: E402

GOLDEN = ROOT / "tests" / "golden"
DATA = GOLDEN / "data"
AREAS = AREAS_MOD.AREAS
THREADS = (1, 3, 10)
SMALL = 1024  # arrays up to this many elements are stored element by element
CHUNK = 256  # a larger array's elements are hashed in chunks of at least this many ...
MAX_CHUNKS = 256  # ... and at most this many chunks per array (longer chunks beyond 65,536 elements)
CHUNK_HEX = 8  # hex digits kept of each chunk's SHA-256
CASE_TIMEOUT = 120.0  # seconds; GOLDEN_TIMEOUT overrides
FORMAT = 2

# --------------------------------------------------------------------------------------------
# encoding


def fhex(v):
    """A float as exact text: float.hex, with the bits of a NaN that is not the default one."""
    if v != v:
        bits = struct.unpack("<Q", struct.pack("<d", v))[0]
        return "nan" if bits == 0x7FF8000000000000 else f"nan:{bits:016x}"
    return float.hex(v)


def unhex(s):
    if s.startswith("nan:"):
        return struct.unpack("<d", struct.pack("<Q", int(s[4:], 16)))[0]
    return float.fromhex(s)


def qualname(t):
    """A class as module.qualname: builtins.ValueError, stardust.linalg.LinAlgError."""
    return f"{t.__module__}.{t.__qualname__}"


def canon(e):
    """The normalized JSON text of an encoding: what record and check compare."""
    return json.dumps(e, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def same(a, b):
    return a is b or canon(a) == canon(b)


BUILTIN_BASES = (bool, int, float, complex, str, bytes, list, dict, set, frozenset)


class Encoder:
    """enc(x): a JSON-able form of a result, exact to the bit and to the type."""

    def __init__(self, L, area_encode=None):
        self.L, self.sd, self.np = L, L.sd, L.np
        self.area_encode = area_encode
        self.stack = []

    def __call__(self, x):
        return self.enc(x)

    def enc(self, x):
        np, sd = self.np, self.sd
        if isinstance(x, BaseException):
            return {"error": qualname(type(x)), "msg": str(x)}
        if self.area_encode is not None:
            e = self.area_encode(x, self.L, self)
            if e is not None:
                return e
        t = type(x)
        if x is None or t is bool or t is str or t is int:
            return x
        if t is float:
            return {"f": fhex(x)}
        if t is complex:
            return {"c": [fhex(x.real), fhex(x.imag)]}
        if isinstance(x, sd.ndarray):
            return self.array(np.asarray(x), "nd", str(x.dtype), list(x.shape), list(x.strides))
        if isinstance(x, np.ndarray):
            return self.array(x, "np", x.dtype.str, list(x.shape), list(x.strides))
        if isinstance(x, np.generic):  # before the built-in types: np.float64 is a float
            return {"npscalar": x.dtype.str, "v": self.enc(x.item())}
        if isinstance(x, type(sd.float64)):
            return {"dtype": str(x)}
        if isinstance(x, tuple):  # tuple, named tuple, structseq: the class is part of the result
            out = {"tuple": qualname(t), "items": [self.enc(v) for v in x]}
            if hasattr(t, "_fields"):
                out["fields"] = list(t._fields)
            return out
        if t is list:
            return [self.enc(v) for v in x]
        if t is dict:
            return {"dict": [[self.enc(k), self.enc(v)] for k, v in x.items()]}
        if t is set or t is frozenset:
            items = [self.enc(v) for v in x]
            return {t.__name__: sorted(items, key=canon)}
        if t is bytes or t is bytearray:
            return {"bytes": bytes(x).hex()}
        base = next((b for b in BUILTIN_BASES if isinstance(x, b)), None)
        if base is not None:  # IntEnum, a str or dict subclass: the class and the plain value
            return {"subclass": qualname(t), "v": self.enc(base(x))}
        if isinstance(x, (range, slice)) or x is Ellipsis:
            return {"repr": repr(x)}
        if isinstance(x, sd.random.Generator):
            return {"generator": self.enc(x.state)}
        if isinstance(x, type) or inspect.isroutine(x):
            return {"callable": f"{getattr(x, '__module__', '?')}.{getattr(x, '__qualname__', repr(x))}"}
        if t.__module__.startswith("stardust"):
            return self.obj(x)
        raise TypeError(f"golden: no encoding for {qualname(t)}")

    def array(self, a, tag, dtype, shape, strides):
        np = self.np
        out = {tag: dtype, "shape": shape, "strides": strides}
        flat = a.reshape(-1) if a.flags.c_contiguous else np.ascontiguousarray(a).reshape(-1)
        n = flat.size
        kind = a.dtype.kind
        if kind in "fc" or kind in "iub":
            elem = self.elem_fn(kind)
            if n <= SMALL:
                out["v"] = [elem(v) for v in flat.tolist()]
            else:
                raw = np.ascontiguousarray(flat).tobytes()
                out["n"] = n
                out["sha"] = hashlib.sha256(raw).hexdigest()[:32]
                if kind in "fc":  # the sha with every NaN made the default one, when that differs
                    canon_ = np.where(np.isnan(flat), np.nan, flat) if kind == "f" else canon_complex(np, flat)
                    shan = hashlib.sha256(np.ascontiguousarray(canon_).tobytes()).hexdigest()[:32]
                    if shan != out["sha"]:
                        out["shan"] = shan
                step = chunk_len(n)
                size = flat.dtype.itemsize
                out["chunk"] = step
                out["chunks"] = "".join(
                    hashlib.sha256(raw[i * size : (i + step) * size]).hexdigest()[:CHUNK_HEX] for i in range(0, n, step)
                )
                out["sample"] = [[i, elem(flat[i].item())] for i in sample_indices(n)]
        else:  # object, string: element by element through enc
            vals = [self.enc(v) for v in flat.tolist()]
            if n <= SMALL:
                out["v"] = vals
            else:
                out["n"] = n
                out["sha"] = hashlib.sha256(canon(vals).encode()).hexdigest()[:32]
                step = chunk_len(n)
                out["chunk"] = step
                out["chunks"] = "".join(
                    hashlib.sha256(canon(vals[i : i + step]).encode()).hexdigest()[:CHUNK_HEX]
                    for i in range(0, n, step)
                )
                out["sample"] = [[i, vals[i]] for i in sample_indices(n)]
        return out

    @staticmethod
    def elem_fn(kind):
        if kind == "f":
            return fhex
        if kind == "c":
            return lambda z: [fhex(z.real), fhex(z.imag)]
        return lambda v: v  # int, bool

    # Objects of stardust classes: every public attribute, and every public method that takes no
    # arguments (all parameters have defaults), called. Exceptions are results, as elsewhere.
    MAX_DEPTH = 6

    def obj(self, x):
        if any(x is s for s in self.stack) or len(self.stack) >= self.MAX_DEPTH:
            return {"ref": type(x).__qualname__}
        self.stack.append(x)
        try:
            attrs = {}
            names = {n for n in dir(x) if not n.startswith("_")}
            for name in sorted(names):
                try:
                    v = getattr(x, name)
                except Exception as e:  # noqa: BLE001 -- a raising property is a result
                    attrs[name] = self.enc(e)
                    continue
                if inspect.ismethod(v) or inspect.isbuiltin(v) or type(v).__name__ == "nb_bound_method":
                    if callable_without_args(v):
                        try:
                            r = v()
                        except Exception as e:  # noqa: BLE001
                            r = e
                        attrs[name + "()"] = self.enc(r)
                    continue
                attrs[name] = self.enc(v)
            return {"obj": qualname(type(x)), "attrs": attrs}
        finally:
            self.stack.pop()


def chunk_len(n):
    """Elements per hashed chunk of an n-element array: CHUNK, longer when that would make more
    than MAX_CHUNKS chunks."""
    return max(CHUNK, -(-n // MAX_CHUNKS))


def canon_complex(np, z):
    out = np.empty_like(z)  # no arithmetic: inf * 1j would raise numpy's warning inside the case
    out.real = np.where(np.isnan(z.real), np.nan, z.real)
    out.imag = np.where(np.isnan(z.imag), np.nan, z.imag)
    return out


def any_nan(e):
    """An encoding with NaN bits forgotten: every NaN the default one (check --any-nan). A large
    array keeps its NaN-blind sha and sample; its chunk hashes, which see NaN bits, are dropped."""
    if isinstance(e, str):
        return "nan" if e.startswith("nan:") else e
    if isinstance(e, list):
        return [any_nan(v) for v in e]
    if isinstance(e, dict):
        out = {k: any_nan(v) for k, v in e.items() if k not in ("shan", "chunks")}
        if "shan" in e:
            out["sha"] = e["shan"]
        return out
    return e


def callable_without_args(f):
    try:
        sig = inspect.signature(f)
    except TypeError, ValueError:
        return False
    return all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in sig.parameters.values())


def sample_indices(n):
    """A fixed sample of a large array's elements: both ends and 15 evenly spaced inside."""
    return sorted({0, 1, 2, 3, n - 4, n - 3, n - 2, n - 1} | {n * k // 16 for k in range(1, 16)})


# --------------------------------------------------------------------------------------------
# worker: one area at one thread count, in its own process


def load_area(area):
    for p in (str(ROOT), str(GOLDEN)):
        if p not in sys.path:
            sys.path.insert(0, p)
    lib = importlib.import_module("_lib")
    mod = importlib.import_module(f"cases_{area}")
    return lib.Lib(), mod


def case_names(mod, L, area):
    """The case names and thunks of an area, in order; SystemExit on a duplicate name."""
    todo, seen = [], set()
    for name, thunk in mod.cases(L):
        if name in seen:
            raise SystemExit(f"golden: {area}: duplicate case name {name!r}")
        seen.add(name)
        todo.append((name, thunk))
    return todo


def run_worker(area, threads, out, patterns):
    L, mod = load_area(area)
    L.sd.set_num_threads(threads)
    E = Encoder(L, getattr(mod, "encode", None))
    todo = [(name, thunk) for name, thunk in case_names(mod, L, area) if cli.matches(patterns, name)]
    res, times = {}, {}
    trace = os.environ.get("GOLDEN_TRACE")  # print each case name before it runs
    # A case that hangs (even holding the GIL) ends the worker after CASE_TIMEOUT seconds; the
    # driver then names it from the .cur file, which always holds the running case's name.
    timeout = float(os.environ.get("GOLDEN_TIMEOUT", CASE_TIMEOUT))
    cur = open(out + ".cur", "w")
    for i, (name, thunk) in enumerate(todo):
        if trace:
            print(f"{area}/{name}", file=sys.stderr, flush=True)
        cur.seek(0)
        cur.write(f"{i}\t{name}")  # the cases done before it, and its name
        cur.truncate()
        cur.flush()
        faulthandler.dump_traceback_later(timeout, exit=True)
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                r = thunk()
            except Exception as e:  # noqa: BLE001 -- errors are results too
                r = e
            try:
                rec = {"r": E.enc(r)}
            except Exception:  # noqa: BLE001
                msg = f"golden: {area}: case {name!r}: cannot encode its result\n{traceback.format_exc()}"
                raise SystemExit(msg) from None
        if caught:
            rec["w"] = [[qualname(w.category), str(w.message)] for w in caught]
        faulthandler.cancel_dump_traceback_later()
        times[name] = time.perf_counter() - t0
        res[name] = rec
        del r
    cur.close()
    os.unlink(out + ".cur")
    pathlib.Path(out).write_text(json.dumps({"cases": res, "times": times}, separators=(",", ":")))


# --------------------------------------------------------------------------------------------
# driver


def count_cases(areas, patterns):
    """{area: the number of cases a worker will run} (the cases are made, not run: about a second)."""
    out = {}
    for area in areas:
        L, mod = load_area(area)
        out[area] = sum(1 for name, _ in mod.cases(L) if cli.matches(patterns, name))
    return out


def run_all(areas, threads, jobs, patterns, con, counts):
    """{area: {threads: {name: record}}}, {area: {name: seconds}}, seconds, [failures] -- the
    workers in parallel, a bar over every case at every thread count."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="sd-golden-"))
    try:
        return _run_all(tmp, areas, threads, jobs, patterns, con, counts)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_all(tmp, areas, threads, jobs, patterns, con, counts):
    env = dict(os.environ, PYTHONHASHSEED="0", GOLDEN_TMP=str(tmp))
    # Big areas first so the slowest workers start early.
    order = {"frame": 0, "array": 1, "regression": 2, "stats": 3, "special": 4}
    tasks = sorted(((a, n) for a in areas for n in threads), key=lambda t: (order.get(t[0], 9), t[1]))
    total = sum(counts[a] for a, _ in tasks)
    running, results, times, failed = [], {}, {}, []
    finished, last_tenth, last_drawn, seen = 0, -1, 0.0, {}
    t0 = time.perf_counter()
    match = [a for p in patterns for a in ("--match", p)]
    con.start(total, "golden")
    while tasks or running:
        while tasks and len(running) < jobs:
            a, n = tasks.pop(0)
            out = tmp / f"{a}-{n}.json"
            cmd = [sys.executable, __file__, "_worker", a, str(n), str(out), *match]
            err = open(tmp / f"{a}-{n}.err", "w+")
            p = subprocess.Popen(cmd, env=env, cwd=ROOT, stderr=err, text=True)
            running.append((a, n, out, p, err, time.perf_counter()))
        time.sleep(0.05)
        for item in list(running):
            a, n, out, p, errf, started = item
            if p.poll() is None:
                continue
            running.remove(item)
            errf.seek(0)
            err = errf.read()
            errf.close()
            pathlib.Path(errf.name).unlink()
            secs = time.perf_counter() - started
            finished += counts[a]
            if p.returncode != 0:
                cur = pathlib.Path(str(out) + ".cur")
                where = f"in case {cur_case(cur)[1]!r}" if cur.exists() else ""
                cur.unlink(missing_ok=True)
                failed.append((f"{a} at {n} thread{'s' if n > 1 else ''}", f"exit {p.returncode} {where}".strip(), err))
                con.event(
                    con.color("failed", "31"), f"{a} · {n} thread{'s' if n > 1 else ''}", f"exit {p.returncode} {where}"
                )
                continue
            data = json.loads(out.read_text())
            out.unlink()
            results.setdefault(a, {})[n] = data["cases"]
            for k, v in data["times"].items():
                times.setdefault(a, {})[k] = max(times.get(a, {}).get(k, 0.0), v)
            con.event("done", f"{a} · {n} thread{'s' if n > 1 else ''}", f"{counts[a]:,} cases  {C.fmt_duration(secs)}")
        for r in running:  # a worker removes its .cur file as it ends: keep the last count seen
            seen[r[:2]] = max(seen.get(r[:2], 0), cur_case(pathlib.Path(str(r[2]) + ".cur"))[0])
        done = finished + sum(seen[r[:2]] for r in running)
        el = time.perf_counter() - t0
        tenth = done * 10 // max(total, 1)
        plain = f"  golden: {done:,}/{total:,} cases  ({C.fmt_duration(el)} elapsed)" if tenth != last_tenth else ""
        last_tenth = tenth
        if plain or el - last_drawn >= 0.2 or not (tasks or running):
            now = ", ".join(f"{r[0]}·{r[1]}" for r in running)
            what = f"running {now}" if len(now) <= 36 else f"running {now[:35]}…"
            left = el / done * (total - done) if done else None
            con.step(done, "golden", what, el, left, plain if not con.tty else "")
            last_drawn = el
    con.close()
    return results, times, time.perf_counter() - t0, failed


def cur_case(path):
    """(cases done, running case) from a worker's .cur file; (0, "") before it has one."""
    try:
        k, _, name = path.read_text().partition("\t")
        return int(k), name
    except OSError, ValueError:
        return 0, ""


def merge_threads(per, allowed=()):
    """{name: record, or {"per_threads": {n: record}} for a name in `allowed`}, [names whose
    record depends on the thread count and is not allowed to (they are left out)]. Records are
    compared as normalized JSON text."""
    names = list(next(iter(per.values())).keys())
    out, unstable = {}, []
    for name in names:
        recs = {n: per[n].get(name) for n in per}
        texts = {canon(v) for v in recs.values()}
        if len(texts) == 1:
            out[name] = next(iter(recs.values()))
        elif name in allowed:
            out[name] = {"per_threads": {str(n): v for n, v in recs.items()}}
        else:
            unstable.append(name)
    return out, unstable


# --------------------------------------------------------------------------------------------
# differences, in words


def show_float(s):
    try:
        v = unhex(s)
    except ValueError, OverflowError:
        return s
    return f"{s} ({v!r})"


def ulps(a, b):
    """The ULP distance (bench.accuracy.common.ulp_distance: the doubles between them) of two
    finite floats written by fhex, else None."""
    try:
        x, y = unhex(a), unhex(b)
    except ValueError, OverflowError:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    from bench.accuracy.common import ulp_distance

    return int(ulp_distance(x, y))


def show_elem(e):
    if isinstance(e, str):
        return show_float(e)
    if isinstance(e, list) and len(e) == 2 and all(isinstance(t, str) for t in e):
        return f"{show_float(e[0])} + {show_float(e[1])}j"
    s = canon(e)
    return s if len(s) <= 160 else s[:157] + "..."


def elem_ulps(g, w):
    if isinstance(g, str) and isinstance(w, str):
        u = ulps(g, w)
        return f", {u} ulp" if u else ""  # none for +0 against -0
    if isinstance(g, list) and isinstance(w, list) and len(g) == len(w) == 2:
        us = [ulps(a, b) for a, b in zip(g, w)]
        if all(u is not None for u in us) and any(us):
            return f", {us[0]} / {us[1]} ulp"
    return ""


def unravel(i, shape):
    if not shape:
        return "()"
    idx = []
    for d in reversed(shape):
        idx.append(i % d if d else 0)
        i = i // d if d else 0
    return "(" + ", ".join(str(t) for t in reversed(idx)) + ")"


TAGS = {"f", "c", "npscalar", "subclass", "error", "tuple", "dict", "set", "frozenset", "bytes", "repr", "generator",
        "callable", "obj", "dtype", "nd", "np", "ref", "per_threads"}  # fmt: skip


def kind(e):
    """The tags of an encoded dict: what sort of value it encodes (an area's own encodings too)."""
    tags = TAGS & e.keys()
    return tags or frozenset(e.keys())


def diff(got, want, path, out):
    """Append (path, text) for every difference between two encodings (types included: true is
    not 1)."""
    if same(got, want):
        return
    if isinstance(got, dict) and isinstance(want, dict) and kind(got) != kind(want):
        out.append((path, f"got {short(got)}, want {short(want)}"))  # a different type: the whole value
        return
    if isinstance(got, dict) and isinstance(want, dict):
        tag = next((t for t in ("nd", "np") if t in got and t in want), None)
        if tag is not None:
            return diff_array(got, want, path, out, tag)
        if "f" in got and "f" in want and len(got) == len(want) == 1:
            if any_nan(got["f"]) == any_nan(want["f"]):
                out.append((path, f"only NaN bits differ: got {got['f']}, want {want['f']}"))
                return
            out.append(
                (path, f"got {show_float(got['f'])}, want {show_float(want['f'])}{elem_ulps(got['f'], want['f'])}")
            )
            return
        if "error" in got or "error" in want:
            out.append((path, f"got {short(got)}, want {short(want)}"))
            return
        for k in list(want) + [k for k in got if k not in want]:
            if k not in got:
                out.append((f"{path}.{k}", "missing now"))
            elif k not in want:
                out.append((f"{path}.{k}", f"new: {short(got[k])}"))
            else:
                diff(got[k], want[k], f"{path}.{k}", out)
        return
    if isinstance(got, list) and isinstance(want, list):
        if len(got) != len(want):
            out.append((path, f"length {len(got)}, want {len(want)}"))
            return
        for i, (g, w) in enumerate(zip(got, want)):
            diff(g, w, f"{path}[{i}]", out)
        return
    out.append((path, f"got {short(got)}, want {short(want)}"))


def diff_array(got, want, path, out, tag):
    for k in (tag, "shape", "strides"):
        if not same(got.get(k), want.get(k)):
            out.append((path, f"{'dtype' if k == tag else k} {got.get(k)}, want {want.get(k)}"))
    if got.get("shape") != want.get("shape"):
        return
    shape = want["shape"]
    if "v" in got and "v" in want:
        bad = [i for i, (g, w) in enumerate(zip(got["v"], want["v"])) if not same(g, w)]
        if bad:
            first = bad[0]
            g, w = got["v"][first], want["v"][first]
            if all(same(any_nan(got["v"][i]), any_nan(want["v"][i])) for i in bad):
                n_all = len(want["v"])
                text = f"only NaN bits differ (sign or payload), in {len(bad)} of {n_all} elements"
                out.append((path, f"{text}; first at {unravel(first, shape)}: got {show_elem(g)}, want {show_elem(w)}"))
                return
            out.append(
                (
                    path,
                    f"{len(bad)} of {len(want['v'])} elements differ; first at {unravel(first, shape)}: "
                    f"got {show_elem(g)}, want {show_elem(w)}{elem_ulps(g, w)}",
                )
            )
            for i in bad[1:3]:
                g, w = got["v"][i], want["v"][i]
                out.append(
                    (path, f"  also at {unravel(i, shape)}: got {show_elem(g)}, want {show_elem(w)}{elem_ulps(g, w)}")
                )
    elif "sha" in got and "sha" in want:
        if got["sha"] != want["sha"]:
            n = want["n"]
            if got.get("shan", got["sha"]) == want.get("shan", want["sha"]):
                out.append((path, f"only NaN bits differ (sign or payload; {n} elements, sha)"))
                return
            out.append((path, chunk_text(got, want, n, shape)))
            gs, ws = dict(map(tuple, got["sample"])), dict(map(tuple, want["sample"]))
            for i in [i for i in ws if not same(gs.get(i), ws[i])][:3]:
                g, w = gs.get(i), ws[i]
                out.append(
                    (
                        path,
                        f"  sampled at {unravel(i, shape)}: got {show_elem(g)}, want {show_elem(w)}{elem_ulps(g, w)}",
                    )
                )
    else:
        out.append((path, "stored differently (size class changed)"))


def chunk_text(got, want, n, shape):
    """Where a large array differs: the chunks whose hashes differ, counted, and the first one's
    element range."""
    step = want.get("chunk")
    if not step or got.get("chunk") != step or "chunks" not in got or "chunks" not in want:
        return f"bytes differ ({n} elements, sha)"
    gc, wc = got["chunks"], want["chunks"]
    m = len(wc) // CHUNK_HEX
    bad = [k for k in range(m) if gc[k * CHUNK_HEX : (k + 1) * CHUNK_HEX] != wc[k * CHUNK_HEX : (k + 1) * CHUNK_HEX]]
    if not bad:
        return f"bytes differ ({n} elements, sha), no chunk hash does"
    lo, hi = bad[0] * step, min(n, (bad[0] + 1) * step) - 1
    return (
        f"{len(bad)} of {m} chunks of {step} elements differ ({n} elements); the first is elements "
        f"{lo}-{hi}, {unravel(lo, shape)} to {unravel(hi, shape)}"
    )


def short(e):
    s = canon(e)
    return s if len(s) <= 200 else s[:197] + "..."


# --------------------------------------------------------------------------------------------
# commands


def golden_path(area, out=None):
    return pathlib.Path(out) if out else DATA / f"{area}.json"


def shown_path(path):
    path = pathlib.Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def selection(args):
    return " ".join(f"--match {p}" for p in args.match) if args.match else "every case"


def header(con, what, args, areas, threads, counts, log):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    total = sum(counts.values()) * len(threads)
    con.lines(
        C.block(
            f"stardust golden {what}",
            [
                ("Cases", f"{sum(counts.values()):,}"),
                ("Runs", f"{total:,}"),
                ("Threads", ",".join(map(str, threads))),
                ("Workers", str(args.jobs)),
                ("Commit", commit.stdout.strip() or "?"),
            ],
            [
                ("Areas", " ".join(areas)),
                ("Selection", selection(args)),
                ("Golden", shown_path(args.out or DATA)),
                ("Log", shown_path(log)),
            ],
        )  # fmt: skip
    )


def worker_failures(con, failed):
    """The failed workers, with the end of each one's error output; exit status 2."""
    con.print(con.color(f"{len(failed)} worker{'s' if len(failed) > 1 else ''} failed:", "31"))
    for what, why, err in failed:
        con.print(f"  {what}: {why}")
        con.lines([f"    {line.replace(str(ROOT) + '/', '')}" for line in err.rstrip().splitlines()[-15:]])
    con.lines(C.notes(["GOLDEN_TRACE=1 prints each case name as a worker starts it; GOLDEN_TIMEOUT sets\n"
                       f"the seconds a case may take (default {CASE_TIMEOUT:.0f})."]))  # fmt: skip
    return cli.ERROR


def read_golden(path):
    """(the file's document, None) or (None, why it cannot be compared against)."""
    if not path.exists():
        return None, "no golden file"
    doc = json.loads(path.read_text())
    if doc.get("format") != FORMAT:
        return None, "old format"
    return doc, None


def cmd_record(args, areas, threads, con, log):
    if args.match:  # a partial record keeps the rest of a file it can read
        for area in areas:
            path = golden_path(area, args.out)
            old, why = read_golden(path)
            if old is None and path.exists():
                con.print(con.color(f"{shown_path(path)}: {why}; record the whole area first (no --match)", "31"))
                return cli.ERROR
    counts = count_cases(areas, args.match)
    header(con, "record", args, areas, threads, counts, log)
    results, times, wall, failed = run_all(areas, threads, args.jobs, args.match, con, counts)
    if failed:
        return worker_failures(con, failed)
    rows, unstable_all, allowed_all, total = [], [], [], 0
    for area in areas:
        L, mod = load_area(area)
        allowed = getattr(mod, "THREAD_DEPENDENT", {})
        merged, unstable = merge_threads(results[area], allowed)
        path = golden_path(area, args.out)
        old, _ = read_golden(path)
        rec_threads = list(threads)
        if args.match and old is not None:  # replace the matching cases, drop matching names now gone
            names = {name for name, _ in case_names(mod, L, area)}
            kept = {k: v for k, v in old["cases"].items() if not cli.matches(args.match, k) or k in names}
            merged = {**kept, **merged}
            rec_threads = sorted(set(old["threads"]) | set(threads))
        for name in unstable:  # a thread-dependent result is not stored: the old record (if any) stays
            if old is not None and name in old["cases"]:
                merged[name] = old["cases"][name]
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"format": FORMAT, "area": area, "threads": rec_threads, "cases": merged}
        path.write_text(json.dumps(doc, indent=0, sort_keys=True, ensure_ascii=False) + "\n")
        nerr = sum(1 for v in merged.values() if isinstance(v.get("r"), dict) and "error" in v["r"])
        nw = sum(1 for v in merged.values() if "w" in v)
        size = path.stat().st_size / 1e6
        rows.append([area, f"{len(merged):,}", f"{nerr:,}", f"{nw:,}", f"{size:.1f} MB", shown_path(path)])
        total += len(merged)
        unstable_all += [f"{area}/{name}" for name in unstable]
        allowed_all += [f"{area}/{name}" for name, v in merged.items() if "per_threads" in v]
    status = cli.FAIL if unstable_all else cli.OK
    con.lines(
        [""]
        + C.block(
            "Golden record summary",
            [
                ("Cases", f"{total:,}"),
                ("Areas", str(len(areas))),
                ("Threads", ",".join(map(str, threads))),
                ("Thread-dependent", f"{len(unstable_all):,}"),
                ("Time", C.fmt_duration(wall)),
                ("Partial", "yes" if args.match else "no"),
                ("Exit status", str(status)),
            ],
        )  # fmt: skip
        + C.table(["Area", "Cases", "Errors", "Warnings", "Size", "File"], rows, "lrrrrl")
    )
    if unstable_all:
        con.print("")
        con.print(
            con.color("Results that depend on the thread count (not stored; a bug, DESIGN.md §1.6 rule 1):", "31")
        )
        con.lines([f"  {name}" for name in unstable_all])
    if allowed_all:
        con.print("")
        con.print("Stored per thread count (THREAD_DEPENDENT of the case file):")
        con.lines([f"  {name}" for name in allowed_all])
    slow_table(con, times, args.slow)
    notes = [
        "Errors and warnings are results: a case that raises stores the error's class and message.",
        "Check the record against the library again with: .venv/bin/python tools/golden.py check",
        f"The log: {shown_path(log)}",
    ]
    if unstable_all:
        notes.insert(0, "A result that depends on the thread count breaks DESIGN.md §1.6 rule 1; a case whose result\n"
                        "may rightly do so goes into THREAD_DEPENDENT of its case file, with the reason.")  # fmt: skip
    con.lines([""] + C.notes(notes))
    return status


def cmd_check(args, areas, threads, con, log):
    counts = count_cases(areas, args.match)
    header(con, "check", args, areas, threads, counts, log)
    results, times, wall, failed = run_all(areas, threads, args.jobs, args.match, con, counts)
    if failed:
        return worker_failures(con, failed)
    rows, lines, totals = [], [], {"cases": 0, "changed": 0, "new": 0, "gone": 0, "missing": 0}
    for area in areas:
        per = results[area]
        names = list(next(iter(per.values())).keys())
        totals["cases"] += len(names)
        doc, why = read_golden(golden_path(area, args.out))
        if doc is None:
            rows.append([area, f"{len(names):,}", "", "", "", why])
            totals["missing"] += 1
            continue
        golden = {k: v for k, v in doc["cases"].items() if cli.matches(args.match, k)}
        changed, new, gone = 0, 0, sorted(set(golden) - set(names))
        for name in names:
            if name not in golden:
                new += 1
                lines.append(f"{con.color('NEW', '33')}      {area}/{name}: not in the golden file")
                continue
            want_all = golden[name]
            groups = {}  # identical differences across thread counts are reported once
            for n in threads:
                got = per[n][name]
                if "per_threads" in want_all:
                    want = want_all["per_threads"].get(str(n))
                    if want is None:
                        groups.setdefault("not recorded", ([("", f"not recorded at {n} threads")], []))[1].append(n)
                        continue
                else:
                    want = want_all
                if same(got, want) or (args.any_nan and same(any_nan(got), any_nan(want))):
                    continue
                d = []
                diff(got, want, "", d)
                groups.setdefault(canon(d), (d, []))[1].append(n)
            if not groups:
                continue
            changed += 1
            for d, ns in groups.values():
                lines.append(f"{con.color('CHANGED', '31')}  {area}/{name} [threads {','.join(map(str, ns))}]")
                shown = d if args.verbose else d[:4]
                for p, text in shown:
                    lines.append(f"           {p or '.'}: {text}")
                if len(d) > len(shown):
                    lines.append(f"           ... {len(d) - len(shown)} more differences (-v shows all)")
        for name in gone:
            lines.append(f"{con.color('GONE', '33')}     {area}/{name}: in the golden file, no longer a case")
        totals["changed"] += changed
        totals["new"] += new
        totals["gone"] += len(gone)
        ok = not (changed or new or gone)
        status = "equal" if ok else "differs"
        rows.append([area, f"{len(names):,}", f"{changed:,}", f"{new:,}", f"{len(gone):,}", status])
    nbad = totals["changed"] + totals["new"] + totals["gone"] + totals["missing"]
    verdict = "all equal" if not nbad else f"{nbad:,} differ"
    con.lines(
        [""]
        + C.block(
            "Golden check summary",
            [
                ("Cases", f"{totals['cases']:,}"),
                ("Result", verdict),
                ("Changed", f"{totals['changed']:,}"),
                ("New", f"{totals['new']:,}"),
                ("Gone", f"{totals['gone']:,}"),
                ("Not comparable", str(totals["missing"])),
                ("Threads", ",".join(map(str, threads))),
                ("Time", C.fmt_duration(wall)),
                ("NaN bits", "ignored" if args.any_nan else "compared"),
                ("Exit status", str(cli.FAIL if nbad else cli.OK)),
            ],
        )  # fmt: skip
        + status_colors(con, C.table(["Area", "Cases", "Changed", "New", "Gone", "Status"], rows, "lrrrrl"))
    )
    if lines:
        con.print("")
        con.print("Differences:")
        con.lines(lines)
    slow_table(con, times, args.slow)
    notes = []
    if nbad:
        what = " ".join(a for a, r in zip(areas, rows) if r[-1] != "equal")
        notes.append(
            "After a change meant to change results, record those areas again (and commit the files):\n"
            f".venv/bin/python tools/golden.py record {what}"
        )
        if not args.verbose:
            notes.append("-v lists every difference of a case, not the first four.")
        if not args.any_nan:
            notes.append("--any-nan treats NaNs that differ only in sign or payload as equal.")
    notes.append(f"The log: {shown_path(log)}")
    con.lines([""] + C.notes(notes))
    return cli.FAIL if nbad else cli.OK


def status_colors(con, lines):
    """A table's status column colored after layout (the table measures plain text)."""
    out = []
    for line in lines:
        for word, code in (("equal", "32"), ("differs", "31"), ("no golden file", "31"), ("old format", "31")):
            if line.endswith("   " + word):
                line = line[: -len(word)] + con.color(word, code)
                break
        out.append(line)
    return out


def cmd_list(areas, patterns, con):
    rows, total = [], 0
    for area in areas:
        L, mod = load_area(area)
        n = sum(1 for name, _ in mod.cases(L) if cli.matches(patterns, name))
        rows.append([area, f"{n:,}", shown_path(GOLDEN / f"cases_{area}.py")])
        total += n
    con.lines(
        C.block("stardust golden cases", [("Areas", str(len(areas))), ("Cases", f"{total:,}")])
        + C.table(["Area", "Cases", "Defined in"], rows, "lrl")
    )
    return cli.OK


def slow_table(con, times, n):
    if not n:
        return
    flat = sorted(((t, f"{a}/{k}") for a, d in times.items() for k, t in d.items()), reverse=True)[:n]
    con.print("")
    con.print(f"The {len(flat)} slowest cases (the slowest thread count):")
    con.lines(C.table(["Case", "Seconds"], [[k, f"{t:.3f}"] for t, k in flat], "lr"))


def parser():
    ap = argparse.ArgumentParser(prog="tools/golden.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["record", "check", "list"])
    cli.add_areas(ap)
    cli.add_match(ap)
    cli.add_jobs(ap)
    cli.add_out(ap, "the golden file of the one area given (default tests/golden/data/<area>.json)")
    ap.add_argument("--threads", default=",".join(map(str, THREADS)), help="thread counts (default 1,3,10)")
    ap.add_argument("--slow", type=int, default=0, metavar="N", help="print the N slowest cases")
    ap.add_argument("-v", "--verbose", action="store_true", help="every difference of a case, not the first four")
    ap.add_argument("--any-nan", action="store_true", help="check: NaNs that differ only in sign or payload are equal")
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_worker"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("_w")
        ap.add_argument("area")
        ap.add_argument("threads", type=int)
        ap.add_argument("out")
        cli.add_match(ap)
        a = ap.parse_args(argv)
        return run_worker(a.area, a.threads, a.out, a.match)
    ap = parser()
    args = ap.parse_args(argv)
    areas = cli.areas(ap, args)
    try:
        threads = tuple(int(t) for t in args.threads.split(","))
    except ValueError:
        ap.error(f"--threads takes thread counts like 1,3,10, not {args.threads!r}")
    if args.out is not None and len(areas) != 1:
        ap.error("--out names the golden file of one area: give exactly one AREA")
    if args.command == "list":
        return cmd_list(areas, args.match, C.Console(stream=sys.stdout))
    log = cli.log_file("golden")
    with open(log, "w") as fh:
        con = C.Console(log=fh, stream=sys.stdout)
        return (cmd_record if args.command == "record" else cmd_check)(args, areas, threads, con, log)


if __name__ == "__main__":
    sys.exit(main())
