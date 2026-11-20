"""Golden outputs: the results of the public API at fixed inputs, stored exactly, so a change
that must not change results (a move, a refactor, the formatter) can be checked bit for bit.

    .venv/bin/python tools/golden.py record [AREA ...]      # run the cases, write tests/golden/data/AREA.json
    .venv/bin/python tools/golden.py check  [AREA ...]      # run the cases, list every case that differs
    .venv/bin/python tools/golden.py list   [AREA ...]      # case counts per area (no run)

Options: --case REGEX (only matching case names; check reports only those, record updates only
those), --threads 1,3,10, --jobs N (worker processes, default 8, at most 10), -v (every difference, not the
first few per case), --slow N (print the N slowest cases), --out DIR (golden files elsewhere), --any-nan (check: NaNs
that differ only in sign or payload are equal; such differences are named as such either way).
Environment: GOLDEN_TIMEOUT (seconds a case may take before its worker is stopped and the case
named, default 120), GOLDEN_TRACE=1 (workers print each case name as it starts).

The cases live in tests/golden/cases_AREA.py: ``cases(L)`` yields (name, thunk); thunk() returns
a result or raises (an error is a result too: its type and message are stored). An optional
``encode(x, L, enc)`` gives the exact form of an area's own objects (enc encodes the parts).
L is tests/golden/_lib.py's context: L.sd, L.np, L.rng(seed), L.arr(numpy array).

Every area runs once per thread count, each in its own process (PYTHONHASHSEED=0), and the
results must agree across thread counts; a case whose result depends on the thread count is
stored per thread count and reported. Results are encoded exactly:
  float -> float.hex (NaN keeps its bits), arrays with at most SMALL elements -> every element,
  larger arrays -> dtype, shape, strides, the SHA-256 of their bytes and a fixed sample of
  elements, objects of stardust classes -> every public attribute and every method callable
  without arguments, warnings raised by a case -> category and message.
Exit status of check: 0 when nothing differs, 1 otherwise; 2 when a worker failed.

The terminal shows a header block, a progress bar over the cases of every area and thread count
(the workers that finish are named above it), a summary block and tables (bench/console.py);
the same lines, time-stamped, go to build/golden/<stamp>.log.
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
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

import procrun

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench import console as C  # noqa: E402

GOLDEN = ROOT / "tests" / "golden"
DATA = GOLDEN / "data"
AREAS = ("array", "linalg", "random", "special", "stats", "frame", "regression", "optimize")
THREADS = (1, 3, 10)
MAX_JOBS = 10
DEFAULT_JOBS = 8
SMALL = 1024  # arrays up to this many elements are stored element by element
CASE_TIMEOUT = 120.0  # seconds; GOLDEN_TIMEOUT overrides
FORMAT = 1

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


class Encoder:
    """enc(x): a JSON-able form of a result, exact to the bit."""

    def __init__(self, L, area_encode=None):
        self.L, self.sd, self.np = L, L.sd, L.np
        self.area_encode = area_encode
        self.stack = []

    def __call__(self, x):
        return self.enc(x)

    def enc(self, x):
        np, sd = self.np, self.sd
        if isinstance(x, BaseException):
            return {"error": type(x).__name__, "msg": str(x)}
        if self.area_encode is not None:
            e = self.area_encode(x, self.L, self)
            if e is not None:
                return e
        if x is None or isinstance(x, (bool, str)):
            return x
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return {"f": fhex(x)}
        if isinstance(x, complex):
            return {"c": [fhex(x.real), fhex(x.imag)]}
        if isinstance(x, sd.ndarray):
            return self.array(np.asarray(x), "nd", str(x.dtype), list(x.shape), list(x.strides))
        if isinstance(x, np.ndarray):
            return self.array(x, "np", x.dtype.str, list(x.shape), list(x.strides))
        if isinstance(x, np.generic):
            return {"npscalar": x.dtype.str, "v": self.enc(x.item())}
        if isinstance(x, type(sd.float64)):
            return {"dtype": str(x)}
        if isinstance(x, tuple):
            out = {"tuple": type(x).__name__, "items": [self.enc(v) for v in x]}
            if hasattr(type(x), "_fields"):
                out["fields"] = list(type(x)._fields)
            return out
        if isinstance(x, list):
            return [self.enc(v) for v in x]
        if isinstance(x, dict):
            return {"dict": [[self.enc(k), self.enc(v)] for k, v in x.items()]}
        if isinstance(x, (set, frozenset)):
            items = [self.enc(v) for v in x]
            return {"set": sorted(items, key=lambda e: json.dumps(e, sort_keys=True))}
        if isinstance(x, (bytes, bytearray)):
            return {"bytes": bytes(x).hex()}
        if isinstance(x, (range, slice)) or x is Ellipsis:
            return {"repr": repr(x)}
        if isinstance(x, sd.random.Generator):
            return {"generator": self.enc(x.state)}
        if isinstance(x, type) or inspect.isroutine(x):
            return {"callable": f"{getattr(x, '__module__', '?')}.{getattr(x, '__qualname__', repr(x))}"}
        if type(x).__module__.startswith("stardust"):
            return self.obj(x)
        raise TypeError(f"golden: no encoding for {type(x).__module__}.{type(x).__qualname__}")

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
                out["n"] = n
                out["sha"] = hashlib.sha256(np.ascontiguousarray(flat).tobytes()).hexdigest()[:32]
                if kind in "fc":  # the sha with every NaN made the default one, when that differs
                    canon = np.where(np.isnan(flat), np.nan, flat) if kind == "f" else canon_complex(np, flat)
                    shan = hashlib.sha256(np.ascontiguousarray(canon).tobytes()).hexdigest()[:32]
                    if shan != out["sha"]:
                        out["shan"] = shan
                out["sample"] = [[i, elem(flat[i].item())] for i in sample_indices(n)]
        else:  # object, string: element by element through enc
            vals = [self.enc(v) for v in flat.tolist()]
            if n <= SMALL:
                out["v"] = vals
            else:
                out["n"] = n
                blob = json.dumps(vals, sort_keys=True, separators=(",", ":")).encode()
                out["sha"] = hashlib.sha256(blob).hexdigest()[:32]
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
            return {"obj": f"{type(x).__module__}.{type(x).__qualname__}", "attrs": attrs}
        finally:
            self.stack.pop()


def canon_complex(np, z):
    out = np.empty_like(z)  # no arithmetic: inf * 1j would raise numpy's warning inside the case
    out.real = np.where(np.isnan(z.real), np.nan, z.real)
    out.imag = np.where(np.isnan(z.imag), np.nan, z.imag)
    return out


def any_nan(e):
    """An encoding with NaN bits forgotten: every NaN the default one (check --any-nan)."""
    if isinstance(e, str):
        return "nan" if e.startswith("nan:") else e
    if isinstance(e, list):
        return [any_nan(v) for v in e]
    if isinstance(e, dict):
        out = {k: any_nan(v) for k, v in e.items() if k != "shan"}
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


def run_worker(area, threads, out, case_rx):
    L, mod = load_area(area)
    rx = re.compile(case_rx) if case_rx else None
    L.sd.set_num_threads(threads)
    E = Encoder(L, getattr(mod, "encode", None))
    todo, seen = [], set()
    for name, thunk in mod.cases(L):
        if name in seen:
            raise SystemExit(f"golden: {area}: duplicate case name {name!r}")
        seen.add(name)
        if rx is None or rx.search(name):
            todo.append((name, thunk))
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
            rec["w"] = [[w.category.__name__, str(w.message)] for w in caught]
        faulthandler.cancel_dump_traceback_later()
        times[name] = time.perf_counter() - t0
        res[name] = rec
        del r
    cur.close()
    os.unlink(out + ".cur")
    pathlib.Path(out).write_text(json.dumps({"cases": res, "times": times}, separators=(",", ":")))


# --------------------------------------------------------------------------------------------
# driver


def count_cases(areas, case_rx):
    """{area: the number of cases a worker will run} (the cases are made, not run: about a second)."""
    rx = re.compile(case_rx) if case_rx else None
    out = {}
    for area in areas:
        L, mod = load_area(area)
        out[area] = sum(1 for name, _ in mod.cases(L) if rx is None or rx.search(name))
    return out


def run_all(areas, threads, jobs, case_rx, con, counts):
    """{area: {threads: {name: record}}}, {area: {name: seconds}}, seconds, [failures] -- the
    workers in parallel, a bar over every case at every thread count."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="sd-golden-"))
    env = dict(os.environ, PYTHONHASHSEED="0")
    # Big areas first so the slowest workers start early.
    order = {"frame": 0, "array": 1, "regression": 2, "stats": 3, "special": 4}
    tasks = sorted(((a, n) for a in areas for n in threads), key=lambda t: (order.get(t[0], 9), t[1]))
    total = sum(counts[a] for a, _ in tasks)
    running, results, times, failed = [], {}, {}, []
    finished, last_tenth, last_drawn, seen = 0, -1, 0.0, {}
    t0 = time.perf_counter()
    con.start(total, "golden")
    while tasks or running:
        while tasks and len(running) < jobs:
            a, n = tasks.pop(0)
            out = tmp / f"{a}-{n}.json"
            cmd = [sys.executable, __file__, "_worker", a, str(n), str(out)] + (["--case", case_rx] if case_rx else [])
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
    shutil.rmtree(tmp, ignore_errors=True)
    return results, times, time.perf_counter() - t0, failed


def cur_case(path):
    """(cases done, running case) from a worker's .cur file; (0, "") before it has one."""
    try:
        k, _, name = path.read_text().partition("\t")
        return int(k), name
    except OSError, ValueError:
        return 0, ""


def merge_threads(per):
    """{name: record or {"per_threads": {n: record}}}, [names whose record depends on threads]."""
    names = list(next(iter(per.values())).keys())
    out, unstable = {}, []
    for name in names:
        recs = {n: per[n].get(name) for n in per}
        vals = list(recs.values())
        if all(v == vals[0] for v in vals[1:]):
            out[name] = vals[0]
        else:
            unstable.append(name)
            out[name] = {"per_threads": {str(n): v for n, v in recs.items()}}
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
    """Units in the last place between two finite floats of the same sign, else None."""
    try:
        x, y = unhex(a), unhex(b)
    except ValueError, OverflowError:
        return None
    if not (math.isfinite(x) and math.isfinite(y)) or math.copysign(1.0, x) != math.copysign(1.0, y):
        return None

    def key(v):
        i = struct.unpack("<q", struct.pack("<d", v))[0]
        return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)

    return abs(key(x) - key(y))


def show_elem(e):
    if isinstance(e, str):
        return show_float(e)
    if isinstance(e, list) and len(e) == 2 and all(isinstance(t, str) for t in e):
        return f"{show_float(e[0])} + {show_float(e[1])}j"
    s = json.dumps(e, sort_keys=True)
    return s if len(s) <= 160 else s[:157] + "..."


def elem_ulps(g, w):
    if isinstance(g, str) and isinstance(w, str):
        u = ulps(g, w)
        return f", {u} ulp" if u is not None else ""
    if isinstance(g, list) and isinstance(w, list) and len(g) == len(w) == 2:
        us = [ulps(a, b) for a, b in zip(g, w)]
        if all(u is not None for u in us):
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


def diff(got, want, path, out):
    """Append (path, text) for every difference between two encodings."""
    if got == want:
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
        if got.get(k) != want.get(k):
            out.append((path, f"{'dtype' if k == tag else k} {got.get(k)}, want {want.get(k)}"))
    if got.get("shape") != want.get("shape"):
        return
    shape = want["shape"]
    if "v" in got and "v" in want:
        bad = [i for i, (g, w) in enumerate(zip(got["v"], want["v"])) if g != w]
        if bad:
            first = bad[0]
            g, w = got["v"][first], want["v"][first]
            if all(any_nan(got["v"][i]) == any_nan(want["v"][i]) for i in bad):
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
            if got.get("shan", got["sha"]) == want.get("shan", want["sha"]):
                out.append((path, f"only NaN bits differ (sign or payload; {want['n']} elements, sha)"))
                return
            gs, ws = dict(map(tuple, got["sample"])), dict(map(tuple, want["sample"]))
            bad = [i for i in ws if gs.get(i) != ws[i]]
            text = f"bytes differ ({want['n']} elements, sha); {len(bad)} of {len(ws)} sampled elements differ"
            if bad:
                i = bad[0]
                g, w = gs.get(i), ws[i]
                text += f"; at {unravel(i, shape)}: got {show_elem(g)}, want {show_elem(w)}{elem_ulps(g, w)}"
            out.append((path, text))
    else:
        out.append((path, "stored differently (size class changed)"))


def short(e):
    s = json.dumps(e, sort_keys=True, ensure_ascii=False)
    return s if len(s) <= 200 else s[:197] + "..."


# --------------------------------------------------------------------------------------------
# commands


def golden_path(area, out_dir=None):
    return (pathlib.Path(out_dir) if out_dir else DATA) / f"{area}.json"


def shown_path(path):
    path = pathlib.Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


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
                ("Selection", f"--case {args.case}" if args.case else "every case"),
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
    return 2


def cmd_record(args, areas, threads, con, log):
    counts = count_cases(areas, args.case)
    header(con, "record", args, areas, threads, counts, log)
    results, times, wall, failed = run_all(areas, threads, args.jobs, args.case, con, counts)
    if failed:
        return worker_failures(con, failed)
    rows, unstable_all, total = [], [], 0
    for area in areas:
        merged, unstable = merge_threads(results[area])
        path = golden_path(area, args.out)
        if args.case:  # a partial record updates only the matching cases
            old = json.loads(path.read_text())["cases"] if path.exists() else {}
            old.update(merged)
            merged = old
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"format": FORMAT, "area": area, "threads": list(threads), "cases": merged}
        path.write_text(json.dumps(doc, indent=0, sort_keys=True, ensure_ascii=False) + "\n")
        nerr = sum(1 for v in merged.values() if isinstance(v.get("r"), dict) and "error" in v["r"])
        nw = sum(1 for v in merged.values() if "w" in v)
        size = path.stat().st_size / 1e6
        rows.append([area, f"{len(merged):,}", f"{nerr:,}", f"{nw:,}", f"{size:.1f} MB", shown_path(path)])
        total += len(merged)
        unstable_all += [f"{area}/{name}" for name in unstable]
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
                ("Partial", "yes" if args.case else "no"),
            ],
        )  # fmt: skip
        + C.table(["Area", "Cases", "Errors", "Warnings", "Size", "File"], rows, "lrrrrl")
    )
    if unstable_all:
        con.print("")
        con.print("Results that depend on the thread count (stored per thread count):")
        con.lines([f"  {name}" for name in unstable_all])
    slow_table(con, times, args.slow)
    notes = [
        "Errors and warnings are results: a case that raises stores the error's type and message.",
        "Check the record against the library again with: .venv/bin/python tools/golden.py check",
        f"The log: {shown_path(log)}",
    ]
    con.lines([""] + C.notes(notes))
    return 0


def cmd_check(args, areas, threads, con, log):
    counts = count_cases(areas, args.case)
    header(con, "check", args, areas, threads, counts, log)
    results, times, wall, failed = run_all(areas, threads, args.jobs, args.case, con, counts)
    if failed:
        return worker_failures(con, failed)
    rx = re.compile(args.case) if args.case else None
    rows, lines, totals = [], [], {"cases": 0, "changed": 0, "new": 0, "gone": 0, "missing": 0}
    for area in areas:
        path = golden_path(area, args.out)
        per = results[area]
        names = list(next(iter(per.values())).keys())
        totals["cases"] += len(names)
        if not path.exists():
            rows.append([area, f"{len(names):,}", "", "", "", "no golden file"])
            totals["missing"] += 1
            continue
        golden = json.loads(path.read_text())["cases"]
        if rx is not None:
            golden = {k: v for k, v in golden.items() if rx.search(k)}
        changed, new, gone = 0, 0, sorted(set(golden) - set(names))
        for name in names:
            if name not in golden:
                new += 1
                lines.append(f"{con.color('NEW', '33')}      {area}/{name}: not in the golden file")
                continue
            want_all = golden[name]
            groups = {}  # identical differences across thread counts are reported once
            for n in threads:
                want = want_all["per_threads"].get(str(n)) if "per_threads" in want_all else want_all
                got = per[n][name]
                if got == want or (args.any_nan and any_nan(got) == any_nan(want)):
                    continue
                d = []
                diff(got, want, "", d)
                key = json.dumps(d)
                groups.setdefault(key, (d, []))[1].append(n)
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
                ("No golden file", str(totals["missing"])),
                ("Threads", ",".join(map(str, threads))),
                ("Time", C.fmt_duration(wall)),
                ("NaN bits", "ignored" if args.any_nan else "compared"),
                ("Exit status", "1" if nbad else "0"),
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
        what = " ".join(a for a, r in zip(areas, rows) if "differ" in r[-1] or "no golden" in r[-1])
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
    return 1 if nbad else 0


def status_colors(con, lines):
    """A table's status column colored after layout (the table measures plain text)."""
    out = []
    for line in lines:
        for word, code in (("equal", "32"), ("differs", "31"), ("no golden file", "31")):
            if line.endswith("   " + word):
                line = line[: -len(word)] + con.color(word, code)
                break
        out.append(line)
    return out


def cmd_list(areas, con):
    rows, total = [], 0
    for area in areas:
        L, mod = load_area(area)
        n = sum(1 for _ in mod.cases(L))
        rows.append([area, f"{n:,}", shown_path(GOLDEN / f"cases_{area}.py")])
        total += n
    con.lines(
        C.block("stardust golden cases", [("Areas", str(len(areas))), ("Cases", f"{total:,}")])
        + C.table(["Area", "Cases", "Defined in"], rows, "lrl")
    )
    return 0


def slow_table(con, times, n):
    if not n:
        return
    flat = sorted(((t, f"{a}/{k}") for a, d in times.items() for k, t in d.items()), reverse=True)[:n]
    con.print("")
    con.print(f"The {len(flat)} slowest cases (the slowest thread count):")
    con.lines(C.table(["Case", "Seconds"], [[k, f"{t:.3f}"] for t, k in flat], "lr"))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        ap = argparse.ArgumentParser()
        ap.add_argument("_w")
        ap.add_argument("area")
        ap.add_argument("threads", type=int)
        ap.add_argument("out")
        ap.add_argument("--case")
        a = ap.parse_args()
        return run_worker(a.area, a.threads, a.out, a.case)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["record", "check", "list"])
    ap.add_argument("areas", nargs="*", metavar="AREA", help=f"default: all ({', '.join(AREAS)})")
    ap.add_argument("--case", help="regex on case names")
    ap.add_argument("--threads", default=",".join(map(str, THREADS)), help="thread counts (default 1,3,10)")
    ap.add_argument(
        "--jobs", type=int, default=DEFAULT_JOBS, help=f"worker processes (default {DEFAULT_JOBS}, at most {MAX_JOBS})"
    )
    ap.add_argument("--out", help="directory of the golden files (default tests/golden/data)")
    ap.add_argument("--slow", type=int, default=0, metavar="N", help="print the N slowest cases")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--any-nan", action="store_true", help="check: NaNs that differ only in sign or payload are equal")
    args = ap.parse_args()
    areas = args.areas or list(AREAS)
    for a in areas:
        if a not in AREAS:
            ap.error(f"unknown area {a!r} (one of {', '.join(AREAS)})")
    threads = tuple(int(t) for t in args.threads.split(","))
    args.jobs = max(1, min(args.jobs, MAX_JOBS))
    if args.command == "list":
        return cmd_list(areas, C.Console(stream=sys.stdout))
    log = procrun.log_path("golden")
    with open(log, "w") as fh:
        con = C.Console(log=fh, stream=sys.stdout)
        return (cmd_record if args.command == "record" else cmd_check)(args, areas, threads, con, log)


if __name__ == "__main__":
    sys.exit(main())
