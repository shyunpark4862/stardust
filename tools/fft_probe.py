"""Probes of sd.fft's foundations and of the libraries it is compared with (docs/design/fft.md §3.1,
§3.2, §4.4, §5, §7.4). The records the tests read go to tests/fft/data/; the timings go to the
terminal and to build/logs/fft_probe/<stamp>.log, and docs/design/fft.md quotes them.

    .venv/bin/python tools/fft_probe.py                   # every section
    .venv/bin/python tools/fft_probe.py paths accuracy    # only these sections

Sections (positional; default all):
  paths     decision FT22: for every length class vDSP accepts (2^k, 3 2^k, 5 2^k, 15 2^k; complex
            and real) the prototype of the own path (csrc/fft/cfft.hpp, rfft.hpp) against vDSP,
            both end to end from sd's layouts (interleaved complex128 in and out; vDSP's best of
            zop with the (de)interleaving and, up to 4096, its Interleaved setup), three runs in
            turn, the median of each; a class goes to the path with the smaller geometric mean
            time over its lengths 16 <= n <= 2^20 -> tests/fft/data/prototype.json
  twiddle   csrc/fft/twiddle.hpp's factors against Arb, in ULP of each part -> prototype.json
  vdsp      vDSP's length sets (every n to 4096 for zop, zrop, Interleaved and the float DCT, the
            last in a child that may die; f 2^k to 2^22), bit determinism (repeat, misaligned,
            10 threads, in place, recreated, sharing `previous`, inverse vs conj(forward(conj))),
            setup time and memory per class, CPU / wall time of one execution, and the split and
            Interleaved costs -> tests/fft/data/vdsp.json
  accuracy  the contract units (tests/fft/refdata.py) of numpy, scipy, the prototype and vDSP on
            the stored c2c and real references of the class lengths -> prototype.json
  numpy     numpy 2.5.4's edge behaviours and errors, rfft vs the start of fft, the NaN and inf
            patterns, fftfreq's formula, fftshift vs roll -> tests/fft/data/numpy.json
  scipy     scipy 1.18.1's edge behaviours and errors (a call that may kill the process runs in a
            child), DCT and DST edges, next_fast_len / prev_fast_len, the NaN and inf patterns
            against numpy's -> tests/fft/data/scipy.json
  r         R 4.6.2's stats::fft, mvfft and nextn (values as hex floats) -> tests/fft/data/r.json
  time      the numpy and scipy timings of docs/design/fft.md §5 (best of five)

The C++ program (tools/fft_probe_kernels.cpp) is compiled into build/fft_probe/ with sd's flags
(-O3 -mcpu=native -ffp-contract=off) under the shared build lock (build/.agent-build-lock). It runs
as one child at a time with a time limit; its resident memory is read every 0.1 s and the child is
killed above MEMORY_CAP (a vDSP setup is about 14-34 bytes per point, and the program creates
none above 512 MB).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import pathlib
import statistics
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench import cli  # noqa: E402

BUILD = ROOT / "build" / "fft_probe"
DATA = ROOT / "tests" / "fft" / "data"
LOCK = ROOT / "build" / ".agent-build-lock"
FLAGS = ["-std=c++20", "-O3", "-mcpu=native", "-ffp-contract=off", "-fno-builtin-sin", "-fno-builtin-cos"]
SRC = ROOT / "tools" / "fft_probe_kernels.cpp"
DEPS = [SRC] + [ROOT / "csrc" / "fft" / f for f in ("cfft.hpp", "rfft.hpp", "twiddle.hpp", "vdsp.hpp")]
SECTIONS = ("paths", "twiddle", "vdsp", "accuracy", "numpy", "scipy", "r", "time")
CHILD_SECONDS = 1800
MEMORY_CAP = 3 << 30  # bytes of resident memory of the C++ child
PATHS_RUNS = 3
PATHS_MAXN = 1 << 20
SETUP_CAP = 1 << 22
CLASSES = (1, 3, 5, 15)


def S():
    from tests.fft import signals

    return signals


def R():
    from tests.fft import refdata

    return refdata


class Log:
    def __init__(self):
        self.path = cli.log_file("fft_probe")
        print(f"log: {self.path.relative_to(ROOT)}", flush=True)

    def __call__(self, text=""):
        print(text, flush=True)
        with self.path.open("a") as f:
            f.write(text + "\n")


def load(arg):
    return f"load average {os.getloadavg()[0]:.1f} {arg}"


# ---------------------------------------------------------------- the C++ program


def build() -> pathlib.Path:
    exe = BUILD / "kernels"
    if exe.exists() and exe.stat().st_mtime >= max(d.stat().st_mtime for d in DEPS):
        return exe
    BUILD.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            LOCK.mkdir()
            break
        except FileExistsError:
            time.sleep(5)
    try:
        subprocess.run(
            ["clang++", *FLAGS, f"-I{ROOT / 'csrc'}", "-o", str(exe), str(SRC), "-framework", "Accelerate"],
            check=True,
            timeout=600,
        )
    finally:
        LOCK.rmdir()
    return exe


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
PROC_PIDTASKINFO = 4
TASKINFO_SIZE = 96  # struct proc_taskinfo; pti_resident_size is its second uint64


def resident(pid: int) -> int:
    buf = ctypes.create_string_buffer(TASKINFO_SIZE)
    got = _libproc.proc_pidinfo(pid, PROC_PIDTASKINFO, ctypes.c_uint64(0), buf, TASKINFO_SIZE)
    return struct.unpack_from("Q", buf.raw, 8)[0] if got == TASKINFO_SIZE else 0


def run_watched(cmd, timeout=CHILD_SECONDS):
    """(returncode, stdout, stderr, peak resident bytes) of one child, read every 0.1 s and killed
    above MEMORY_CAP or past the time limit (returncode -9 then, with the reason in stderr)."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        p = subprocess.Popen([str(c) for c in cmd], stdout=out, stderr=err, text=True)
        peak, start, why = 0, time.monotonic(), ""
        while p.poll() is None:
            rss = resident(p.pid)
            peak = max(peak, rss)
            if rss > MEMORY_CAP:
                why = f"killed: resident {rss / 2**30:.2f} GB above the cap"
            elif time.monotonic() - start > timeout:
                why = f"killed: past {timeout} s"
            if why:
                p.kill()
                p.wait()
                break
            time.sleep(0.1)
        out.seek(0), err.seek(0)
        return p.returncode, out.read(), (why + "\n" if why else "") + err.read(), peak


def tsv(text, tag):
    return [line.split("\t")[1:] for line in text.splitlines() if line.startswith(tag + "\t")]


# ---------------------------------------------------------------- records


def plain(v):
    """A JSON-ready form of a result: arrays as {"array", "dtype", "shape"} (complex entries as
    [re, im]), floats as floats (NaN and inf as JSON's NaN and Infinity)."""
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            arr = [[float(z.real), float(z.imag)] for z in v.ravel()]
        else:
            arr = v.ravel().tolist()
        return {"array": arr, "dtype": str(v.dtype), "shape": list(v.shape)}
    if isinstance(v, (complex, np.complexfloating)):
        return {"complex": [float(v.real), float(v.imag)]}
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return {"np_int": int(v)}
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return {"type": type(v).__name__, "repr": repr(v)[:200]}


def outcome(fn):
    """{"value": ...} or {"error": type, "msg": message} of fn(), warnings recorded."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            v = fn()
            out = {"value": plain(v)}
        except Exception as e:  # noqa: BLE001 -- recording what the library raises
            out = {"error": type(e).__name__, "msg": str(e)[:300]}
    if w:
        out["warnings"] = sorted({x.category.__name__ for x in w})
    return out


def child_outcome(code):
    """The outcome of Python code run in a child (a call that may kill the process)."""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=ROOT)
    if p.returncode < 0 or p.returncode > 128:
        return {"crash": p.returncode}
    if p.returncode:
        last = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
        return {"error": last.split(":")[0], "msg": last[:300]}
    return {"stdout": p.stdout.strip()[:2000]}


def write_records(name, meta, records, merge=False):
    """tests/fft/data/<name>: one JSON object, "_provenance" first, one record per line. merge keeps
    the records of other sections already in the file."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    if merge and path.exists():
        old = json.loads(path.read_text())
        old.pop("_provenance", None)
        old.update(records)
        records = old
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in sorted(records.items())]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")
    return f"{len(records)} records -> {path.relative_to(ROOT)}"


def provenance(**extra):
    import scipy

    return {"tool": "tools/fft_probe.py", "numpy": np.__version__, "scipy": scipy.__version__, **extra}


# ---------------------------------------------------------------- paths (FT22)


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def paths(log):
    exe = build()
    runs = []
    for r in range(PATHS_RUNS):
        log(f"## paths run {r + 1} of {PATHS_RUNS} ({load('before')})")
        code, out, err, peak = run_watched([exe, "paths", PATHS_MAXN])
        if code:
            log(f"paths failed ({code}): {err[-1000:]}")
            return
        runs.append(tsv(out, "paths"))
        log(f"  peak resident {peak / 2**20:.0f} MB")
    table = {}
    for run in runs:
        for kind, f, k, n, own, vdsp, ilv, diff in run:
            key = (kind, int(f), int(k))
            vbest = min(float(vdsp), float(ilv)) if float(ilv) > 0 else float(vdsp)
            table.setdefault(key, {"n": int(n), "own": [], "vdsp": [], "diff": float(diff)})
            table[key]["own"].append(float(own))
            table[key]["vdsp"].append(vbest)
    rec = {}
    log("")
    log("| 종류 | 꼴 | n 범위 | 길이 수 | vDSP/자체 기하평균 | 범위 | n ≤ 4096 | n > 4096 | 길 |")
    log("|---|---|---|---:|---:|---|---:|---:|---|")
    for kind in ("c2c", "r2c"):
        for f in CLASSES:
            rows = sorted((k, v) for (kd, ff, k), v in table.items() if kd == kind and ff == f)
            if not rows:
                continue
            ratios, small, large, per = [], [], [], []
            for k, v in rows:
                own, vd = statistics.median(v["own"]), statistics.median(v["vdsp"])
                ratio = vd / own
                ratios.append(ratio)
                (small if v["n"] <= 4096 else large).append(ratio)
                per.append({"k": k, "n": v["n"], "own_us": own, "vdsp_us": vd, "ratio": ratio, "rel_diff": v["diff"]})
            g = geomean(ratios)
            path = "own" if g > 1.0 else "vdsp"
            rec[f"paths/{kind}/{f}"] = {"lengths": per, "geomean_vdsp_over_own": g, "path": path, "runs": PATHS_RUNS}
            name = "2^k" if f == 1 else f"{f}·2^k"
            log(
                f"| {'복소' if kind == 'c2c' else '실수'} | {name} | {per[0]['n']}–{per[-1]['n']} | {len(per)} | "
                f"{g:.2f} | "
                f"{min(ratios):.2f}–{max(ratios):.2f} | {geomean(small) if small else math.nan:.2f} | "
                f"{geomean(large) if large else math.nan:.2f} | {'자체' if path == 'own' else 'vDSP'} |"
            )
    log("")
    for (kind, f, k), v in sorted(table.items()):
        own, vd = statistics.median(v["own"]), statistics.median(v["vdsp"])
        log(
            f"  {kind} {f}·2^{k} = {v['n']}: own {own:.3f} us, vDSP {vd:.3f} us, ratio {vd / own:.2f}, "
            f"rel diff {v['diff']:.1e}"
        )
    log(f"({load('after')})")
    log(write_records("prototype.json", provenance(machine=machine()), rec, merge=True))


def machine():
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        osv = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip()
        return f"{cpu}, macOS {osv}"
    except OSError:
        return ""


# ---------------------------------------------------------------- twiddle


def twiddle(log):
    from flint import arb

    exe = build()
    rec = {}
    log("## twiddle: e^(-2 pi i k / n) against Arb, ULP of each part and |w_hat - w| / u")
    worst_all, mu_all = 0.0, 0.0
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "w.bin"
        for n in (7, 8, 12, 15, 24, 100, 120, 1000, 1009, 1024, 3072, 15360, 65536, 65537, 1 << 20):
            code, _, err, _ = run_watched([exe, "twiddle", n, out], timeout=300)
            if code:
                log(f"twiddle {n} failed: {err[-500:]}")
                continue
            w = np.fromfile(out, dtype=np.float64).reshape(n, 2)
            ks = range(n) if n <= 65537 else range(0, n, 97)
            worst, mean, count, mu = 0.0, 0.0, 0, 0.0
            from arbref import precision

            with precision(160):
                two_pi_n = 2 * arb.pi() / n
                for k in ks:
                    if (4 * k) % n == 0:  # a quarter turn: the parts are exactly 0 and +-1
                        q = (4 * k) // n
                        exact = [(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)][q % 4]
                        ok = w[k, 0] == exact[0] and w[k, 1] == exact[1]
                        worst = max(worst, 0.0 if ok else math.inf)
                        count += 2
                        continue
                    ang = two_pi_n * k
                    c, s_ = ang.cos(), -ang.sin()
                    for got, true in ((w[k, 0], c), (w[k, 1], s_)):
                        t = float(true.mid())
                        e = abs(float((arb(float(got)) - true).mid())) / math.ulp(t)
                        worst = max(worst, e)
                        mean += e
                        count += 1
                    d = (arb(float(w[k, 0])) - c) ** 2 + (arb(float(w[k, 1])) - s_) ** 2
                    mu = max(mu, float(d.sqrt().mid()) / 2.0**-53)
            rec[f"twiddle/{n}"] = {
                "max_ulp": worst,
                "mean_ulp": mean / max(count, 1),
                "checked": count,
                "mu_over_u": mu,
            }
            worst_all, mu_all = max(worst_all, worst), max(mu_all, mu)
            log(
                f"  n = {n}: max {worst:.3f} ULP, mean {mean / max(count, 1):.3f} ULP over {count} parts; "
                f"|w_hat - w| <= {mu:.3f} u"
            )
    log(f"  largest: {worst_all:.3f} ULP, |w_hat - w| <= {mu_all:.3f} u")
    log(write_records("prototype.json", provenance(machine=machine()), rec, merge=True))


# ---------------------------------------------------------------- vdsp


def vdsp(log):
    exe = build()
    rec = {}
    log(f"## vdsp ({load('at start')})")
    for kind in ("zop", "zrop", "ilv", "dct"):
        first, accepted, crashes = 1, [], []
        while first <= 4096:
            code, out, err, _ = run_watched([exe, "lengths", kind, first, 4096], timeout=900)
            accepted += [int(n) for _, n in tsv(out, "lengths")]
            if code == 0:
                break
            tried = [int(s) for s in err.split() if s.isdigit()]
            at = tried[-1] if tried else first
            crashes.append(at)
            first = at + 1
        merged = {}
        for n in accepted:
            f, k = n, 0
            while f % 2 == 0:
                f, k = f // 2, k + 1
            m = merged.setdefault(f, [k, 0])
            m[0], m[1] = min(m[0], k), m[1] + 1
        rec[f"lengths/{kind}"] = {
            "max": 4096,
            "accepted": accepted,
            "crashed": crashes,
            "odd_parts": {str(f): {"min_power": k, "count": c} for f, (k, c) in sorted(merged.items())},
        }
        log(f"  {kind} n <= 4096: " + ", ".join(f"{f}·2^>={k} ({c})" for f, (k, c) in sorted(merged.items())))
        if crashes:
            log(
                f"  {kind}: the setup call ended the process at {len(crashes)} lengths: {crashes[:8]}"
                f"{' ...' if len(crashes) > 8 else ''} (residues mod 4: {sorted({c % 4 for c in crashes})})"
            )
    code, out, err, peak = run_watched([exe, "big", SETUP_CAP])
    rec["big"] = [
        {"kind": kd, "f": int(f), "k": int(k), "n": int(n), "ok": bool(int(ok))} for kd, f, k, n, ok in tsv(out, "big")
    ]
    for kd in ("zop", "zrop", "ilv"):
        ok = [f"{r['f']}·2^{r['k']}" for r in rec["big"] if r["kind"] == kd and r["ok"] and r["n"] > 4096]
        log(f"  {kd} above 4096 up to 2^22: {len(ok)} lengths ({', '.join(ok[:6])}{' ...' if len(ok) > 6 else ''})")
    log(f"  peak resident {peak / 2**20:.0f} MB")
    code, out, err, _ = run_watched([exe, "det"])
    fields = ("repeat", "misaligned", "threads", "in_place", "recreated", "shared_previous", "inverse_is_conj")
    rec["det"] = [
        {"kind": kd, "n": int(n), **{f: int(v) for f, v in zip(fields, vals)}} for kd, n, *vals in tsv(out, "det")
    ]
    for r in rec["det"]:
        log(f"  det {r['kind']} n = {r['n']}: " + ", ".join(f"{f} {r[f]}" for f in fields))
    code, out, err, peak = run_watched([exe, "setup", SETUP_CAP])
    rec["setup"] = [
        {
            "kind": kd,
            "f": int(f),
            "k": int(k),
            "n": int(n),
            "forward_us": float(t1),
            "forward_mb": float(m1),
            "inverse_us": float(t2),
            "inverse_mb": float(m2),
            "own_us": float(t3),
            "own_mb": float(m3),
        }
        for kd, f, k, n, t1, m1, t2, m2, t3, m3 in tsv(out, "setup")
    ]
    for kd in ("zop", "zrop"):
        for f in CLASSES:
            rows = [r for r in rec["setup"] if r["kind"] == kd and r["f"] == f and r["n"] >= 1 << 14]
            if rows:
                per = [r["forward_mb"] * 2**20 / r["n"] for r in rows]
                big = rows[-1]
                log(
                    f"  setup {kd} {f}·2^k (n >= 2^14): {min(per):.1f}–{max(per):.1f} bytes per point; n = {big['n']}: "
                    f"{big['forward_us'] / 1e3:.1f} ms, {big['forward_mb']:.1f} MB "
                    f"(inverse sharing: +{big['inverse_mb']:.2f} MB)"
                )
    log(f"  peak resident {peak / 2**20:.0f} MB")
    code, out, err, _ = run_watched([exe, "cpu"])
    rec["cpu"] = [{"n": int(n), "us": float(us), "cpu_over_wall": float(r)} for n, us, r in tsv(out, "cpu")]
    for r in rec["cpu"]:
        log(f"  cpu n = {r['n']}: {r['us']:.0f} us, CPU / wall {r['cpu_over_wall']:.2f}")
    code, out, err, _ = run_watched([exe, "split"])
    rec["split"] = [
        {"n": int(n), "execute_us": float(a), "with_conversion_us": float(b), "interleaved_us": float(c)}
        for n, a, b, c in tsv(out, "split")
    ]
    for r in rec["split"]:
        ilv = f"{r['interleaved_us']:.2f}" if r["interleaved_us"] > 0 else "—"
        log(
            f"  split n = {r['n']}: execute {r['execute_us']:.2f} us, with (de)interleaving "
            f"{r['with_conversion_us']:.2f} us, Interleaved {ilv} us"
        )
    log(f"({load('at end')})")
    log(write_records("vdsp.json", provenance(machine=machine(), sdk="MacOSX26.sdk vDSP.h"), rec))


# ---------------------------------------------------------------- accuracy


def accuracy(log):
    import scipy.fft as sf

    exe = build()
    Rd, M = R(), S()
    rec = {}
    log("## accuracy: contract units (max over the stored signals of each length) and ||e|| / (u log2 n ||y||)")
    log("| n | 종류 | numpy | scipy | 자체 시제품 | vDSP |")
    log("|---:|---|---:|---:|---:|---:|")
    fams = (("c2c", "forward", "c2c"), ("real", "forward", "r2c"))
    with tempfile.TemporaryDirectory() as d:
        fin, fout = pathlib.Path(d) / "in.bin", pathlib.Path(d) / "out.bin"
        for fam, method, kind in fams:
            ref = Rd.ref(fam)
            m = ref.methods[method]
            for j, k in enumerate(m.index):
                sid = str(ref.inputs["signal"][k])
                n = int(ref.inputs["n"][k])
                if not sid.split("/")[1] == "uniform":
                    continue
                f, p = n, 0
                while f % 2 == 0:
                    f //= 2
                    p += 1
                if f not in CLASSES or n < 16 or (f > 1 and p < (3 if kind == "c2c" else 4)):
                    continue  # vDSP's classes only (csrc/fft/vdsp.hpp)
                x = ref.inputs["x"][k]
                hi, lo = m.hi[j], m.lo[j]
                kd = "c2c" if kind == "c2c" else "real"
                b = Rd.bound(kd, n, Rd.norm2(x))
                ynorm = Rd.norm2(hi)
                np.ascontiguousarray(x).tofile(fin)
                code, _, err, _ = run_watched([exe, "dump", kind, n, fin, fout], timeout=300)
                if code:
                    log(f"dump {kind} {n} failed: {err[-300:]}")
                    continue
                y = np.fromfile(fout, dtype=np.float64).view(np.complex128)
                half = len(y) // 2
                got = {
                    "numpy": np.fft.fft(x) if kind == "c2c" else np.fft.rfft(x),
                    "scipy": sf.fft(x) if kind == "c2c" else sf.rfft(x),
                    "own": y[:half],
                    "vdsp": y[half:],
                }
                row = {}
                for lib, v in got.items():
                    e = Rd.error(v, hi, lo)
                    row[lib] = {"units": Rd.units(e, b), "u_log2n": e / (Rd.U * math.log2(n) * ynorm)}
                rec[f"accuracy/{kind}/{n}"] = {"signal": sid, "digest": M.digest(x), **row}
                log(
                    f"| {n} | {kind} | "
                    + " | ".join(
                        f"{row[lib]['units']:.4f} ({row[lib]['u_log2n']:.2f})"
                        for lib in ("numpy", "scipy", "own", "vdsp")
                    )
                    + " |"
                )
    for lib in ("numpy", "scipy", "own", "vdsp"):
        us = [r[lib]["units"] for r in rec.values()]
        ul = [r[lib]["u_log2n"] for r in rec.values()]
        log(f"  {lib}: largest {max(us):.4f} units, u log2 n {min(ul):.2f}–{max(ul):.2f}")
    log(write_records("prototype.json", provenance(machine=machine()), rec, merge=True))


# ---------------------------------------------------------------- numpy and scipy


def _nonfinite_patterns(m):
    """{length: [the NaN/inf mask of fft and rfft of every one-entry non-finite real signal]}."""
    out = {}
    for n in range(2, 65):
        for _, x in S().nonfinite(n):
            if x.dtype != np.float64:
                continue
            for f in (m.fft, m.rfft):
                with np.errstate(all="ignore"):
                    y = f(x)
                code = np.isnan(y.real) * 1 + np.isinf(y.real) * 2 + np.isnan(y.imag) * 4 + np.isinf(y.imag) * 8
                out.setdefault(n, []).append(code.astype(np.int8).tobytes())
    return out


def _common_edges(m, lib):
    x = np.arange(1.0, 5.0)
    rec = {}
    e = rec.setdefault
    e("fft/empty", outcome(lambda: m.fft(np.array([]))))
    e("fft/n=0", outcome(lambda: m.fft(x, n=0)))
    e("fft/n=-1", outcome(lambda: m.fft(x, n=-1)))
    e("fft/n=1", outcome(lambda: m.fft(x, n=1)))
    e("fft/n=6 (zero padding)", outcome(lambda: m.fft(x, n=6)))
    e("fft/n=2.0", outcome(lambda: m.fft(x, n=2.0)))
    e("fft/n=True", outcome(lambda: m.fft(x, n=True)))
    e("fft/positional n", outcome(lambda: m.fft(x, 8)))
    e("fft/0-d", outcome(lambda: m.fft(np.float64(3.0))))
    e("fft/int64", outcome(lambda: m.fft(np.arange(4))))
    e("fft/bool", outcome(lambda: m.fft(np.array([True, False, True]))))
    e("fft/float32", outcome(lambda: m.fft(np.ones(4, np.float32))))
    e("fft/list", outcome(lambda: m.fft([1, 2, 3])))
    e("fft/norm=bad", outcome(lambda: m.fft(x, norm="bad")))
    e("fft/norm=None equals backward", outcome(lambda: bool(np.array_equal(m.fft(x, norm=None), m.fft(x)))))
    e("fft/axis=1 of 1-d", outcome(lambda: m.fft(x, axis=1)))
    e("fft/length-0 batch axis", outcome(lambda: m.fft(np.ones((0, 4)))))
    e("fft/length-0 transform axis", outcome(lambda: m.fft(np.ones((4, 0)))))
    e("fft/length 1 is a copy", outcome(lambda: m.fft(np.array([2.5 - 1j]))))
    e("ifft/[1,2,3,4] norm=forward", outcome(lambda: m.ifft(x, norm="forward")))
    e("fft/[1,2,3,4] n=6 entry 3", outcome(lambda: m.fft(x, n=6)[3]))
    e("rfft/complex input", outcome(lambda: m.rfft(x + 1j)))
    e("rfft/n=1", outcome(lambda: m.rfft(x, n=1)))
    e("rfft/imag of DC and Nyquist", outcome(lambda: m.rfft(np.arange(1.0, 9.0))[[0, 4]].imag))
    e("irfft/length 1 default n", outcome(lambda: m.irfft(np.array([2.0 + 1j]))))
    e("irfft/length 1 n=1", outcome(lambda: m.irfft(np.array([2.0 + 1j]), n=1)))
    e("irfft/default n", outcome(lambda: m.irfft(np.array([1, 2 + 1j, 3 + 5j]))))
    e(
        "irfft/DC and Nyquist imag dropped",
        outcome(
            lambda: bool(np.array_equal(m.irfft(np.array([1 + 9j, 2 + 1j, 3 + 5j])), m.irfft(np.array([1, 2 + 1j, 3]))))
        ),
    )
    e("irfft/n=5 odd", outcome(lambda: m.irfft(np.array([1 + 9j, 2 + 1j, 3 + 5j]), n=5)))
    e("irfft/real input", outcome(lambda: m.irfft(np.array([1.0, 2.0, 3.0]))))
    e("irfft/n=0", outcome(lambda: m.irfft(np.array([1.0, 2.0, 3.0]), n=0)))
    e("hfft/[1,2+1j,3+5j]", outcome(lambda: m.hfft(np.array([1, 2 + 1j, 3 + 5j]))))
    e("ihfft/[1,2,3,4]", outcome(lambda: m.ihfft(x)))
    e("ihfft/DC imag sign", outcome(lambda: math.copysign(1.0, m.ihfft(x)[0].imag)))
    e("ihfft/complex input", outcome(lambda: m.ihfft(x + 1j)))
    e("fftn/axes=(0,0)", outcome(lambda: m.fftn(np.arange(6.0).reshape(2, 3), axes=(0, 0))))
    e(
        "fftn/axes=(1,1) equals twice",
        outcome(
            lambda: bool(
                np.allclose(
                    m.fftn(np.arange(6.0).reshape(2, 3), axes=(1, 1)),
                    m.fft(m.fft(np.arange(6.0).reshape(2, 3), axis=1), axis=1),
                )
            )
        ),
    )
    e("fftn/s=(4,) without axes", outcome(lambda: m.fftn(np.ones((2, 3)), s=(4,)).shape))
    e("fftn/s=(4,5) without axes, 3-d", outcome(lambda: m.fftn(np.ones((2, 2, 3)), s=(4, 5)).shape))
    e("fftn/s=(-1,4)", outcome(lambda: m.fftn(np.ones((2, 3)), s=(-1, 4)).shape))
    e("fftn/s=(None,4)", outcome(lambda: m.fftn(np.ones((2, 3)), s=(None, 4)).shape))
    e("fftn/len(s) != len(axes)", outcome(lambda: m.fftn(np.ones((2, 3)), s=(4,), axes=(0, 1))))
    e("fftn/axes=() returns its input", outcome(lambda: (lambda a: m.fftn(a, axes=()) is a)(np.ones((2, 3)))))
    e("fftn/axes=() dtype", outcome(lambda: str(m.fftn(np.ones((2, 3)), axes=()).dtype)))
    e("rfftn/axes=()", outcome(lambda: m.rfftn(np.ones((2, 3)), axes=())))
    e("fft2/1-d", outcome(lambda: m.fft2(x)))
    e("fftn/0-d", outcome(lambda: m.fftn(np.array(3.0))))
    e("irfftn/default s", outcome(lambda: m.irfftn(np.ones((2, 3), complex)).shape))
    e("fftfreq/0", outcome(lambda: m.fftfreq(0)))
    e("fftfreq/d=0", outcome(lambda: m.fftfreq(5, d=0)))
    e("fftfreq/5", outcome(lambda: m.fftfreq(5)))
    e("fftfreq/positional d", outcome(lambda: m.fftfreq(4, 0.1)))
    e("fftfreq/5.0", outcome(lambda: m.fftfreq(5.0)))
    e("fftfreq/-3", outcome(lambda: m.fftfreq(-3)))
    e("fftfreq/d=-0.5", outcome(lambda: m.fftfreq(4, d=-0.5)))
    e("rfftfreq/6", outcome(lambda: m.rfftfreq(6)))
    e("rfftfreq/0", outcome(lambda: m.rfftfreq(0)))
    e("fftshift/int64", outcome(lambda: m.fftshift(np.arange(5))))
    e("fftshift/axes=(0,0)", outcome(lambda: m.fftshift(np.arange(6).reshape(2, 3), axes=(0, 0))))
    e("fftshift/list", outcome(lambda: m.fftshift([1, 2, 3])))
    # bits: fftfreq is k * (1 / (n d)); fftshift is roll; rfft is the start of fft
    same_freq, same_shift = True, True
    for n in range(1, 200):
        for d in (1.0, 0.1, 3.7, 1e-3):
            k = np.concatenate([np.arange(0, (n - 1) // 2 + 1), np.arange(-(n // 2), 0)])
            same_freq &= bool(np.array_equal(m.fftfreq(n, d=d), k * (1.0 / (n * d))))
        a = np.arange(n * 3.0).reshape(3, n)
        same_shift &= bool(np.array_equal(m.fftshift(a, axes=1), np.roll(a, n // 2, axis=1)))
        same_shift &= bool(np.array_equal(m.ifftshift(a, axes=1), np.roll(a, -(n // 2), axis=1)))
    rec["fftfreq/bits are k * (1 / (n d))"] = {"value": same_freq}
    rec["fftshift/bits are roll"] = {"value": same_shift}
    differ = []
    for n in list(range(1, 129)) + [255, 256, 1000, 1009, 1024, 3072]:
        xr = S().get(f"r/uniform/{n}").x
        if not np.array_equal(m.rfft(xr), m.fft(xr)[: n // 2 + 1]):
            differ.append(n)
    rec["rfft/differs from the start of fft at lengths"] = {"value": differ, "checked": 134}
    x16 = S().get("c/uniform/16").x
    rec["fft/scaling by 2 is exact"] = {"value": bool(np.array_equal(m.fft(2 * x16), 2 * m.fft(x16)))}
    return rec


def numpy_records(log):
    rec = _common_edges(np.fft, "numpy")
    rec["fft/out="] = outcome(lambda: np.fft.fft(np.arange(4.0), out=np.empty(4, complex)).dtype.name)
    rec["fft/device="] = outcome(lambda: np.fft.fftfreq(4, device="cpu"))
    log(write_records("numpy.json", provenance(), rec))


SCIPY_CHILD = (
    "import numpy as np, scipy.fft as sf; r = {call}; print(type(r).__name__, getattr(r, 'tolist', lambda: r)())"
)


def scipy_records(log):
    import scipy.fft as sf

    rec = _common_edges(sf, "scipy")
    x = np.arange(1.0, 5.0)
    e = rec.setdefault
    for call in (
        "sf.dct(np.array([2.0]), type=1)",
        "sf.idct(np.array([2.0]), type=1)",
        "sf.dct(np.array([2.0]), type=1, norm='ortho')",
        "sf.dst(np.array([2.0]), type=1)",
    ):
        e(f"child/{call}", child_outcome(SCIPY_CHILD.format(call=call)))
    e("dct/type=2 n=1", outcome(lambda: sf.dct(np.array([2.0]))))
    e("dct/empty", outcome(lambda: sf.dct(np.array([]))))
    e("dct/type=5", outcome(lambda: sf.dct(x, type=5)))
    e("dct/type=2.0", outcome(lambda: sf.dct(x, type=2.0)))
    e("dct/positional type", outcome(lambda: sf.dct(x, 2)))
    e("dct/complex input", outcome(lambda: sf.dct(x + 2j)))
    e(
        "dct/complex is parts",
        outcome(lambda: bool(np.array_equal(sf.dct(x + 2j * x[::-1]), sf.dct(x) + 1j * sf.dct(x[::-1])))),
    )
    e("dct/int64", outcome(lambda: sf.dct(np.arange(4))))
    e("dct/n=0", outcome(lambda: sf.dct(x, n=0)))
    e("dct/type=1 n=2", outcome(lambda: sf.dct(np.array([1.0, 2.0]), type=1)))
    for t in (1, 2, 3, 4):
        e(
            f"dct/type={t} ortho orthogonalize=False",
            outcome(lambda t=t: sf.dct(x, type=t, norm="ortho", orthogonalize=False)),
        )
        e(f"dct/type={t} backward orthogonalize=True", outcome(lambda t=t: sf.dct(x, type=t, orthogonalize=True)))
        e(f"dst/type={t} ortho", outcome(lambda t=t: sf.dst(x, type=t, norm="ortho")))
    e("dctn/axes=(0,0)", outcome(lambda: sf.dctn(np.ones((2, 3)), axes=(0, 0))))
    e("dctn/s=(-1,4)", outcome(lambda: sf.dctn(np.ones((2, 3)), s=(-1, 4)).shape))
    nfl = [sf.next_fast_len(k) for k in range(1, 2001)]
    nflr = [sf.next_fast_len(k, True) for k in range(1, 2001)]
    rec["next_fast_len/1..2000"] = {"value": nfl}
    rec["next_fast_len/real 1..2000"] = {"value": nflr}
    rec["prev_fast_len/1..2000"] = {"value": [sf.prev_fast_len(k) for k in range(1, 2001)]}
    rec["prev_fast_len/real 1..2000"] = {"value": [sf.prev_fast_len(k, True) for k in range(1, 2001)]}
    for call, f in (
        ("next_fast_len(0)", lambda: sf.next_fast_len(0)),
        ("next_fast_len(-1)", lambda: sf.next_fast_len(-1)),
        ("next_fast_len(7.0)", lambda: sf.next_fast_len(7.0)),
        ("next_fast_len(True)", lambda: sf.next_fast_len(True)),
        ("next_fast_len(2**62 + 1)", lambda: sf.next_fast_len(2**62 + 1)),
        ("next_fast_len(2**63)", lambda: sf.next_fast_len(2**63)),
        ("prev_fast_len(0)", lambda: sf.prev_fast_len(0)),
        ("prev_fast_len(1)", lambda: sf.prev_fast_len(1)),
    ):
        e(f"helper/{call}", outcome(f))
    e("fft/workers=-1", outcome(lambda: sf.fft(x, workers=-1)))
    e("fft/workers=0", outcome(lambda: sf.fft(x, workers=0)))
    e("fft/plan=object()", outcome(lambda: sf.fft(x, plan=object())))
    e("fft/overwrite_x", outcome(lambda: sf.fft(x.copy(), overwrite_x=True)))
    e("hfft2/ones (2,3)", outcome(lambda: sf.hfft2(np.ones((2, 3), complex))))
    e("ihfftn/ones (2,4)", outcome(lambda: sf.ihfftn(np.ones((2, 4)))))
    # the NaN and inf patterns of numpy and scipy (docs/design/fft.md §2.8)
    a, b = _nonfinite_patterns(np.fft), _nonfinite_patterns(sf)
    differ = sum(int(p != q) for n in a for p, q in zip(a[n], b[n]))
    total = sum(len(v) for v in a.values())
    rec["nonfinite/patterns differing from numpy"] = {
        "value": differ,
        "of": total,
        "lengths": "2..64",
        "what": "fft and rfft of every real signal with one NaN, inf or -inf",
    }
    log(f"  NaN and inf patterns: scipy and numpy differ in {differ} of {total} transforms")
    log(write_records("scipy.json", provenance(), rec))


# ---------------------------------------------------------------- R


R_SIGNALS = ("c/uniform/8", "c/uniform/24", "c/uniform/1000", "c/uniform/1009", "c/impulse/7")


def r_records(log):
    M = S()
    rec = {}
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        script = ["out <- list()", 'hx <- function(z) paste(sprintf("%a", Re(z)), sprintf("%a", Im(z)))']
        for k, sid in enumerate(R_SIGNALS):
            x = M.get(sid).x
            np.savetxt(d / f"x{k}.txt", np.column_stack([x.real, x.imag]), fmt="%.17g")
            script.append(
                f'm <- as.matrix(read.table("{d / f"x{k}.txt"}")); z <- complex(real = m[, 1], imaginary = m[, 2])'
            )
            script.append(f'out[["fft|{sid}"]] <- hx(fft(z))')
            script.append(f'out[["inverse|{sid}"]] <- hx(fft(z, inverse = TRUE))')
        z2 = M.get("c/uniform/24").x[:24].reshape(12, 2, order="F")
        np.savetxt(d / "mv.txt", np.column_stack([z2.real.ravel(order="F"), z2.imag.ravel(order="F")]), fmt="%.17g")
        script.append(
            f'm <- as.matrix(read.table("{d / "mv.txt"}")); '
            "Z <- matrix(complex(real = m[, 1], imaginary = m[, 2]), 12, 2)"
        )
        script.append('out[["mvfft|c/uniform/24 as 12x2"]] <- hx(as.vector(mvfft(Z)))')
        script.append('out[["nextn|1..2000"]] <- as.character(sapply(1:2000, nextn))')
        script.append('for (k in names(out)) cat(k, "\\t", paste(out[[k]], collapse = " "), "\\n", sep = "")')
        (d / "run.R").write_text("\n".join(script))
        p = subprocess.run(
            ["Rscript", str(d / "run.R")],
            capture_output=True,
            text=True,
            timeout=600,
            env=dict(os.environ, OMP_NUM_THREADS="1"),
        )
        if p.returncode:
            log(f"R failed: {p.stderr[-2000:]}")
            return
        ver = subprocess.run(["Rscript", "-e", "cat(R.version.string)"], capture_output=True, text=True, timeout=60)
    for line in p.stdout.splitlines():
        key, _, vals = line.partition("\t")
        what, _, sid = key.partition("|")
        toks = vals.split()
        if what == "nextn":
            rec["nextn/1..2000"] = {"value": [int(t) for t in toks]}
            continue
        nums = [float.fromhex(t) for t in toks]
        z = [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]
        entry = {"values": z}
        if what in ("fft", "inverse"):
            entry["digest"] = M.digest(M.get(sid).x)
        else:
            entry["digest"] = M.digest(z2)
        rec[f"{what}/{sid}"] = entry
    log(write_records("r.json", {"tool": "tools/fft_probe.py", "R": ver.stdout.strip()}, rec))


# ---------------------------------------------------------------- timings


def best_us(fn, reps=5):
    fn()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if time.perf_counter() - t0 > 2e-3 or n > 1 << 16:
            break
        n *= 2
    out = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        out = min(out, (time.perf_counter() - t0) / n)
    return out * 1e6


def timings(log):
    import scipy.fft as sf

    g = np.random.Generator(np.random.PCG64(20261123))
    log(f"## time ({load('at start')}), us, best of 5")
    log("| 일 | numpy | scipy | scipy workers=10 |")
    log("|---|---:|---:|---:|")

    def row(what, fnp, fsp, fsp10=None):
        cells = [f"{best_us(f):,.1f}" if f else "—" for f in (fnp, fsp, fsp10)]
        log(f"| {what} | " + " | ".join(cells) + " |")

    for n in (1024, 4096, 3072, 15360, 20480, 1 << 16, 1 << 20, 1000, 1009, 10007, 100003):
        x = g.uniform(-1, 1, n) + 1j * g.uniform(-1, 1, n)
        big = n >= 1 << 16 or n in (10007, 100003)
        row(
            f"fft n = {n}",
            lambda x=x: np.fft.fft(x),
            lambda x=x: sf.fft(x),
            (lambda x=x: sf.fft(x, workers=10)) if big else None,
        )
    for n in (1024, 1 << 16, 1 << 20):
        x = g.uniform(-1, 1, n)
        row(f"rfft n = {n}", lambda x=x: np.fft.rfft(x), lambda x=x: sf.rfft(x))
    for n in (1 << 16, 1 << 20):
        x = g.uniform(-1, 1, n)
        row(f"dct type 2 n = {n}", None, lambda x=x: sf.dct(x))
    X = g.uniform(-1, 1, (1000, 1024)) + 0j
    row("fft (1000, 1024) last axis", lambda: np.fft.fft(X), lambda: sf.fft(X), lambda: sf.fft(X, workers=10))
    X = g.uniform(-1, 1, (2048, 2048)) + 0j
    row("fft2 (2048, 2048)", lambda: np.fft.fft2(X), lambda: sf.fft2(X), lambda: sf.fft2(X, workers=10))
    log(f"({load('at end')})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools/fft_probe.py", description=__doc__.split("\n\n")[0])
    p.add_argument("sections", nargs="*", metavar="SECTION", help=f"sections (default all of {', '.join(SECTIONS)})")
    args = p.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        p.error(f"unknown section {unknown[0]!r} (choose from {', '.join(SECTIONS)})")
    log = Log()
    table = {
        "paths": paths,
        "twiddle": twiddle,
        "vdsp": vdsp,
        "accuracy": accuracy,
        "numpy": numpy_records,
        "scipy": scipy_records,
        "r": r_records,
        "time": timings,
    }
    for s in SECTIONS:
        if not args.sections or s in args.sections:
            table[s](log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
