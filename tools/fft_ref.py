"""Reference values for sd.fft (docs/design/fft.md §4, §7.2) on the test signals of
tests/fft/signals.py, with Arb (python-flint's acb.dft and ball arithmetic).

    .venv/bin/python tools/fft_ref.py                  # every family, on 3 processes
    .venv/bin/python tools/fft_ref.py c2c real         # only these families
    .venv/bin/python tools/fft_ref.py --list           # the families and their signals

writes tests/fft/ref/<family>.npz (layout: tests/reffile.py), read by tests/fft/refdata.py (which
describes the families). Every point names its signal (input "signal", the id in
tests/fft/signals.py) and holds the input itself; "digest" is signals.digest of the input, so a
changed generator shows. The inputs are doubles taken as exact, and acb.dft gives a ball that holds
the true transform whatever algorithm Arb uses inside.

    c2c        forward and backward DFT: uniform signals at every length of 1..128, the edges of
               vDSP's set and of 2^k, 2^k up to 2^14, f 2^k, primes and large prime factors, 7 2^k
               and 11-smooth lengths (backward up to 4096); the other kinds on a short list
    real       the forward DFT's half spectrum of real signals
    hermitian  the unnormalised inverse DFT of the Hermitian extension (irfft times n)
    r2r        the eight real transforms dct1 .. dst4 (scipy's norm='backward' definitions) from
               the DFT of the even or odd extension of length 2(N - 1), 2(N + 1), 4N or 8N
    nd         the n-dimensional forward and backward DFT (one acb.dft per axis)

Settling: a point is computed at 192 bits and again at twice the precision until every entry's
radius (real and imaginary part) is below 2^-120 of the output's 2-norm (FFT entries have no
relative accuracy of their own: docs/design/fft.md §4.1); then each entry's midpoint is stored as a
double-double. A point that does not settle by 8192 bits or within its time limit is dropped and
reported. Workers are forked, at most 3 (the area's rule for long runs), replaced every 8 tasks
(maxtasksperchild), each task under a time limit (SIGALRM). No mpmath (the project's rule). The
console output also goes to build/logs/fft_ref/.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import signal
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import arbref  # noqa: E402
from arbref import NotSettled, to_dd  # noqa: E402

OUT = ROOT / "tests" / "fft" / "ref"
TOOL = "tools/fft_ref.py"
SECONDS = 600  # a task's time limit
TARGET_BITS = 120  # every entry within 2^-120 of the output's norm
START_PREC = 192
MAX_PREC = 1 << 13
JOBS_MAX = 3
BACKWARD_MAX = 4096  # c2c and nd: the backward method up to this length

# The short list the other value kinds run on.
SHORT = (1, 2, 3, 4, 5, 7, 8, 13, 16, 24, 40, 64, 100, 120, 127, 128, 1000, 1024)
OTHER_KINDS = ("normal", "impulse", "constant", "sine", "sine_half", "wide", "huge", "tiny")


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def S():
    from tests.fft import signals

    return signals


# ---------------------------------------------------------------- the points of each family


def c2c_lengths():
    M = S()
    out = set(M.lengths("all128", "edge"))
    out.update(n for n in M.LENGTHS["pow2"] if n <= 1 << 14)
    out.update((192, 768, 1536, 320, 1280, 2560, 240, 960, 3840))  # f 2^k
    out.update((1009, 10007, 2018, 254, 1028))  # primes and large prime factors
    out.update((56, 224, 896, 3584, 99, 500, 539, 1000, 2310))  # 7 2^k and 11-smooth
    return sorted(out)


def real_lengths():
    return sorted(set(range(1, 129)) | {255, 256, 257, 1000, 1009, 1024, 1536, 2560, 3840, 4096})


def hermitian_lengths():
    return sorted(set(range(1, 65)) | {100, 127, 128, 255, 256, 1000, 1024, 4095, 4096})


def r2r_lengths():
    return sorted(set(range(1, 65)) | {100, 127, 128, 255, 256, 1000, 1024, 4096})


def _ids(domain, kinds_lengths):
    M = S()
    return [M.sid(domain, k, n) for k, ns in kinds_lengths for n in ns]


def c2c_points():
    return _ids("c", [("uniform", c2c_lengths()), ("normal", (3072,))] + [(k, SHORT) for k in OTHER_KINDS])


def real_points():
    return _ids("r", [("uniform", real_lengths())] + [(k, SHORT) for k in OTHER_KINDS])


def hermitian_points():
    short = (1, 2, 3, 4, 5, 8, 16, 17, 64, 100)
    return _ids("h", [("uniform", hermitian_lengths())] + [(k, short) for k in ("impulse", "constant", "wide", "tiny")])


def r2r_points():
    short = tuple(range(1, 17)) + (100,)
    return _ids("r", [("uniform", r2r_lengths())] + [(k, short) for k in ("normal", "impulse", "constant", "wide")])


def nd_points():
    M = S()
    return [M.sid("nd", "uniform", shape) for shape in M.SHAPES] + [M.sid("nd", "impulse", (8, 8))]


# ---------------------------------------------------------------- Arb


def _acb_list(x):
    from flint import acb

    return [acb(float(v.real), float(v.imag)) for v in np.asarray(x, np.complex128)]


def _conj(v):
    return [a.conjugate() for a in v]


def _norm_mid(vals) -> float:
    """The 2-norm of the balls' midpoints, as a float (for the settling test)."""
    from flint import arb

    s = arb(0)
    for a in vals:
        s += a.real.mid() ** 2 + a.imag.mid() ** 2
    return float(s.sqrt().mid())


def _settled(vals, scale: float) -> bool:
    from flint import arb

    if scale == 0.0:
        return all(a.real.is_exact() and a.imag.is_exact() for a in vals)
    limit = arb(scale) * arb(2) ** -TARGET_BITS
    return all(a.real.rad() <= limit and a.imag.rad() <= limit for a in vals)


def _rigorous(fn):
    """fn() -> {method: list of acb} at doubling precision until every method settles."""
    p = START_PREC
    while p <= MAX_PREC:
        with arbref.precision(p):
            out = fn()
            if all(_settled(v, _norm_mid(v)) for v in out.values()):
                return out
        p *= 2
    raise NotSettled(fn)


def _dd(vals, real=False):
    his, los = [], []
    for a in vals:
        rh, rl = to_dd(a.real.mid())
        if real:
            his.append(rh), los.append(rl)
            continue
        ih, il = to_dd(a.imag.mid())
        his.append(complex(rh, ih)), los.append(complex(rl, il))
    dt = np.float64 if real else np.complex128
    return np.array(his, dt), np.array(los, dt)


def _dft(v):
    from flint import acb

    return acb.dft(v) if v else []


def c2c_task(x):
    v = _acb_list(x)

    def f():
        out = {"forward": _dft(v)}
        if len(v) <= BACKWARD_MAX:
            out["backward"] = _conj(_dft(_conj(v)))
        return out

    return {k: _dd(vals) for k, vals in _rigorous(f).items()}


def real_task(x):
    v = _acb_list(x)
    m = len(v) // 2 + 1
    res = _rigorous(lambda: {"forward": _dft(v)[:m]})
    return {"forward": _dd(res["forward"])}


def hermitian_task(x, n):
    """The n reals sum_k Z_k e^(+2 pi i jk/n) of the Hermitian extension Z of the half spectrum x."""
    m = n // 2 + 1
    z = np.zeros(m, np.complex128)
    z[: min(m, len(x))] = np.asarray(x)[:m]
    z[0] = z[0].real
    if n % 2 == 0:
        z[m - 1] = z[m - 1].real
    full = np.concatenate([z, np.conj(z[1 : n - m + 1][::-1])])
    v = _acb_list(full)
    res = _rigorous(lambda: {"c2r": _conj(_dft(_conj(v)))})
    return {"c2r": _dd(res["c2r"], real=True)}


def _r2r_values(x, method):
    """The eight real transforms as the DFT of an extension (docs/design/fft.md §7.2)."""
    from flint import acb, arb

    xs = [arb(float(a)) for a in x]
    N = len(xs)
    zero = arb(0)

    def dft_of(v):
        return _dft([acb(a) for a in v])

    if method == "dct1":  # even extension of length 2(N - 1)
        v = xs + xs[-2:0:-1]
        return [a.real for a in dft_of(v)[:N]]
    if method == "dst1":  # odd extension of length 2(N + 1); y_k = -Im DFT_(k+1)
        v = [zero] + xs + [zero] + [-a for a in xs[::-1]]
        return [-a.imag for a in dft_of(v)[1 : N + 1]]
    if method in ("dct2", "dst2"):  # length 4N: v_(2n+1) = x_n, v_(4N-2n-1) = +-x_n
        sign = 1 if method == "dct2" else -1
        v = [zero] * (4 * N)
        for i, a in enumerate(xs):
            v[2 * i + 1] = a
            v[4 * N - 2 * i - 1] = a if sign > 0 else -a
        d = dft_of(v)
        return [a.real for a in d[:N]] if sign > 0 else [-a.imag for a in d[1 : N + 1]]
    if method == "dct3":  # length 4N at the odd frequencies: v_0 = x_0, v_m = v_(4N-m) = x_m
        v = [zero] * (4 * N)
        v[0] = xs[0]
        for i in range(1, N):
            v[i] = xs[i]
            v[4 * N - i] = xs[i]
        d = dft_of(v)
        return [d[2 * k + 1].real for k in range(N)]
    if method == "dst3":  # length 4N: v_(n+1) = x_n, v_(4N-n-1) = -x_n, the last term halved
        v = [zero] * (4 * N)
        for i in range(N):
            a = xs[i] / 2 if i == N - 1 else xs[i]
            v[i + 1] += a
            v[4 * N - i - 1] += -a
        d = dft_of(v)
        return [-d[2 * k + 1].imag for k in range(N)]
    if method in ("dct4", "dst4"):  # length 8N at the odd frequencies
        sign = 1 if method == "dct4" else -1
        v = [zero] * (8 * N)
        for i, a in enumerate(xs):
            v[2 * i + 1] = a
            v[8 * N - 2 * i - 1] = a if sign > 0 else -a
        d = dft_of(v)
        return [d[2 * k + 1].real if sign > 0 else -d[2 * k + 1].imag for k in range(N)]
    raise KeyError(method)


def r2r_task(x):
    from flint import acb

    from tests.fft import refdata

    methods = [m for m in refdata.R2R if not (m == "dct1" and len(x) < 2)]

    def f():
        return {m: [acb(a) for a in _r2r_values(x, m)] for m in methods}

    return {k: _dd(vals, real=True) for k, vals in _rigorous(f).items()}


def _nd_dft(arr, inverse_sign):
    out = arr
    for axis in range(arr.ndim):
        moved = np.moveaxis(out, axis, -1)
        res = np.empty(moved.shape, dtype=object)
        for idx in np.ndindex(moved.shape[:-1]):
            line = list(moved[idx])
            res[idx] = _conj(_dft(_conj(line))) if inverse_sign else _dft(line)
        out = np.moveaxis(res, -1, axis)
    return out


def nd_task(x):
    flat = _acb_list(x.ravel())
    arr = np.empty(x.shape, dtype=object)
    for i, a in enumerate(flat):
        arr[np.unravel_index(i, x.shape)] = a

    def f():
        out = {"forward": list(_nd_dft(arr, False).ravel())}
        if x.size <= BACKWARD_MAX:
            out["backward"] = list(_nd_dft(arr, True).ravel())
        return out

    return {k: tuple(a.reshape(x.shape) for a in _dd(vals)) for k, vals in _rigorous(f).items()}


# ---------------------------------------------------------------- driver

FAMILIES = {
    "c2c": c2c_points,
    "real": real_points,
    "hermitian": hermitian_points,
    "r2r": r2r_points,
    "nd": nd_points,
}


def _task(fam, sid):
    s = S().get(sid)
    if fam == "c2c":
        return c2c_task(s.x)
    if fam == "real":
        return real_task(s.x)
    if fam == "hermitian":
        return hermitian_task(s.x, s.n)
    if fam == "r2r":
        return r2r_task(s.x)
    return nd_task(s.x)


def _point(task):
    fam, k, sid = task
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SECONDS)
    t0 = time.monotonic()
    try:
        res, err = _task(fam, sid), None
    except (_Timeout, NotSettled) as e:
        res, err = None, type(e).__name__
    finally:
        signal.alarm(0)
    return fam, k, res, time.monotonic() - t0, err


def write_family(fam, plan, results, dropped):
    M = S()
    sigs = [M.get(sid) for sid in plan]
    inputs = {
        "signal": np.array(plan),
        "digest": np.array([M.digest(s.x) for s in sigs]),
        "n": np.array([s.n for s in sigs], np.int64),
        "x": [np.asarray(s.x).ravel() for s in sigs],
    }
    if fam == "nd":
        inputs["shape"] = [np.array(s.x.shape, np.int64) for s in sigs]
    methods = {}
    order = []
    for k in sorted(results):
        for m in results[k]:
            if m not in order:
                order.append(m)
    for m in order:
        idx = [k for k in sorted(results) if m in results[k]]
        methods[m] = {
            "over": "point",
            "index": np.array(idx, np.int64),
            "hi": [np.asarray(results[k][m][0]).ravel() for k in idx],
            "lo": [np.asarray(results[k][m][1]).ravel() for k in idx],
            "dropped": np.array(sorted(dropped), np.int64),
        }
    from tests import reffile

    reffile.write(
        OUT / f"{fam}.npz",
        **arbref.provenance(target_bits=TARGET_BITS, extra="acb.dft; each entry to 2^-120 of the output's 2-norm"),
        tool=TOOL,
        inputs=inputs,
        methods=methods,
        n_requested=len(plan),
    )


def build(names, jobs):
    import resource

    start = time.monotonic()
    plans = {f: FAMILIES[f]() for f in names}
    # the long points first, so the pool ends together
    tasks = sorted(((f, k, sid) for f in names for k, sid in enumerate(plans[f])), key=lambda t: -S().get(t[2]).n)
    results = {f: {} for f in names}
    dropped = {f: {} for f in names}
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
        for fam, k, res, secs, err in pool.imap_unordered(_point, tasks, chunksize=1):
            if res is None:
                dropped[fam][k] = err
            else:
                results[fam][k] = res
            if secs > 30:
                print(f"  {fam} {plans[fam][k]}: {secs:.0f} s", flush=True)
    for f in names:
        write_family(f, plans[f], results[f], dropped[f])
        size = (OUT / f"{f}.npz").stat().st_size / 2**20
        out = (OUT / f"{f}.npz").relative_to(ROOT)
        print(
            f"{f}: {len(results[f])} of {len(plans[f])} points -> {out} ({size:.1f} MB)",
            flush=True,
        )
        for k, err in sorted(dropped[f].items()):
            print(f"  dropped {plans[f][k]}: {err}", flush=True)
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    print(f"{len(tasks)} points, {(time.monotonic() - start):.0f} s, largest worker RSS {rss:.2f} GB", flush=True)


def main(argv=None):
    from bench import cli

    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="FAMILY", help=", ".join(FAMILIES))
    ap.add_argument("--list", action="store_true", help="list the families and their signals")
    cli.add_jobs(ap, default=JOBS_MAX)
    a = ap.parse_args(argv)
    bad = [f for f in a.names if f not in FAMILIES]
    if bad:
        ap.error(f"unknown family {bad[0]!r} (known: {', '.join(FAMILIES)})")
    if a.jobs > JOBS_MAX:
        ap.error(f"--jobs: at most {JOBS_MAX} for this tool")
    names = a.names or list(FAMILIES)
    if a.list:
        for f in names:
            pts = FAMILIES[f]()
            total = sum(S().get(p).n for p in pts)
            print(
                f"{f} ({len(pts)} signals, {total} entries): " + ", ".join(pts[:12]) + (" ..." if len(pts) > 12 else "")
            )
        return 0
    with arbref.logged("fft_ref"):
        build(names, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
