"""Boost.Math as a speed comparator (bench/speed/boost_speed.cpp), built on first use.

The headers are the checkout the accuracy comparator uses (bench/accuracy/boost.py: SD_BOOST_MATH,
else build/comparators/boost-math). The library is built into build/comparators/ with the system
clang++ and the accuracy comparator's flags (bench.accuracy.boost.FLAGS: sd's own -O3 -mcpu=native
-ffp-contract=off, CMakeLists.txt), and rebuilt when the source changes. A comparator's callable makes one ctypes call per array: the loop
over the elements runs in C++ (one inlined Boost call per element), so the Python cost is one call
per array, as for sd and scipy. Without the checkout the Boost columns are left out.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import subprocess

import numpy as np

from bench.accuracy.boost import FLAGS  # one set of build flags for both comparators (sd's own)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = pathlib.Path(__file__).with_name("boost_speed.cpp")
_LIB = [None]


def headers() -> pathlib.Path | None:
    from bench.accuracy import boost as acc

    return acc.headers()


def available() -> bool:
    return headers() is not None


def version() -> str | None:
    from bench.accuracy import boost as acc

    return acc.version()


def _lib():
    if _LIB[0] is not None:
        return _LIB[0]
    h = headers()
    key = hashlib.sha256(SRC.read_bytes() + str(h).encode() + " ".join(FLAGS).encode()).hexdigest()[:12]
    out = ROOT / "build" / "comparators" / f"libsdboostspeed-{key}.dylib"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        cmd = ["clang++", *FLAGS, "-shared", "-fPIC", "-DBOOST_MATH_STANDALONE", f"-I{h / 'include'}", str(SRC),
               "-o", str(tmp)]  # fmt: skip
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tmp.replace(out)  # several workers may build at once: the last rename wins, each whole
    lib = ctypes.CDLL(str(out))
    P = ctypes.POINTER(ctypes.c_double)
    PP = ctypes.POINTER(P)
    lib.sdb_special.restype = ctypes.c_int
    lib.sdb_special.argtypes = [ctypes.c_char_p, ctypes.c_int, PP, ctypes.c_long, P]
    lib.sdb_dist.restype = ctypes.c_int
    lib.sdb_dist.argtypes = [ctypes.c_char_p, ctypes.c_char_p, P, ctypes.c_int, P, ctypes.c_long, P]
    lib.sdb_dist_arrays.restype = ctypes.c_int
    lib.sdb_dist_arrays.argtypes = [ctypes.c_char_p, ctypes.c_char_p, PP, ctypes.c_int, P, ctypes.c_long, P]
    lib.sdb_names.restype = ctypes.c_char_p
    _LIB[0] = lib
    return lib


def names() -> set:
    return set(_lib().sdb_names().decode().split())


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def special(name: str, args):
    """A zero-argument callable evaluating scipy.special's `name` over the arrays `args` (all
    of one size) into a preallocated output; it raises on an unknown name."""
    lib = _lib()
    arrs = [_f64(a) for a in args]
    n = arrs[0].size
    out = np.empty(n)
    ptrs = (ctypes.POINTER(ctypes.c_double) * len(arrs))(*(_ptr(a) for a in arrs))
    cname, o = name.encode(), _ptr(out)
    rc = lib.sdb_special(cname, len(arrs), ptrs, n, o)
    if rc != 0:
        raise ValueError(f"boost: {name} ({'unknown' if rc == -1 else 'wrong argument count'})")

    def call(lib=lib, cname=cname, k=len(arrs), ptrs=ptrs, n=n, o=o, keep=(arrs, out)):
        return lib.sdb_special(cname, k, ptrs, n, o)

    return call


def dist(dist: str, method: str, params, x):
    """A zero-argument callable evaluating `method` of distribution `dist` (the accuracy
    comparator's names: "normal", "negativebinomial", ...; sd's parameters in order) at the
    points x. Scalar parameters build the distribution once per call; arrays of x's size build
    it per element."""
    lib = _lib()
    xs = _f64(x)
    n = xs.size
    out = np.empty(n)
    cd, cm, o, px = dist.encode(), method.encode(), _ptr(out), _ptr(xs)
    if all(np.ndim(p) == 0 for p in params):
        ps = np.array([float(p) for p in params])
        fn, pp = lib.sdb_dist, _ptr(ps)
        keep = (xs, out, ps)
    else:
        arrs = [_f64(np.broadcast_to(p, xs.shape)).ravel() for p in params]
        fn, pp = lib.sdb_dist_arrays, (ctypes.POINTER(ctypes.c_double) * len(arrs))(*(_ptr(a) for a in arrs))
        keep = (xs, out, arrs)
    rc = fn(cd, cm, pp, len(params), px, n, o)
    if rc != 0:
        raise ValueError(f"boost: {dist}:{method} ({'unknown' if rc == -1 else 'wrong parameter count'})")

    def call(fn=fn, cd=cd, cm=cm, pp=pp, k=len(params), px=px, n=n, o=o, keep=keep):
        return fn(cd, cm, pp, k, px, n, o)

    return call
