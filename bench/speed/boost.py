"""Boost.Math as a speed comparator (bench/speed/boost_speed.cpp), built on first use.

The headers, the flags and the build are the accuracy comparator's (bench/accuracy/boost.py,
build: SD_BOOST_MATH, else build/comparators/boost-math; FLAGS -O3 -mcpu=native -ffp-contract=off
-std=c++17); the library is rebuilt when the source, the flags or the checkout's commit change.
Nothing is built until a comparator is first called: listing the cases needs only the static
name tables (BOOST_SPECIAL and BOOST_DISTS in bench/accuracy/compare.py).

A comparator's callable makes one ctypes call per array: the loop over the elements runs in
C++ (one inlined Boost call per element), so the Python cost is one call per array, as for sd
and scipy. It returns its preallocated output (also its `out` attribute), which the smoke
run's results check reads. Without the checkout the Boost columns are left out.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

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
    from bench.accuracy import boost as acc

    lib = ctypes.CDLL(str(acc.build(SRC, "libsdboostspeed")))
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

    def call(lib=lib, cname=cname, k=len(arrs), ptrs=ptrs, n=n, o=o, out=out, keep=arrs):
        lib.sdb_special(cname, k, ptrs, n, o)
        return out

    call.out = out
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

    def call(fn=fn, cd=cd, cm=cm, pp=pp, k=len(params), px=px, n=n, o=o, out=out, keep=keep):
        fn(cd, cm, pp, k, px, n, o)
        return out

    call.out = out
    return call
