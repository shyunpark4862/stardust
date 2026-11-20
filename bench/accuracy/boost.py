"""Boost.Math as a comparator (bench/accuracy/boost_comparator.cpp), built on first use.

The headers are a checkout of github.com/boostorg/math outside the repository: SD_BOOST_MATH, else
build/comparators/boost-math (git clone --depth 1 https://github.com/boostorg/math.git there).
The library is built into build/comparators/ with the system clang++ (standalone mode, -O2) and
rebuilt when the source changes. Without the checkout, the Boost columns read "not run".
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import subprocess

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = pathlib.Path(__file__).with_name("boost_comparator.cpp")
_LIB = [None]


def headers() -> pathlib.Path | None:
    d = pathlib.Path(os.environ.get("SD_BOOST_MATH", ROOT / "build" / "comparators" / "boost-math"))
    return d if (d / "include" / "boost" / "math").is_dir() else None


def available() -> bool:
    return headers() is not None


def version() -> str | None:
    h = headers()
    if h is None:
        return None
    r = subprocess.run(
        ["git", "-C", str(h), "log", "-1", "--format=%h %cd", "--date=short"], capture_output=True, text=True
    )
    return f"boostorg/math {r.stdout.strip()}" if r.returncode == 0 else "boostorg/math"


def _lib():
    if _LIB[0] is not None:
        return _LIB[0]
    h = headers()
    key = hashlib.sha256(SRC.read_bytes() + str(h).encode()).hexdigest()[:12]
    out = ROOT / "build" / "comparators" / f"libsdboost-{key}.dylib"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        cmd = ["clang++", "-O2", "-std=c++17", "-shared", "-fPIC", "-DBOOST_MATH_STANDALONE",
               f"-I{h / 'include'}", str(SRC), "-o", str(tmp)]  # fmt: skip
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tmp.replace(out)  # several workers may build at once: the last rename wins, each whole
    lib = ctypes.CDLL(str(out))
    lib.sd_boost.restype = ctypes.c_int
    lib.sd_boost.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_long,
                             ctypes.c_void_p]  # fmt: skip
    lib.sd_boost_names.restype = ctypes.c_char_p
    _LIB[0] = lib
    return lib


def names() -> set:
    return set(_lib().sd_boost_names().decode().split())


def _call(name, arrs):
    n = arrs[0].size
    out = np.empty(n)
    ptrs = (ctypes.c_void_p * len(arrs))(*(a.ctypes.data for a in arrs))
    rc = _lib().sd_boost(name.encode(), len(arrs), ptrs, n, out.ctypes.data)
    if rc != 0:
        raise ValueError(f"boost: {name} ({'unknown' if rc == -1 else 'wrong argument count'})")
    return out


def evaluate(name: str, *args, seconds=30.0) -> np.ndarray:
    """name over the broadcast args, in a forked child that is killed after `seconds`: a Boost
    call is C++ that no Python signal interrupts, and a few never finish in useful time (the
    non-central chi-squared mode searches without end at some parameters)."""
    import tempfile
    import time

    arrs = [np.ascontiguousarray(a, dtype=np.float64) for a in np.broadcast_arrays(*(np.asarray(a) for a in args))]
    _lib()  # build and load in the parent
    with tempfile.TemporaryDirectory(prefix="sd-boost-") as d:
        path = pathlib.Path(d) / "out.npy"
        pid = os.fork()
        if pid == 0:  # the child: compute, write, leave without running the parent's cleanup
            code = 1
            try:
                np.save(path, _call(name, arrs))
                code = 0
            except ValueError:
                code = 2
            finally:
                os._exit(code)
        end = time.monotonic() + seconds
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() > end:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
                raise TimeoutError(f"boost: {name} over {seconds:g} s")
            time.sleep(0.01)
        if os.waitstatus_to_exitcode(status) == 2:
            raise ValueError(f"boost: {name} (unknown or wrong argument count)")
        if os.waitstatus_to_exitcode(status) != 0:
            raise RuntimeError(f"boost: {name} failed")
        return np.load(path).reshape(arrs[0].shape)
