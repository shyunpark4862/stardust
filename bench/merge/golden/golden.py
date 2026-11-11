"""Golden outputs: the merged library must give the old packages' results, bit for bit
(ROADMAP.md, safety net 2).

    .venv-legacy/bin/python bench/merge/golden/golden.py record AREA          # old packages -> AREA.json
    .venv/bin/python        bench/merge/golden/golden.py check AREA           # stardust vs AREA.json
    ... golden.py record AREA --out other.json                               # e.g. to check record is repeatable

AREA is array, linalg or frame. Each area's cases live in cases_<AREA>.py:

    def cases(L):          # yields (name, thunk); thunk() returns a result or raises
    def encode(x, L):      # optional: a JSON-able form of an area-specific object, or None

L gives the libraries under test, so one case file runs against both the old
packages and stardust:
    L.A   array module   (stardust_numpy | stardust)
    L.LA  linalg module  (stardust_linalg | stardust.linalg)
    L.F   frame module   (stardust_pandas | stardust.frame)
    L.np  numpy, L.rng(seed) a numpy Generator, L.arr(np_array) -> L.A array

Every case runs at 1, 3 and 10 threads. A result is stored as a SHA-256 of
its bytes plus dtype, shape and strides (so a layout change is a mismatch
too); errors as their type and message. Messages are compared after the old
package names are mapped to the new ones, and a difference only in the
message is reported apart from real mismatches.
"""

import argparse
import hashlib
import importlib
import json
import math
import pathlib
import sys
import traceback

import numpy as np

HERE = pathlib.Path(__file__).parent
THREADS = (1, 3, 10)
AREAS = ("array", "linalg", "frame")

# Old spellings in error messages and type names -> the merged library's
# (module names, the import aliases messages use, project names in messages).
RENAMES = [
    ("stardust_linalg", "stardust.linalg"),
    ("stardust_pandas", "stardust.frame"),
    ("stardust_numpy", "stardust"),
    ("stardust-numpy", "stardust"),
    ("sdnp.", "sd."),
]


def normalize(text):
    for old, new in RENAMES:
        text = text.replace(old, new)
    return text


class Libs:
    def __init__(self, lib, area):
        self.np = np
        self.A = self.LA = self.F = None
        if lib == "legacy":
            if area in ("array", "linalg"):
                self.A = importlib.import_module("stardust_numpy")
            if area == "linalg":
                self.LA = importlib.import_module("stardust_linalg")
            if area == "frame":
                self.F = importlib.import_module("stardust_pandas")
            # stardust-linalg follows stardust_numpy's thread count.
            self._threads = self.F if area == "frame" else self.A
        else:
            sd = importlib.import_module("stardust")
            self.A = sd
            if area == "linalg":
                self.LA = importlib.import_module("stardust.linalg")
            if area == "frame":
                self.F = importlib.import_module("stardust.frame")
            self._threads = sd

    def set_threads(self, n):
        self._threads.set_num_threads(n)

    @staticmethod
    def rng(seed):
        return np.random.Generator(np.random.PCG64(seed))

    def arr(self, a):
        return self.A.asarray(np.asarray(a))


def sha(b):
    return hashlib.sha256(b).hexdigest()[:24]


def float_key(v):
    return float(v).hex() if not math.isnan(v) else "nan:" + np.float64(v).tobytes().hex()


def encode(x, L, area_encode):
    """A JSON-able description of a result, exact to the bit."""
    if isinstance(x, BaseException):
        return {"error": type(x).__name__, "msg": normalize(str(x))}
    if area_encode is not None:
        e = area_encode(x, L)
        if e is not None:
            return e
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return {"int": str(x)}
    if isinstance(x, float):
        return {"float": float_key(x)}
    if isinstance(x, complex):
        return {"complex": [float_key(x.real), float_key(x.imag)]}
    if L.A is not None and isinstance(x, L.A.ndarray):
        a = np.asarray(x)  # zero-copy through the buffer protocol
        return {
            "ndarray": normalize(str(x.dtype)),
            "shape": list(x.shape),
            "strides": list(x.strides),
            "sha": sha(np.ascontiguousarray(a).tobytes()),
        }
    if L.A is not None and isinstance(x, type(L.A.float64)):  # a dtype object
        return {"dtype": normalize(str(x))}
    if isinstance(x, tuple):
        return {"tuple": type(x).__name__, "items": [encode(v, L, area_encode) for v in x]}
    if isinstance(x, list):
        return {"list": [encode(v, L, area_encode) for v in x]}
    if isinstance(x, dict):
        return {"dict": {str(k): encode(v, L, area_encode) for k, v in x.items()}}
    raise TypeError(f"golden: no encoding for {normalize(type(x).__module__)}.{type(x).__name__}")


def run_area(lib, area):
    L = Libs(lib, area)
    sys.path.insert(0, str(HERE))
    mod = importlib.import_module(f"cases_{area}")
    area_encode = getattr(mod, "encode", None)
    names = [name for name, _ in mod.cases(L)]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"duplicate case names: {dup[:10]}")
    out = {}
    for n in THREADS:
        L.set_threads(n)
        for name, thunk in mod.cases(L):  # fresh inputs for every thread count
            try:
                r = thunk()
            except Exception as e:  # noqa: BLE001 -- errors are results too
                r = e
            try:
                out.setdefault(name, {})[str(n)] = encode(r, L, area_encode)
            except Exception:  # noqa: BLE001
                raise SystemExit(f"case {name!r}: cannot encode its result\n{traceback.format_exc()}")
    return out


def strip_msgs(e):
    if isinstance(e, dict):
        return {k: strip_msgs(v) for k, v in e.items() if k != "msg"}
    if isinstance(e, list):
        return [strip_msgs(v) for v in e]
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("area", choices=AREAS)
    ap.add_argument("--out", help="record: output file (default: <area>.json next to this script)")
    ap.add_argument("--golden", help="check: golden file (default: <area>.json next to this script)")
    args = ap.parse_args()
    default = HERE / f"{args.area}.json"

    if args.mode == "record":
        res = run_area("legacy", args.area)
        # The old packages promise results independent of the thread count;
        # store one result per case and flag any case that breaks the promise.
        golden, unstable = {}, []
        for name, per in res.items():
            vals = list(per.values())
            if any(v != vals[0] for v in vals[1:]):
                unstable.append(name)
                golden[name] = {"per_threads": per}
            else:
                golden[name] = vals[0]
        path = pathlib.Path(args.out) if args.out else default
        path.write_text(json.dumps({"cases": golden, "unstable": unstable}, indent=1, sort_keys=True) + "\n")
        nerr = sum(1 for v in golden.values() if isinstance(v, dict) and "error" in v)
        print(f"{args.area}: {len(golden)} cases ({nerr} expected errors) -> {path}")
        if unstable:
            print(f"  results that depend on the thread count in the OLD package: {unstable}")
        return

    golden = json.loads(pathlib.Path(args.golden or default).read_text())["cases"]
    res = run_area("new", args.area)
    bad, msg_only, missing = [], [], sorted(set(golden) - set(res))
    for name, per in res.items():
        if name not in golden:
            bad.append((name, "-", "case not in the golden file"))
            continue
        want_all = golden[name]  # None, bool and str are results too
        per_threads = isinstance(want_all, dict) and "per_threads" in want_all
        for n, got in per.items():
            want = want_all["per_threads"][n] if per_threads else want_all
            if got == want:
                continue
            if strip_msgs(got) == strip_msgs(want):
                msg_only.append((name, n, got, want))
            else:
                bad.append((name, n, f"got {got}\n      want {want}"))
    for name, n, why in bad:
        print(f"MISMATCH {name} [threads {n}]: {why}")
    for name, n, got, want in msg_only:
        print(f"MESSAGE  {name} [threads {n}]: got {got.get('msg')!r} want {want.get('msg')!r}")
    for name in missing:
        print(f"MISSING  {name}")
    total = len(golden)
    print(f"{args.area}: {total} golden cases, {len(bad)} mismatches, {len(msg_only)} message-only, {len(missing)} missing")
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
