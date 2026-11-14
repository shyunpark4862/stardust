# Boost.Math vs sd.special (bench/special/boost/compare.md): accuracy on the reference files (guaranteed region, the same
# error measure as bench/special/accuracy.py) and single-thread speed on the same 10^6-sample
# as bench/special/speed.py. Prints one line per function at once; writes results.json.
#   clang++ -O3 -std=c++20 -I<boostorg/math>/include boost_eval.cpp -o <dir>/boost_eval
#   .venv/bin/python bench/special/boost/drive.py <dir> [NAME ...]
import json, subprocess, sys, time, pathlib
import numpy as np
D = pathlib.Path(sys.argv[1])  # a scratch directory holding boost_eval (built from boost_eval.cpp)
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/special"))
from tests.special import refdata
import common
import stardust as sd
EXE = str(D / "boost_eval")
names = subprocess.run([EXE, "list"], capture_output=True, text=True).stdout.split()
names = sys.argv[2:] or names
ours_speed = {}
for l in open(ROOT / "bench/special/final/speed.md"):
    if l.startswith("| `"):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        ours_speed[c[0].strip("`")] = float(c[2]) * 1e6 / 1e6  # ms per 1e6 = ns per element
def write(path, xs):
    with open(path, "wb") as f:
        np.array([xs[0].size, len(xs)], dtype=np.int64).tofile(f)
        for x in xs: np.ascontiguousarray(x, dtype=np.float64).tofile(f)
res = {}
for name in names:
    xs, hi, lo, g = refdata.load(name)
    sc = refdata.scale(name)
    write(D / "in.bin", xs)
    t0 = time.time()
    try:
        subprocess.run([EXE, "eval", name, str(D / "in.bin"), str(D / "out.bin")], timeout=300, check=True)
    except subprocess.TimeoutExpired:
        print(f"{name}: eval TIMEOUT (>300 s)", flush=True); res[name] = {"timeout": True}; continue
    b = np.fromfile(D / "out.bin", dtype=np.float64)
    ours = np.asarray(common.sd_function(name)(*(sd.asarray(x) for x in xs)))
    eb = refdata.ulp_error(b, hi, lo) / sc
    eo = refdata.ulp_error(ours, hi, lo) / sc
    m = g if g.any() else np.ones_like(g)
    finite_ref = np.isfinite(hi)
    nonfinite_b = (~np.isfinite(b) & finite_ref & m).sum()
    fb = m & np.isfinite(b) & finite_ref
    r = {"n": int(m.sum()), "boost_max_finite": float(eb[fb].max()) if fb.any() else None,
         "boost_p999_finite": float(np.quantile(eb[fb], .999)) if fb.any() else None,
         "boost_max_all": float(eb[m].max()), "boost_nonfinite": int(nonfinite_b),
         "boost_over45": int((eb[m] > 45).sum()), "ours_max": float(eo[m].max()), "eval_s": round(time.time() - t0, 1)}
    # speed: the same sample as speed.py
    s = common.sample_inputs(name, 10**6)
    write(D / "in.bin", s)
    try:
        out = subprocess.run([EXE, "time", name, str(D / "in.bin")], timeout=300, capture_output=True, text=True, check=True)
        r["boost_ns"] = float(out.stdout.strip())
    except subprocess.TimeoutExpired:
        r["boost_ns"] = None
    r["ours_ns"] = ours_speed.get(name)
    res[name] = r
    print(f"{name}: n={r['n']} sd max {r['ours_max']:.3g} | boost max(finite) {r['boost_max_finite']} >45ULP {r['boost_over45']} nonfinite {r['boost_nonfinite']} | ns sd {r['ours_ns']} boost {r['boost_ns']}", flush=True)
    json.dump(res, open(D / "results.json", "w"), indent=1)
