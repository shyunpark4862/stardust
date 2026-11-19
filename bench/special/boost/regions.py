"""Boost.Math vs sd.special by region, for the candidates of bench/special/boost/compare.md:
accuracy on the reference points of each region (guaranteed and loose), and single-thread
speed on 2e5 points drawn from that region (sd: one array call; Boost: the C++ loop).

    .venv/bin/python bench/special/boost/regions.py <dir with boost_eval> > bench/special/boost/regions.md
"""

import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/special"))
import common  # noqa: E402

import stardust as sd  # noqa: E402
from tests.special import refdata  # noqa: E402

D = pathlib.Path(sys.argv[1])
EXE = str(D / "boost_eval")
EM = -np.exp(-1.0)

# name: [(label, predicate on the input tuple)]
REGIONS = {
    "hyp1f1": [
        ("guaranteed", None),
        ("loose: a < 0, x > 0", lambda a, b, x: (a < 0) & (x > 0)),
        ("loose: other", lambda a, b, x: ~((a < 0) & (x > 0))),
    ],
    "k0": [("x < 2", lambda x: x < 2), ("2 <= x < 8", lambda x: (x >= 2) & (x < 8)), ("x >= 8", lambda x: x >= 8)],
    "k1": [("x < 2", lambda x: x < 2), ("2 <= x < 8", lambda x: (x >= 2) & (x < 8)), ("x >= 8", lambda x: x >= 8)],
    "i0": [("|x| < 8", lambda x: np.abs(x) < 8), ("|x| >= 8", lambda x: np.abs(x) >= 8)],
    "i1": [("|x| < 8", lambda x: np.abs(x) < 8), ("|x| >= 8", lambda x: np.abs(x) >= 8)],
    "iv": [
        ("|v| < 5", lambda v, x: np.abs(v) < 5),
        ("5 <= |v| < 50", lambda v, x: (np.abs(v) >= 5) & (np.abs(v) < 50)),
        ("|v| >= 50", lambda v, x: np.abs(v) >= 50),
    ],
    "kv": [
        ("|v| < 5", lambda v, x: np.abs(v) < 5),
        ("5 <= |v| < 50", lambda v, x: (np.abs(v) >= 5) & (np.abs(v) < 50)),
        ("|v| >= 50", lambda v, x: np.abs(v) >= 50),
    ],
    "zeta": [("x > 1", lambda x: x > 1), ("0 <= x < 1", lambda x: (x >= 0) & (x < 1)), ("x < 0", lambda x: x < 0)],
    "expi": [
        ("x < -700", lambda x: x < -700),
        ("-700 <= x < 0", lambda x: (x >= -700) & (x < 0)),
        ("0 < x <= 709", lambda x: (x > 0) & (x <= 709)),
        ("x > 709", lambda x: x > 709),
    ],
    "lambertw": [("|x + 1/e| < 0.3", lambda x: np.abs(x - EM) < 0.3), ("elsewhere", lambda x: np.abs(x - EM) >= 0.3)],
    "lambertw_m1": [
        ("|x + 1/e| < 0.3", lambda x: np.abs(x - EM) < 0.3),
        ("elsewhere", lambda x: np.abs(x - EM) >= 0.3),
    ],
}
EXTRA = {"hyp1f1": "hyp1f1_bigneg"}  # extra reference sets (bench/special/boost/extra_refs.py)


def write(path, xs):
    with open(path, "wb") as f:
        np.array([xs[0].size, len(xs)], dtype=np.int64).tofile(f)
        for x in xs:
            np.ascontiguousarray(x, dtype=np.float64).tofile(f)


def boost_eval(name, xs):
    write(D / "in.bin", xs)
    subprocess.run([EXE, "eval", name, str(D / "in.bin"), str(D / "out.bin")], timeout=600, check=True)
    return np.fromfile(D / "out.bin", dtype=np.float64)


def boost_ns(name, xs):
    write(D / "in.bin", xs)
    out = subprocess.run(
        [EXE, "time", name, str(D / "in.bin")], timeout=600, capture_output=True, text=True, check=True
    )
    return float(out.stdout)


def sd_ns(name, xs):
    f = common.sd_function(name)
    sxs = tuple(sd.asarray(x) for x in xs)
    sd.set_num_threads(1)
    ns = common.best_ns(lambda: f(*sxs), repeat=5) / xs[0].size
    sd.set_num_threads(10)
    return ns


def stats(e):
    return (f"{e.max():.3g}", int((e > 45).sum())) if e.size else ("-", 0)


def rows(name, label, xs, hi, lo, sc, m, timing=True):
    f = common.sd_function(name)
    ours = np.asarray(f(*(sd.asarray(x) for x in xs)))
    b = boost_eval(name, xs)
    eo = refdata.ulp_error(ours, hi, lo) / sc
    eb = refdata.ulp_error(b, hi, lo) / sc
    so, sb = stats(eo[m]), stats(eb[m])
    t_sd = t_b = "-"
    if timing and m.sum() > 0:
        idx = np.random.default_rng(0).choice(np.flatnonzero(m), 200_000)
        s = tuple(np.ascontiguousarray(x[idx]) for x in xs)
        t_sd, t_b = sd_ns(name, s), boost_ns(name, s)
        ratio = f"{t_b / t_sd:.2f}"
        t_sd, t_b = f"{t_sd:.1f}", f"{t_b:.1f}"
    else:
        ratio = "-"
    return f"| `{name}` | {label} | {int(m.sum())} | {so[0]} | {so[1]} | {sb[0]} | {sb[1]} | {t_sd} | {t_b} | {ratio} |"


def main():
    print(
        "Boost.Math (boostorg/math develop 4682f44) vs sd.special by region. ULP of the reference "
        "(divided by the file's scale: envelope or kappa); 'over 45' counts points above 45 ULP. "
        "ns per element, one thread, 2e5 points of the region; Boost / sd > 1: sd faster.\n"
    )
    print(
        "| function | region | points | sd max | sd over 45 | Boost max | Boost over 45 | sd ns | Boost ns | Boost / sd |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, regions in REGIONS.items():
        xs, hi, lo, g = refdata.load(name)
        sc = refdata.scale(name)
        for label, pred in regions:
            if pred is None:
                m = g
            else:
                m = pred(*xs) & (g if name != "hyp1f1" else ~g)
            print(rows(name, label, xs, hi, lo, sc, m), flush=True)
        if name in EXTRA:
            p = D / f"{EXTRA[name]}.npz"
            if p.exists():
                with np.load(p) as f:
                    xs2 = tuple(f[f"x{k}"] for k in range(3))
                    hi2, lo2 = f["hi"], f["lo"]
                print(
                    rows(
                        name,
                        "large |a|, a < 0 (extra set)",
                        xs2,
                        hi2,
                        lo2,
                        np.ones(hi2.size),
                        np.ones(hi2.size, dtype=bool),
                        timing=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
