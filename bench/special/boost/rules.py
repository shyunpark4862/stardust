# bench/special/boost: run as .venv/bin/python bench/special/boost/rules.py <dir with boost_eval> (from the repository root); the candidate rules of routing.md
# Try routing rules: use Boost where rule(inputs, boost_value) is True, ours elsewhere; report
# contract violations of the combination, and the share of points routed to Boost.
import sys, numpy as np, subprocess
sys.path.insert(0, ".")
from tests.special import refdata
from tests.special.test_reference import contract_error, LOOSE
sys.path.insert(0, "bench/special"); import common
import stardust as sd
D = sys.argv[1]
def write(p, xs):
    with open(p, "wb") as f:
        np.array([xs[0].size, len(xs)], dtype=np.int64).tofile(f)
        for x in xs: np.ascontiguousarray(x, dtype=np.float64).tofile(f)
def errs(name, v, xs, hi, lo, g):
    sc = refdata.scale(name)
    e = contract_error(v, hi, lo) / sc / 1e-14
    if name in refdata.MIXED: e = np.where(g, e, refdata.mixed_error(v, hi, lo) / 1e-14)
    elif (~g).any(): e = np.where(g, e, e * 1e-14 / LOOSE.get(name, 1e-14))
    return e
tiny = lambda v: ~(np.abs(v) >= 1e-300) | ~np.isfinite(v)
RULES = {
 "airy_ai": [("x>-1e1", lambda x, v: x > -10), ("-20<x<100", lambda x, v: (x > -20) & (x < 100)), ("-15<x<100", lambda x, v: (x > -15) & (x < 100))],
 "airy_bi": [("-15<x<100", lambda x, v: (x > -15) & (x < 100)), ("-15<x<10", lambda x, v: (x > -15) & (x < 10))],
 "digamma": [("x>0", lambda x, v: x > 0)],
 "zeta": [("x>-60", lambda x, v: x > -60), ("x>=0", lambda x, v: x >= 0)],
 "beta": [("a,b>0", lambda a, b, v: (a > 0) & (b > 0)), ("a,b>0 & a+b<100", lambda a, b, v: (a > 0) & (b > 0) & (a + b < 100)),
          ("a,b>0 & a,b<50", lambda a, b, v: (a > 0) & (b > 0) & (a < 50) & (b < 50)), ("a,b>0 & a+b<40", lambda a, b, v: (a > 0) & (b > 0) & (a + b < 40))],
 "gammaincinv": [("result normal", lambda a, p, v: ~tiny(v))],
 "iv": [("|v|<300", lambda n, x, v: np.abs(n) < 300), ("|v|<300 & result>=1e-150", lambda n, x, v: (np.abs(n) < 300) & (np.abs(v) >= 1e-150)),
        ("result>=1e-150", lambda n, x, v: np.abs(v) >= 1e-150), ("|v|<50", lambda n, x, v: np.abs(n) < 50)],
 "kv": [("|v|<400", lambda n, x, v: np.abs(n) < 400), ("|v|<300", lambda n, x, v: np.abs(n) < 300)],
 "hyp1f1": [("loose", None), ("loose | x>=0", "x>=0"), ("loose | (x>=0 & b<11)", "b<11")],
 "betainc": [("a,b<=9 & normal", lambda a, b, x, v: (a <= 9) & (b <= 9) & ~tiny(v)), ("a,b<=5 & normal", lambda a, b, x, v: (a <= 5) & (b <= 5) & ~tiny(v))],
 "betaincc": [("a,b<=6 & normal", lambda a, b, x, v: (a <= 6) & (b <= 6) & ~tiny(v)), ("a,b<=5 & normal", lambda a, b, x, v: (a <= 5) & (b <= 5) & ~tiny(v))],
 "betaincinv": [("normal", lambda a, b, p, v: ~tiny(v)), ("a,b>=1 & normal", lambda a, b, p, v: (a >= 1) & (b >= 1) & ~tiny(v)),
                ("a,b>=10 & normal", lambda a, b, p, v: (a >= 10) & (b >= 10) & ~tiny(v))],
 "betainccinv": [("normal", lambda a, b, p, v: ~tiny(v)), ("a,b>=1 & normal", lambda a, b, p, v: (a >= 1) & (b >= 1) & ~tiny(v))],
}
for name, rules in RULES.items():
    xs, hi, lo, g = refdata.load(name)
    write(D + "/in.bin", xs); subprocess.run([D + "/boost_eval", "eval", name, D + "/in.bin", D + "/out.bin"], check=True)
    b = np.fromfile(D + "/out.bin")
    ours = np.asarray(common.sd_function(name)(*(sd.asarray(x) for x in xs)))
    for label, rule in rules:
        if name == "hyp1f1":
            a, bb, x = xs
            m = ~g
            if rule == "x>=0": m = m | (x >= 0)
            if rule == "b<11": m = m | ((x >= 0) & (bb < 11))
        else:
            m = rule(*xs, b)
        v = np.where(m, b, ours)
        e = errs(name, v, xs, hi, lo, g)
        bad = e > 1
        worst = np.argmax(np.where(np.isfinite(e), e, 1e300))
        print(f"{name:12s} {label:28s} boost share {m.mean():5.1%}  violations {bad.sum():4d}  worst {e.max():.3g}x at {tuple(f'{x[worst]:.4g}' for x in xs)}", flush=True)
