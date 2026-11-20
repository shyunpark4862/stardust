"""The speed of sd.regression's G7a tests on fitted models against statsmodels 0.15.0 and R
4.6.2 (lmtest, car, stats), standalone (not bench/speed's harness):

    .venv/bin/python bench/regression/g7a/speed.py [--quick] [--no-r]

Each case is timed after a warm-up as the best of several runs, sd at 1 and 10 threads (the
others as they come: numpy's BLAS threads for statsmodels, R's reference BLAS). Only the test
itself is timed: the fits it is called on are made before (a result caches nothing these
methods use, except dfbetas' and VIF's own values, so every timed call is fresh: the cached
fields are cleared). The table goes to stdout and to bench/regression/g7a/results.md.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

import stardust as sd

HERE = pathlib.Path(__file__).parent


def best(fn, reps):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return min(t)


def fresh(r):
    """r with the methods' caches cleared (the fit's own fields stay)."""
    for k in list(r._cache):
        if isinstance(k, tuple) or k in (
            "vif",
            "dfbetas",
            "dffits",
            "thin_q",
            "influence",
            "leverage",
            "wresid_public",
        ):
            del r._cache[k]
    return r


def problems(quick):
    rng = np.random.default_rng(2026)
    sizes = [(10_000, 10)] if quick else [(10_000, 10), (100_000, 10)]
    out = []
    for n, k in sizes:
        X = rng.standard_normal((n, k))
        y = X @ np.linspace(1, -1, k) + rng.standard_normal(n) * (1 + np.abs(X[:, 0]))
        out.append((n, k, X, y))
    return out


R_SCRIPT = r"""
suppressMessages({library(lmtest); library(car)})
D <- commandArgs(TRUE)[1]
X <- as.matrix(read.csv(file.path(D, "X.csv"), header=FALSE)); y <- drop(as.matrix(read.csv(file.path(D, "y.csv"), header=FALSE)))
k <- ncol(X); dat <- data.frame(y=y, X); nm <- names(dat)[-1]
big <- lm(as.formula(paste("y ~", paste(nm, collapse="+"))), data=dat)
small <- lm(as.formula(paste("y ~", nm[1])), data=dat)
Rm <- cbind(0, diag(k))[1:2, ]
# system.time counts milliseconds: repeat the call until a run takes 0.2 s, best of three runs
tm <- function(name, expr) {
  f <- function() eval(expr); f(); N <- 1
  repeat { t <- system.time(for (i in seq_len(N)) f())[["elapsed"]]; if (t >= 0.2 || N >= 2^16) break; N <- N * 4 }
  t <- min(t, sapply(1:2, function(i) system.time(for (i in seq_len(N)) f())[["elapsed"]]))
  cat(name, t / N, "\n")
}
tm("wald_test", quote(linearHypothesis(big, Rm)))
tm("compare_f_test", quote(anova(small, big)))
tm("compare_lr_test", quote(lrtest(small, big)))
tm("anova_typ1", quote(anova(big)))
tm("anova_typ3", quote(drop1(big, test="F")))
tm("vif", quote(vif(big)))
tm("dffits", quote(dffits(big)))
tm("dfbetas", quote(dfbetas(big)))
tm("goldfeld_quandt", quote(gqtest(big)))
tm("reset_test", quote(resettest(big)))
"""


def r_times(X, y):
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        np.savetxt(d / "X.csv", X, delimiter=",", fmt="%.17g")
        np.savetxt(d / "y.csv", y[:, None], delimiter=",", fmt="%.17g")
        (d / "run.R").write_text(R_SCRIPT)
        res = subprocess.run(["Rscript", str(d / "run.R"), str(d)], capture_output=True, text=True)
    out = {}
    for line in res.stdout.splitlines():
        p = line.split()
        if len(p) == 2:
            out[p[0]] = float(p[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-r", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.render:
        render(json.loads((HERE / "results.json").read_text()))
        return 0
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_goldfeldquandt, het_white, linear_reset
    from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

    reg = sd.regression
    rows = []
    for n, k, X, y in problems(args.quick):
        reps = 3 if n >= 100_000 else 5
        ex = sm.add_constant(X)
        s_big = sm.OLS(y, ex).fit()
        s_small = sm.OLS(y, ex[:, :2]).fit()
        Rm = np.zeros((2, k + 1))
        Rm[0, 1] = Rm[1, 2] = 1.0
        yb = (np.random.default_rng(1).uniform(size=n) < 1 / (1 + np.exp(-(X[:, 0] - X[:, 1])))).astype(float)
        g_s = sm.GLM(yb, ex, family=sm.families.Binomial()).fit()
        cases = {
            "t_test": (lambda r, s: r.t_test(Rm), lambda: s_big.t_test(Rm)),
            "wald_test": (lambda r, s: r.wald_test(Rm), lambda: s_big.f_test(Rm)),
            "compare_f_test": (lambda r, s: r.compare_f_test(s), lambda: s_big.compare_f_test(s_small)),
            "compare_lr_test": (lambda r, s: r.compare_lr_test(s), lambda: s_big.compare_lr_test(s_small)),
            "anova_typ1": (lambda r, s: reg.anova(r), None),
            "anova_typ3": (lambda r, s: reg.anova(r, typ=3), None),
            "vif": (lambda r, s: fresh(r).vif, lambda: [variance_inflation_factor(ex, j) for j in range(1, k + 1)]),
            "dffits": (lambda r, s: fresh(r).dffits, lambda: OLSInfluence(s_big).dffits),
            "dfbetas": (lambda r, s: fresh(r).dfbetas, lambda: OLSInfluence(s_big).dfbetas),
            "white_test": (lambda r, s: fresh(r).white_test(), lambda: het_white(s_big.resid, ex)),
            "goldfeld_quandt": (
                lambda r, s: r.goldfeld_quandt(),
                lambda: het_goldfeldquandt(y, ex, result_object=False),
            ),
            "reset_test": (lambda r, s: r.reset_test(), lambda: linear_reset(s_big, power=3, use_f=True)),
            "glm influence": ("glm", lambda: g_s.get_influence(observed=False).dfbetas),
        }
        rt = {} if args.no_r else r_times(X, y)
        for name, (fn, smf) in cases.items():
            res = {}
            for th in (1, 10):
                sd.set_num_threads(th)
                big, small = reg.ols(y, X), reg.ols(y, X[:, :1])
                if fn == "glm":
                    g = reg.glm(yb, X, family="binomial")

                    def call(g=g):
                        fresh(g)
                        return g.dfbetas

                    res[th] = best(call, reps)
                else:
                    res[th] = best(lambda: fn(big, small), reps)
            # statsmodels' OLSInfluence dffits and dfbetas refit the model n times (O(n^2 k^2)):
            # timed at n = 1e4 only
            slow = name in ("dffits", "dfbetas") and n > 10_000
            ts = (
                best(smf, 1 if n >= 10_000 and name in ("dffits", "dfbetas") else reps)
                if (smf is not None and not slow)
                else None
            )
            rows.append((f"{name} ({n}, {k})", res[1], res[10], ts, rt.get(name.replace(" ", "_"))))
            print(rows[-1], flush=True)
    sd.set_num_threads(10)
    (HERE / "results.json").write_text(json.dumps(rows))
    render(rows)


PLANNED = """
## bench/speed·bench/accuracy에 넣을 경우 (그 체계의 주인이 넣는다)

- 속도(`bench/speed`의 regression 경우로; statsmodels·R 비교자는 위와 같다): (n, k) = (10⁴, 10), (10⁵, 10)에서
  `t_test`(두 줄), `wald_test`(두 줄), `compare_f_test`·`compare_lr_test`(작은 모형 열 하나), `anova`(Type I, Type III,
  모형 셋), `vif`, `dffits`, `dfbetas`, `white_test`, `goldfeld_quandt`, `reset_test`, GLM(이항)의 `dfbetas`; 배치 1만 개의
  (50, 3)에서 `t_test`·`wald_test`·`vif`·`dfbetas`. statsmodels의 `OLSInfluence.dffits`·`dfbetas`는 행마다 다시 적합해
  (n = 10⁴에서 14 s) 10⁴에서만 잰다.
- 정확도(`bench/accuracy`의 regression; 참값 tests/regression/ref/g7a.json): 문제마다 `t_test`의 effect·sd·t·p,
  `wald_test`, `compare_f_test`, `compare_lr_test`, Type I·III 표, `vif`, `dffits`, `dfbetas`, `white_test`,
  `goldfeld_quandt`, `reset_test`, GLM의 `t_test`·`wald_test`·`compare_lr_test`·Type I·`leverage`·`cooks_distance`·
  `dffits`·`dfbetas`. 비교자는 statsmodels(같은 이름의 함수)와 R(lmtest, car, stats; tools/gen_regression_g7a_r.py와
  같은 호출).
"""


def render(rows):
    def ms(v):
        return "—" if v is None else f"{v * 1e3:.3g}"

    lines = [
        "| 경우 (n, k) | sd 1스레드 ms | sd 10스레드 ms | statsmodels ms | R ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, a1, a10, s_, r_ in rows:
        lines.append(f"| {name} | {ms(a1)} | {ms(a10)} | {ms(s_)} | {ms(r_)} |")
    text = "\n".join(lines)
    print(text)
    (HERE / "results.md").write_text(
        "# sd.regression G7a 속도\n\n`.venv/bin/python bench/regression/g7a/speed.py`가 만든다(`--render`는 results.json에서 "
        "표만 다시 쓴다). 검정만 잰다: 적합은 미리 한다. 같은 함수가 없으면 —. statsmodels는 numpy의 BLAS 스레드, R은 "
        "기본 BLAS다. 다른 작업이 함께 돈 기계에서 잰 값이라 수 % 흔들린다.\n\n" + text + "\n" + PLANNED
    )


if __name__ == "__main__":
    sys.exit(main())
