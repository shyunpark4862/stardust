"""H0: the speed baseline of sd.htest, scipy 1.18.1's (and statsmodels 0.15.0's) time on
representative tests of every milestone (docs/design/htest.md §5). Results: bench/htest/h0/speed.md
(the raw output in speed.json). A standalone script until the area joins bench/speed (the plan is
in docs/design/htest.md §7).

Sizes: one sample (or each of two or three samples) of n = 10, 10^3 and 10^6 standard normal
values (the second sample shifted by 0.1; ties only where the test's data are counts or ranks), and
a batch of 10^4 lanes of 30 (axis=-1) where scipy takes an axis. Tables are fixed: 2 x 2 for
fisher_exact, barnard_exact and boschloo_exact, 5 x 5 for chi2_contingency. A larger size is left
out when the size before took more than 5 s.

Time is the best of up to 5 runs (fewer when the runs pass 2 s), in ms, with numpy's and
Accelerate's threads at 1 (VECLIB_MAXIMUM_THREADS=1, OMP_NUM_THREADS=1): scipy's tests run their
loops on one thread. Warnings are silenced (the comparison is the time).

Run: .venv/bin/python bench/htest/h0/speed.py
"""

import json
import os
import pathlib
import platform
import sys
import time
import warnings

HERE = pathlib.Path(__file__).parent
SIZES = (10, 1000, 10**6)
BATCH = (10**4, 30)


def best(fn, reps=5, budget=2.0):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
        if sum(out) > budget:
            break
    return min(out) * 1e3


def cases():
    """(milestone, test, kind, make) where make(n) returns the call to time (kind 'n' for sizes,
    'batch' for the (10^4, 30) batch, 'fixed' for a fixed input)."""
    import numpy as np
    import scipy.stats as st
    from statsmodels.stats import multitest as mt
    from statsmodels.stats import proportion as pr

    g = np.random.Generator(np.random.PCG64(20261120))

    def x(n, loc=0.0):
        return g.normal(loc, 1.0, size=n)

    def two(n):
        return x(n), x(n, 0.1)

    def b(loc=0.0):
        return g.normal(loc, 1.0, size=BATCH)

    out = []

    def add(ms, name, f, batch=None, cap=None):
        out.append((ms, name, "n", f, cap))
        if batch is not None:
            out.append((ms, name, "batch", batch, None))

    # H1
    add(
        "H1",
        "ttest_1samp",
        lambda n: lambda a=x(n): st.ttest_1samp(a, 0.0),
        lambda: lambda a=b(): st.ttest_1samp(a, 0.0, axis=-1),
    )  # noqa: E501
    add(
        "H1",
        "ttest_ind",
        lambda n: lambda ab=two(n): st.ttest_ind(*ab),
        lambda: lambda a=b(), c=b(0.1): st.ttest_ind(a, c, axis=-1),
    )  # noqa: E501
    add("H1", "ttest_ind (Welch)", lambda n: lambda ab=two(n): st.ttest_ind(*ab, equal_var=False))
    add(
        "H1",
        "ttest_rel",
        lambda n: lambda ab=two(n): st.ttest_rel(*ab),
        lambda: lambda a=b(), c=b(0.1): st.ttest_rel(a, c, axis=-1),
    )  # noqa: E501
    add(
        "H1",
        "f_oneway (3)",
        lambda n: lambda s=(x(n), x(n, 0.1), x(n, 0.2)): st.f_oneway(*s),
        lambda: lambda a=b(), c=b(0.1), d=b(0.2): st.f_oneway(a, c, d, axis=-1),
    )  # noqa: E501
    add("H1", "bartlett (3)", lambda n: lambda s=(x(n), x(n), x(n)): st.bartlett(*s))
    add("H1", "levene (3)", lambda n: lambda s=(x(n), x(n), x(n)): st.levene(*s))
    # H2
    add(
        "H2",
        "chisquare",
        lambda n: lambda f=g.integers(50, 150, size=n).astype(float): st.chisquare(f),
        lambda: lambda f=g.integers(50, 150, size=BATCH).astype(float): st.chisquare(f, axis=-1),
    )  # noqa: E501
    add(
        "H2",
        "jarque_bera",
        lambda n: lambda a=x(n): st.jarque_bera(a),
        lambda: lambda a=b(): st.jarque_bera(a, axis=-1),
    )  # noqa: E501
    add("H2", "normaltest", lambda n: lambda a=x(max(n, 20)): st.normaltest(a))
    add("H2", "shapiro", lambda n: lambda a=x(n): st.shapiro(a), lambda: lambda a=b(): st.shapiro(a, axis=-1))
    add("H2", "anderson (interpolate)", lambda n: lambda a=x(n): st.anderson(a, method="interpolate"))
    add("H2", "ks_1samp (norm)", lambda n: lambda a=x(n): st.ks_1samp(a, st.norm.cdf))
    add("H2", "cramervonmises (norm)", lambda n: lambda a=x(n): st.cramervonmises(a, "norm"))
    # H3
    add(
        "H3",
        "mannwhitneyu",
        lambda n: lambda ab=two(n): st.mannwhitneyu(*ab),
        lambda: lambda a=b(), c=b(0.1): st.mannwhitneyu(a, c, axis=-1),
    )  # noqa: E501
    add(
        "H3", "wilcoxon", lambda n: lambda a=x(n, 0.1): st.wilcoxon(a), lambda: lambda a=b(0.1): st.wilcoxon(a, axis=-1)
    )  # noqa: E501
    add("H3", "ranksums", lambda n: lambda ab=two(n): st.ranksums(*ab))
    add("H3", "kruskal (3)", lambda n: lambda s=(x(n), x(n, 0.1), x(n, 0.2)): st.kruskal(*s))
    add("H3", "brunnermunzel", lambda n: lambda ab=two(n): st.brunnermunzel(*ab))
    add("H3", "ansari", lambda n: lambda ab=two(n): st.ansari(*ab))
    # H4
    add(
        "H4",
        "ks_2samp",
        lambda n: lambda ab=two(n): st.ks_2samp(*ab),
        lambda: lambda a=b(), c=b(0.1): st.ks_2samp(a, c, axis=-1),
    )  # noqa: E501
    add("H4", "cramervonmises_2samp", lambda n: lambda ab=two(n): st.cramervonmises_2samp(*ab))
    add(
        "H4", "anderson_ksamp (2)", lambda n: lambda ab=two(n): st.anderson_ksamp(list(ab), variant="midrank"), cap=1000
    )  # noqa: E501
    add("H4", "epps_singleton_2samp", lambda n: lambda ab=two(max(n, 5)): st.epps_singleton_2samp(*ab))
    # H5
    add(
        "H5",
        "pearsonr",
        lambda n: lambda ab=two(n): st.pearsonr(*ab),
        lambda: lambda a=b(), c=b(): st.pearsonr(a, c, axis=-1),
    )  # noqa: E501
    add(
        "H5",
        "spearmanrho",
        lambda n: lambda ab=two(n): st.spearmanrho(*ab),
        lambda: lambda a=b(), c=b(): st.spearmanrho(a, c, axis=-1),
    )  # noqa: E501
    add(
        "H5",
        "kendalltau",
        lambda n: lambda ab=two(n): st.kendalltau(*ab),
        lambda: lambda a=b(), c=b(): st.kendalltau(a, c, axis=-1),
    )  # noqa: E501
    add("H5", "weightedtau", lambda n: lambda ab=two(n): st.weightedtau(*ab))
    add("H5", "chatterjeexi", lambda n: lambda ab=two(n): st.chatterjeexi(*ab))
    # H6
    t22 = np.array([[3, 7], [9, 2]])
    t55 = g.integers(5, 40, size=(5, 5))
    out.append(("H6", "fisher_exact 2x2", "fixed", lambda: lambda: st.fisher_exact(t22), None))
    out.append(("H6", "chi2_contingency 5x5", "fixed", lambda: lambda: st.chi2_contingency(t55), None))
    out.append(("H6", "barnard_exact 2x2", "fixed", lambda: lambda: st.barnard_exact(t22), None))
    out.append(("H6", "boschloo_exact 2x2", "fixed", lambda: lambda: st.boschloo_exact(t22), None))
    # H7
    add("H7", "binomtest (k = 0.45 n)", lambda n: lambda: st.binomtest(int(0.45 * n), n))
    add("H7", "proportions_ztest (2)", lambda n: lambda: pr.proportions_ztest([0.4 * n, 0.5 * n], [n, n]))
    add("H7", "proportion_confint (wilson)", lambda n: lambda: pr.proportion_confint(0.4 * n, n, method="wilson"))
    # H8
    ps = lambda n: g.uniform(size=n) ** 3  # noqa: E731
    add(
        "H8",
        "false_discovery_control (bh)",
        lambda n: lambda p=ps(n): st.false_discovery_control(p),
        lambda: lambda p=g.uniform(size=BATCH): st.false_discovery_control(p, axis=-1),
    )  # noqa: E501
    add("H8", "multipletests (holm-sidak)", lambda n: lambda p=ps(n): mt.multipletests(p))
    add("H8", "multipletests (hommel)", lambda n: lambda p=ps(n): mt.multipletests(p, method="hommel"), cap=1000)
    add("H8", "combine_pvalues (fisher)", lambda n: lambda p=ps(n): st.combine_pvalues(p))
    add("H8", "tukey_hsd (3)", lambda n: lambda s=(x(n), x(n, 0.1), x(n, 0.2)): st.tukey_hsd(*s))
    return out


def main():
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import scipy
    import statsmodels

    warnings.simplefilter("ignore")
    rows = []
    for ms, name, kind, make, cap in cases():
        if kind == "n":
            for n in SIZES:
                if cap is not None and n > cap:
                    rows.append({"milestone": ms, "test": name, "size": n, "ms": None, "note": "left out"})
                    continue
                t = best(make(n))
                rows.append({"milestone": ms, "test": name, "size": n, "ms": t})
                print(f"{ms} {name} n={n}: {t:.3g} ms", flush=True)
                if t > 5000:
                    cap = n
        else:
            t = best(make())
            size = "10^4 x 30" if kind == "batch" else "fixed"
            rows.append({"milestone": ms, "test": name, "size": size, "ms": t})
            print(f"{ms} {name} {size}: {t:.3g} ms", flush=True)
    env = {
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "machine": platform.machine(),
    }
    (HERE / "speed.json").write_text(json.dumps({"env": env, "rows": rows}, indent=1) + "\n")
    write_md(env, rows)


def fmt(v):
    if v is None:
        return "–"
    return f"{v:.3g}" if v < 1000 else f"{v:,.0f}"


def write_md(env, rows):
    tests = []
    for r in rows:
        key = (r["milestone"], r["test"])
        if key not in tests:
            tests.append(key)
    by = {(r["milestone"], r["test"], r["size"]): r for r in rows}
    lines = [
        "# sd.htest H0 속도 기준선 (scipy·statsmodels)",
        "",
        f"scipy {env['scipy']}, statsmodels {env['statsmodels']}, numpy {env['numpy']}, Python {env['python']}, "
        f"{env['machine']}, 1스레드. 단위는 ms이고 호출 하나의 가장 좋은 시간(최대 5번)이다. "
        "표본은 표준정규(두 번째 표본은 0.1 이동), 배치는 10⁴줄 × 30(axis=-1)이다. 만드는 스크립트는 "
        "bench/htest/h0/speed.py, 원자료는 speed.json이다. –는 재지 않은 칸(앞 크기가 5초를 넘었거나 O(n²) 방법)이다.",
        "",
        "| 마일스톤 | 검정 | n = 10 | n = 10³ | n = 10⁶ | 배치 10⁴ × 30 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ms, t in tests:
        fixed = by.get((ms, t, "fixed"))
        if fixed:
            lines.append(f"| {ms} | {t} | {fmt(fixed['ms'])} (고정 표) | | | |")
            continue
        cells = [fmt(by.get((ms, t, n), {}).get("ms")) for n in SIZES]
        bt = by.get((ms, t, "10^4 x 30"))
        lines.append(f"| {ms} | {t} | {' | '.join(cells)} | {fmt(bt['ms']) if bt else ''} |")
    (HERE / "speed.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
