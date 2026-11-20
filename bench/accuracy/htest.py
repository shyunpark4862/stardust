"""sd.htest's accuracy sweeps: one per reference family of the public tests
(tests/htest/refdata.PUBLIC: tests/htest/ref/<family>.json, tools/htest_ref.py), one region per
field (statistic, pvalue, df, df_num, df_denom, conf_int), each problem of the family a point.

The contract is tests/htest/refdata.py's, as the tests judge it (docs/design/htest.md §4): the
statistic to 1e-14 times its condition number in what it transforms (kappa_stat), the p-value
to 1e-14 max(1, kappa_p) (times kappa_stat), degrees of freedom to 1e-14, the t tests' interval
ends to 1e-14 of the estimate's and half width's sizes. ULP is ulp_distance from the
reference's double (common.ulps).

The comparators are the recorded results of scipy 1.18.1 and statsmodels 0.15.0
(tests/htest/data/scipy_h*.json, statsmodels.json; tools/gen_htest_scipy.py) and of R 4.6.2
(r.json, and nortest's and goftest's r_gof.json; tools/gen_htest_r.py), measured on the statistic and the p-value the same way: the
recorded files, so a table does not change with the installed libraries. A library without the
problem, or a field it does not give (R's statistic where its definition differs, docs/design/
htest.md §7), has no point there.
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region, ulps

# the families whose R statistic is scipy's (R's W, V, S, T and z differ by definition elsewhere)
R_SAME_STATISTIC = ("ttest", "f_oneway", "bartlett", "fligner", "levene", "power_divergence", "ks_1samp", "shapiro")
R_FIELD = {"statistic": "statistic", "pvalue": "p.value"}
# R's goodness-of-fit packages (tests/htest/data/r_gof.json, H44) in the families they share with
# sd: goftest's cvm.test is the same test (Csörgő and Faraway's p-value); nortest's ad.test has
# Anderson-Darling's statistic for the normal family, its p-value from Stephens' formula, not the
# table interpolation (another definition: the statistic only)
R_GOF = {"cramervonmises": "gof_cvm", "anderson": "ad"}
R_GOF_FIELDS = {"cramervonmises": ("statistic", "pvalue"), "anderson": ("statistic",)}


def _regions(rows) -> dict:
    """rows: {field: [(problem, value got, relative error, contract units)]} -> {field: entry}."""
    from tests.htest import refdata as R

    out = {}
    for field, pts in rows.items():
        if not pts:
            continue
        where = np.array([p for p, *_ in pts])
        got = np.array([g for _, g, _, _, _ in pts], dtype=np.float64)
        hi = np.array([h for _, _, h, _, _ in pts], dtype=np.float64)
        rel = np.array([e for *_, e, _ in pts], dtype=np.float64)
        units = np.array([c for *_, c in pts], dtype=np.float64)
        out[field] = region(where, ulp=ulps(got, hi), rel=rel, contract=units * R.BOUND, bound=R.BOUND)
    return out


def _sd_regions(fam) -> dict:
    from tests.htest import problems as P
    from tests.htest import refdata as R

    rows = {}
    for pid, rec in R.load(fam).items():
        r = R.call(P.PROBLEMS[pid])
        for field, (e, c) in R.errors(fam, pid, r).items():
            got = float(r.conf_int()[0]) if field == "conf_int" else float(getattr(r, field))
            hi = rec[field][0][0] if field == "conf_int" else rec[field][0]
            rows.setdefault(field, []).append((pid, got, hi, e, c))
    return _regions(rows)


def _recorded(lib):
    from tests.htest import refdata as R

    if lib == "R":
        return R.data("r")
    if lib == "statsmodels":
        return R.data("statsmodels")
    out = {}
    for path in sorted(R.DATA.glob("scipy_h*.json")):
        out.update(R.data(path.stem))
    return out


def _lib_regions(fam, lib):
    """A comparator's regions (statistic, pvalue) from its recorded results; None without any."""
    from tests.htest import refdata as R

    data = _recorded(lib)
    gof = R.data("r_gof") if lib == "R" else {}
    rows = {}
    for pid, rec in R.load(fam).items():
        res = data.get(pid, {}).get("result") or {}
        if lib == "R" and fam in R_GOF and pid in gof:  # nortest's and goftest's tests (H44)
            g = gof[pid].get("result") or {}
            res = {R_FIELD[f]: g.get(f"{R_GOF[fam]}.{R_FIELD[f]}") for f in R_GOF_FIELDS[fam]}
        for field in ("statistic", "pvalue"):
            if lib == "R":
                if field == "statistic" and fam not in R_SAME_STATISTIC and fam not in R_GOF:
                    continue
                v = res.get(R_FIELD[field])
                v = v[0] if isinstance(v, list) and len(v) == 1 else None
            else:
                v = res.get(field)
                v = v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
            if v is None:
                continue
            v = float(v)
            rows.setdefault(field, []).append(
                (pid, v, rec[field][0], R.rel(v, rec[field]), R.contract(fam, rec, field, v))
            )
    return _regions(rows) or None


def sweeps():
    from tests.htest import refdata as R

    out = []
    for fam in R.PUBLIC:

        def run(fam=fam):
            return _sd_regions(fam)

        def measure(lib, fam=fam):
            return _lib_regions(fam, lib)

        out.append(Custom("htest", fam, run, compare_libs(measure, ("scipy", "statsmodels", "R"))))
    return out
