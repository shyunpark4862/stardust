"""scipy 1.18.1's and statsmodels 0.15.0's results on the sd.htest problem collection
(tests/htest/problems.py; docs/design/htest.md §7). Writes

  tests/htest/data/scipy_h1.json ... scipy_h8.json   scipy.stats (and scipy.stats.contingency)
  tests/htest/data/statsmodels.json                  statsmodels.stats (multipletests, the
                                                     proportion functions, mcnemar, cochrans_q)

Each problem's line holds the inputs' digest (problems.digest), what the call returned -- every
field the result has (statistic, pvalue, df or dof, zstatistic, statistic_location,
statistic_sign, expected_freq, median, table, method, ...), the 95% confidence interval where
the result has confidence_interval() (binomtest: proportion_ci() by its three methods; odds_ratio:
the one-sided ones too) -- or the
exception it raised, and the warnings' categories. Floats are written with repr (NaN and inf as
JSON's NaN and Infinity, which Python reads back); arrays as nested lists.

Not run: fisher_exact on an r x c table (scipy's default is an unseeded Monte Carlo: no fixed
answer to keep). dunnett's p-values come from scipy's randomized quasi-Monte Carlo and are
recorded with rng=12345.

Run: .venv/bin/python tools/gen_htest_scipy.py [--jobs N]   (up to 10 processes)
"""

import json
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "tests" / "htest" / "data"
JOBS = 10

FIELDS = (
    "statistic",
    "pvalue",
    "df",
    "dof",
    "zstatistic",
    "statistic_location",
    "statistic_sign",
    "expected_freq",
    "median",
    "table",
    "method",
    "statistic_type",
    "k",
    "n",
    "alternative",
    "proportion_estimate",
    "relative_risk",
    # statsmodels' test_proportions_2indep
    "compare",
    "diff",
    "ratio",
    "odds_ratio",
    "variance",
    "value",
    "prop1_null",
    "prop2_null",
)


def plain(v):
    """v as JSON: numbers as Python floats (ints kept), arrays as nested lists."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, (list, tuple)):
        return [plain(e) for e in v]
    if v is None:
        return None
    return repr(v)


def scipy_call(p):
    import scipy.stats as st
    from scipy.stats import contingency

    args, kw = list(p.args), dict(p.kwargs)
    if p.func in ("ks_1samp", "kstest") and isinstance(args[1], str):
        args[1] = getattr(st, args[1]).cdf
    if p.func == "dunnett":
        kw["rng"] = 12345
    f = getattr(contingency, p.func) if p.func in ("odds_ratio", "relative_risk") else getattr(st, p.func)
    r = f(*args, **kw)
    if isinstance(r, np.ndarray):  # false_discovery_control: the corrected p-values
        return {"value": plain(r)}
    out = {}
    for k in FIELDS:
        try:
            v = getattr(r, k)
        except AttributeError, ValueError:
            continue
        if callable(v):
            continue
        out[k] = plain(v)
    if hasattr(r, "confidence_interval") and callable(r.confidence_interval):
        try:
            ci = r.confidence_interval(confidence_level=0.95)
            out["confidence_interval"] = [plain(ci.low), plain(ci.high)]
        except Exception as e:  # (scipy raises for some configurations)
            out["confidence_interval"] = f"{type(e).__name__}: {e}"
    if p.func == "odds_ratio":  # its one-sided intervals too
        for alt in ("less", "greater"):
            ci = r.confidence_interval(confidence_level=0.95, alternative=alt)
            out[f"confidence_interval_{alt}"] = [plain(ci.low), plain(ci.high)]
    if p.func == "binomtest":
        for m in ("exact", "wilson", "wilsoncc"):
            ci = r.proportion_ci(confidence_level=0.95, method=m)
            out[f"proportion_ci_{m}"] = [plain(ci.low), plain(ci.high)]
    try:
        out["unpack"] = len(r)
    except TypeError:
        out["unpack"] = None
    return out


def statsmodels_call(p):
    from statsmodels.stats import contingency_tables as ct
    from statsmodels.stats import multitest as mt
    from statsmodels.stats import proportion as pr

    args, kw = p.args, p.kwargs
    if p.func == "multipletests":
        reject, corrected, sidak, bonf = mt.multipletests(*args, **kw)
        return {"reject": plain(reject), "pvals_corrected": plain(corrected), "alphacSidak": sidak, "alphacBonf": bonf}
    if p.func == "proportions_ztest":
        z, pv = pr.proportions_ztest(*args, **kw)
        return {"statistic": plain(z), "pvalue": plain(pv)}
    if p.func == "proportions_chisquare":
        chi2, pv, (table, expected) = pr.proportions_chisquare(*args, **kw)
        return {"statistic": plain(chi2), "pvalue": plain(pv), "table": plain(table), "expected_freq": plain(expected)}
    if p.func == "proportion_confint":
        low, high = pr.proportion_confint(*args, **kw)
        return {"confidence_interval": [plain(low), plain(high)]}
    if p.func == "confint_proportions_2indep":
        low, high = pr.confint_proportions_2indep(*args, **kw)
        return {"confidence_interval": [plain(low), plain(high)]}
    if p.func == "test_proportions_2indep":
        r = pr.test_proportions_2indep(*args, **kw)
        out = {k: plain(getattr(r, k)) for k in FIELDS + ("method",) if hasattr(r, k)}
        return out
    if p.func == "mcnemar":
        r = ct.mcnemar(*args, **kw)
        return {"statistic": plain(r.statistic), "pvalue": plain(r.pvalue)}
    if p.func == "cochrans_q":
        r = ct.cochrans_q(*args, **kw)
        return {"statistic": plain(r.statistic), "pvalue": plain(r.pvalue), "df": plain(r.df)}
    raise KeyError(p.func)


def run(pid):
    from tests.htest import problems as P

    p = P.PROBLEMS[pid]
    rec = {"digest": P.digest(p)}
    if p.func == "fisher_exact" and "rxc" in p.tags:
        rec["skipped"] = "scipy's r x c default is an unseeded Monte Carlo"
        return pid, rec
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            rec["result"] = statsmodels_call(p) if p.func in P.STATSMODELS else scipy_call(p)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
    if caught:
        rec["warnings"] = sorted({w.category.__name__ for w in caught})
    return pid, rec


def write(path, meta, recs):
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in recs]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def main(argv=None):
    import argparse

    import scipy
    import statsmodels

    from bench import cli
    from tests.htest import problems as P

    ap = argparse.ArgumentParser(prog="tools/gen_htest_scipy.py", description=__doc__.split("\n\n")[0])
    cli.add_jobs(ap, default=JOBS)
    jobs = ap.parse_args(argv).jobs
    ids = list(P.PROBLEMS)
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=50) as pool:
        res = dict(pool.imap(run, ids, chunksize=4))
    meta = {
        "tool": "tools/gen_htest_scipy.py",
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "numpy": np.__version__,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    for ms in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"):
        recs = [(i, res[i]) for i in ids if P.PROBLEMS[i].milestone == ms and P.PROBLEMS[i].func not in P.STATSMODELS]
        write(DATA / f"scipy_{ms.lower()}.json", meta, recs)
    recs = [(i, res[i]) for i in ids if P.PROBLEMS[i].func in P.STATSMODELS]
    write(DATA / "statsmodels.json", meta, recs)
    errors = sum("error" in r for r in res.values())
    print(f"{len(res)} problems, {errors} raised")


if __name__ == "__main__":
    sys.exit(main())
