"""sd.htest's references and their contract (docs/design/htest.md §4, §7), as the tests and the
accuracy measurement (bench/accuracy/htest.py) both judge them.

tests/htest/ref/<family>.json (tools/htest_ref.py) holds, per problem of tests/htest/problems.py,
the inputs' digest and each reference field as a double-double [hi, lo] ("kappa", "kappa_stat":
plain condition numbers); its "_provenance" says what computed it. tests/htest/data holds the
other libraries' records (tools/gen_htest_scipy.py: scipy_h*.json, statsmodels.json;
tools/gen_htest_r.py: r.json), each with its own "_provenance".

The contract, per field, in units of its bound (1 or less is within it):
  - statistic: relative error / (1e-14 max(1, kappa_stat)), kappa_stat the statistic's condition
    number in the quantity it transforms (D'Agostino's moments, Anderson-Darling's cancelling
    sum; 1 elsewhere);
  - pvalue: relative error / (1e-14 max(1, kappa) max(1, kappa_stat)); the Cramér-von Mises
    p-value (1 - cdf of Csörgő and Faraway's approximation) also passes within 1e-15 absolute;
  - zstatistic (the normal approximation's z of the rank tests), df, df_num, df_denom, median
    (median_test's grand median): relative error / 1e-14;
  - conf_int (the t tests' 95% interval): each end's error / (1e-14 times the larger of the end
    and the estimate's and half width's sizes: the ends subtract them).
"""

import json
import math
import pathlib

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
BOUND = 1e-14
CVM_ABS = 1e-15

# the families by milestone
H1 = ("ttest", "f_oneway", "alexandergovern", "bartlett", "levene", "fligner")
H2 = ("power_divergence", "jarque_bera", "dagostino", "shapiro", "anderson", "ks_1samp", "cramervonmises")
H3 = (
    "mannwhitneyu",
    "ranksums",
    "wilcoxon",
    "kruskal",
    "friedmanchisquare",
    "brunnermunzel",
    "ansari",
    "mood",
    "median_test",
    "page_trend_test",
)
PUBLIC = H1 + H2 + H3  # the families of the public functions (the accuracy measurement's sweeps)


def load(fam) -> dict:
    """{problem: record} of a reference family (the file's "_provenance" left out)."""
    d = json.loads((REF / f"{fam}.json").read_text())
    d.pop("_provenance", None)
    return d


def provenance(fam) -> dict:
    return json.loads((REF / f"{fam}.json").read_text())["_provenance"]


def families() -> list:
    return sorted(p.stem for p in REF.glob("*.json"))


def data(name) -> dict:
    """{problem: record} of another library's file in tests/htest/data ("_provenance" left out)."""
    d = json.loads((DATA / f"{name}.json").read_text())
    d.pop("_provenance", None)
    return d


def _sd_args(p):
    import stardust as sd

    cdf = {"norm": sd.stats.Normal(), "uniform": sd.stats.Uniform(a=0.0, b=1.0), "expon": sd.stats.Exponential()}
    dist = {"norm": None, "expon": sd.stats.Exponential, "logistic": sd.stats.Logistic, "gumbel_r": sd.stats.Gumbel}
    args = [cdf[a] if isinstance(a, str) else a for a in p.args]
    kw = dict(p.kwargs)
    if p.func == "anderson":
        kw["dist"] = dist[kw.get("dist", "norm")]
    return args, kw


def call(p, **extra):
    """sd's call of a problem: scipy's distribution names as sd.stats objects (cdf) or classes
    (anderson's dist)."""
    import stardust as sd

    args, kw = _sd_args(p)
    return getattr(sd.htest, p.func)(*args, **{**kw, **extra})


def rel(v, ref):
    """|v - ref| / |ref| for ref a double-double [hi, lo] (NaN against NaN is 0, inf against the
    same inf 0)."""
    hi, lo = ref
    if hi != hi:
        return 0.0 if v != v else math.inf
    if math.isinf(hi):
        return 0.0 if v == hi else math.inf
    if v != v:
        return math.inf
    if hi == 0.0:
        return abs(v)
    return abs((v - hi) - lo) / abs(hi)


def factor(rec, key):
    k = rec.get(key, 1.0)
    return 1.0 if k != k else max(1.0, k)


def contract(fam, rec, field, value):
    """value's error in units of the field's bound (module docstring)."""
    ks = factor(rec, "kappa_stat")
    e = rel(value, rec[field])
    if field == "statistic":
        return e / (BOUND * ks)
    if field == "pvalue":
        c = e / (BOUND * factor(rec, "kappa") * ks)
        if fam == "cramervonmises" and rec[field][0] == rec[field][0] and abs(value - rec[field][0]) <= CVM_ABS:
            c = min(c, 1.0)
        return c
    return e / BOUND


def conf_int_contract(rec, ci, scale):
    """The interval's ends against the reference: error / (1e-14 max(|end|, scale)) each."""
    out = []
    for i in range(2):
        hi, lo = rec["conf_int"][i]
        v = float(ci[i])
        if hi != hi or math.isinf(hi):
            out.append(0.0 if v == hi or (v != v and hi != hi) else math.inf)
        else:
            out.append(abs((v - hi) - lo) / (BOUND * max(abs(hi), scale)))
    return out


def ci_scale(p):
    """The size an interval's ends are measured against: the two-sided 95% (or 90% for a
    one-sided test, the same end) interval's estimate and half width."""
    alt = p.kwargs.get("alternative", "two-sided")
    two = call(p) if alt == "two-sided" else call(p, alternative="two-sided")
    lo2, hi2 = two.conf_int(alpha=0.05 if alt == "two-sided" else 0.1).tolist()
    return abs(lo2 + hi2) / 2 + abs(hi2 - lo2) / 2


FIELDS = ("statistic", "pvalue", "zstatistic", "df", "df_num", "df_denom", "median")


def errors(fam, pid, r=None) -> dict:
    """{field: (relative error, error in bound units)} of sd's result for a reference problem
    (conf_int: its two ends)."""
    from . import problems as P

    rec = load(fam)[pid]
    p = P.PROBLEMS[pid]
    r = call(p) if r is None else r
    out = {}
    for f in FIELDS:
        if f in rec and hasattr(r, f):
            v = getattr(r, f)
            out[f] = (rel(v, rec[f]), contract(fam, rec, f, v))
    if "conf_int" in rec:
        ci = r.conf_int()
        c = conf_int_contract(rec, ci, ci_scale(p))
        out["conf_int"] = (max(rel(float(ci[i]), rec["conf_int"][i]) for i in range(2)), max(c))
    return out
