"""stardust.htest: hypothesis tests (docs/api/htest.md, docs/design/htest.md).

Usage: ``import stardust as sd`` then ``r = sd.htest.ttest_ind(a, b)`` and ``r.statistic``,
``r.pvalue``, ``r.conf_int(alpha=0.05)``. The functions and their arguments are scipy.stats'
names (and statsmodels.stats' for multiple-comparison corrections and proportions), the inputs
and options follow this library's rules (positional data, keyword-only options, axis=None as
sd.stats, nan_policy, rng=) and every test returns a read-only result object (decisions H1-H21
in docs/design/htest.md §1.1).

The module holds the result objects (_result.py) and the input layer (_common.py) the tests
share; the tests come milestone by milestone (docs/design/htest.md §8): H1's means and variances
(_parametric.py), H2's goodness of fit (_gof.py), H3's rank tests (_rank.py), H4's goodness of
fit between samples (_gof.py), H5's correlation (_correlation.py), H6's contingency tables (_contingency.py) and H7's
proportions and binomial tests (_proportion.py) so far.
"""

from ._contingency import (
    barnard_exact,
    boschloo_exact,
    chi2_contingency,
    cochrans_q,
    fisher_exact,
    mcnemar,
    odds_ratio,
    relative_risk,
)
from ._correlation import (
    chatterjeexi,
    kendalltau,
    pearsonr,
    pointbiserialr,
    somersd,
    spearmanr,
    spearmanrho,
    weightedtau,
)
from ._gof import (
    anderson,
    anderson_ksamp,
    chisquare,
    cramervonmises,
    cramervonmises_2samp,
    epps_singleton_2samp,
    jarque_bera,
    ks_1samp,
    ks_2samp,
    kurtosistest,
    normaltest,
    power_divergence,
    shapiro,
    skewtest,
)
from ._parametric import (
    alexandergovern,
    bartlett,
    f_oneway,
    fligner,
    levene,
    ttest_1samp,
    ttest_ind,
    ttest_ind_from_stats,
    ttest_rel,
)
from ._proportion import (
    binomtest,
    confint_proportions_2indep,
    poisson_means_test,
    proportion_confint,
    proportions_chisquare,
    proportions_ztest,
    quantile_test,
    test_proportions_2indep,
)
from ._rank import (
    ansari,
    brunnermunzel,
    friedmanchisquare,
    kruskal,
    mannwhitneyu,
    median_test,
    mood,
    page_trend_test,
    ranksums,
    wilcoxon,
)

__all__ = [
    "ttest_1samp",
    "ttest_rel",
    "ttest_ind",
    "ttest_ind_from_stats",
    "f_oneway",
    "alexandergovern",
    "bartlett",
    "levene",
    "fligner",
    "chisquare",
    "power_divergence",
    "skewtest",
    "kurtosistest",
    "normaltest",
    "jarque_bera",
    "shapiro",
    "anderson",
    "ks_1samp",
    "cramervonmises",
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
    "ks_2samp",
    "cramervonmises_2samp",
    "anderson_ksamp",
    "epps_singleton_2samp",
    "pearsonr",
    "spearmanrho",
    "spearmanr",
    "pointbiserialr",
    "kendalltau",
    "weightedtau",
    "somersd",
    "chatterjeexi",
    "chi2_contingency",
    "fisher_exact",
    "barnard_exact",
    "boschloo_exact",
    "mcnemar",
    "cochrans_q",
    "odds_ratio",
    "relative_risk",
    "binomtest",
    "quantile_test",
    "poisson_means_test",
    "proportions_ztest",
    "proportions_chisquare",
    "proportion_confint",
    "test_proportions_2indep",
    "confint_proportions_2indep",
]
