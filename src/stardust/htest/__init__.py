"""stardust.htest: hypothesis tests (docs/api/htest.md, docs/design/htest.md).

Usage: ``import stardust as sd`` then ``r = sd.htest.ttest_ind(a, b)`` and ``r.statistic``,
``r.pvalue``, ``r.conf_int(alpha=0.05)``. The functions and their arguments are scipy.stats'
names (and statsmodels.stats' for multiple-comparison corrections and proportions), the inputs
and options follow this library's rules (positional data, keyword-only options, axis=None as
sd.stats, nan_policy, rng=) and every test returns a read-only result object (decisions H1-H21
in docs/design/htest.md §1.1).

The module holds the result objects (_result.py) and the input layer (_common.py) the tests
share; the tests come milestone by milestone (docs/design/htest.md §8): H1's means and variances
(_parametric.py), H2's goodness of fit (_gof.py) and H3's rank tests (_rank.py) so far.
"""

from ._gof import (
    anderson,
    chisquare,
    cramervonmises,
    jarque_bera,
    ks_1samp,
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
]
