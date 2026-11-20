"""stardust.regression: regression models (docs/api/regression.md).

Usage: ``import stardust as sd`` then ``r = sd.regression.ols(y, X)`` and ``r.params``,
``r.bse``, ``r.pvalues``, ``r.predict(X_new)``. The intercept is put in by default and is
``params[0]``; the result's names are statsmodels'. The decisions are
docs/design/regression.md §1.1.
"""

from .. import _calls
from .._results import BreuschPaganResult, JarqueBeraResult, LinregressResult, SiegelslopesResult, TheilslopesResult
from ._glm import GLMResult, glm
from ._linear import GLSResult, OLSResult, WLSResult, gls, linregress, ols, wls
from ._nls import NLSResult, nls
from ._penalized import (
    ElasticNetCVResult,
    ElasticNetResult,
    LassoCVResult,
    LassoResult,
    PenalizedGLMResult,
    RidgeCVResult,
    RidgeResult,
    elastic_net,
    elastic_net_cv,
    lasso,
    lasso_cv,
    penalized_glm,
    ridge,
    ridge_cv,
)
from ._quantile import QuantRegResult, quantreg
from ._robust import RLMResult, rlm, siegelslopes, theilslopes
from ._tests import (
    AnovaResult,
    CompareFTestResult,
    CompareLRTestResult,
    GoldfeldQuandtResult,
    ResetTestResult,
    TTestResult,
    WaldTestResult,
    WhiteTestResult,
)
from ._tests import anova as _anova

__all__ = [
    "ols",
    "wls",
    "gls",
    "linregress",
    "glm",
    "ridge",
    "lasso",
    "elastic_net",
    "ridge_cv",
    "lasso_cv",
    "elastic_net_cv",
    "penalized_glm",
    "rlm",
    "theilslopes",
    "siegelslopes",
    "quantreg",
    "nls",
    "anova",
    "OLSResult",
    "WLSResult",
    "GLSResult",
    "GLMResult",
    "LinregressResult",
    "JarqueBeraResult",
    "BreuschPaganResult",
    "RidgeResult",
    "LassoResult",
    "ElasticNetResult",
    "RidgeCVResult",
    "LassoCVResult",
    "ElasticNetCVResult",
    "PenalizedGLMResult",
    "RLMResult",
    "TheilslopesResult",
    "SiegelslopesResult",
    "QuantRegResult",
    "NLSResult",
    "TTestResult",
    "WaldTestResult",
    "CompareFTestResult",
    "CompareLRTestResult",
    "AnovaResult",
    "WhiteTestResult",
    "GoldfeldQuandtResult",
    "ResetTestResult",
]


@_calls.function
def anova(*results, typ=None, terms=None, test=None):
    """The analysis of variance (docs/api/regression.md §9.1): with several fitted models of the
    same data, nested in order, the table of each against the one before (R's anova(m1, m2, ...));
    with one, its terms' table, typ=1 (sequential, R's anova) or typ=3 (each term last, its Wald
    test; R's drop1). terms: the columns (positions or names) of each term, every column once;
    None makes each column a term. test: None (R's default), 'F' or 'Chisq'."""
    return _anova(*results, typ=typ, terms=terms, test=test)
