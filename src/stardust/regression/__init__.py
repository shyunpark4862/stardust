"""stardust.regression: regression models (docs/api/regression.md).

Usage: ``import stardust as sd`` then ``r = sd.regression.ols(y, X)`` and ``r.params``,
``r.bse``, ``r.pvalues``, ``r.predict(X_new)``. The intercept is put in by default and is
``params[0]``; the result's names are statsmodels'. The decisions are
docs/design/regression.md §1.1.
"""

from .._results import (BreuschPaganResult, JarqueBeraResult, LinregressResult, SiegelslopesResult,
                        TheilslopesResult)
from ._glm import GLMResult, glm
from ._linear import GLSResult, OLSResult, WLSResult, gls, linregress, ols, wls
from ._nls import NLSResult, nls
from ._penalized import (ElasticNetCVResult, ElasticNetResult, LassoCVResult, LassoResult, PenalizedGLMResult,
                         RidgeCVResult, RidgeResult, elastic_net, elastic_net_cv, lasso, lasso_cv, penalized_glm, ridge,
                         ridge_cv)
from ._quantile import QuantRegResult, quantreg
from ._robust import RLMResult, rlm, siegelslopes, theilslopes

__all__ = ["ols", "wls", "gls", "linregress", "glm", "ridge", "lasso", "elastic_net", "ridge_cv", "lasso_cv",
           "elastic_net_cv", "penalized_glm", "rlm", "theilslopes", "siegelslopes", "quantreg", "nls", "OLSResult", "WLSResult",
           "GLSResult", "GLMResult", "LinregressResult", "JarqueBeraResult", "BreuschPaganResult", "RidgeResult",
           "LassoResult", "ElasticNetResult", "RidgeCVResult", "LassoCVResult", "ElasticNetCVResult",
           "PenalizedGLMResult", "RLMResult", "TheilslopesResult", "SiegelslopesResult", "QuantRegResult", "NLSResult"]
