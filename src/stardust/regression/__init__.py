"""stardust.regression: regression models (docs/api/regression.md).

Usage: ``import stardust as sd`` then ``r = sd.regression.ols(y, X)`` and ``r.params``,
``r.bse``, ``r.pvalues``, ``r.predict(X_new)``. The intercept is put in by default and is
``params[0]``; the result's names are statsmodels'. The decisions are
docs/design/regression.md §1.1.
"""

from .._results import BreuschPaganResult, JarqueBeraResult, LinregressResult
from ._linear import GLSResult, OLSResult, WLSResult, gls, linregress, ols, wls

__all__ = ["ols", "wls", "gls", "linregress", "OLSResult", "WLSResult", "GLSResult", "LinregressResult",
           "JarqueBeraResult", "BreuschPaganResult"]
