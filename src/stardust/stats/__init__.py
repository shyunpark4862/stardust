"""stardust.stats: probability distributions (scipy.stats' new-style API, a subset).

Usage: ``import stardust as sd`` then ``X = sd.stats.Normal(mu=0.0, sigma=2.0)`` and
``X.cdf(x)``; descriptive statistics as ``sd.stats.skew(x)``. The public API is docs/api/stats.md; the decisions are
docs/design/stats.md §1.1. Every method is elementwise and accurate to 1e-14 relative
in each distribution's guaranteed region, like sd.special.
"""

from ._continuous import (GEV, Beta, Cauchy, ChiSquared, Exponential, F, Gamma, Gumbel, InverseGamma, Laplace,
                          Logistic, LogNormal, NoncentralChiSquared, NoncentralF, NoncentralT, Normal, Pareto, Rayleigh,
                          StudentT, Uniform, Weibull)
from ._discrete import (Bernoulli, BetaBinomial, Binomial, DiscreteUniform, Geometric, Hypergeometric, LogSeries,
                        NegativeBinomial, Poisson, Zipf)
from ._descriptive import (DescribeResult, ModeResult, describe, differential_entropy, entropy, iqr, kurtosis, mode,
                           moment, rankdata, sem, skew, zscore)
from ._distribution import ContinuousDistribution, DiscreteDistribution
from ._multivariate import Dirichlet, Multinomial, MultivariateNormal, MultivariateT

__all__ = [
    "ContinuousDistribution", "DiscreteDistribution",
    # T4: continuous (docs/api/stats.md §2.1)
    "Normal", "StudentT", "ChiSquared", "F", "Gamma", "Beta", "Exponential", "LogNormal", "Uniform", "Cauchy",
    "Laplace", "Weibull",
    # T6: extreme value and lifetime
    "Logistic", "Gumbel", "GEV", "Pareto", "Rayleigh", "InverseGamma",
    "NoncentralChiSquared", "NoncentralT", "NoncentralF",
    # T5: discrete
    "Binomial", "Poisson", "Geometric", "NegativeBinomial", "Hypergeometric", "Bernoulli", "DiscreteUniform",
    "BetaBinomial", "LogSeries", "Zipf",
    # T9: multivariate (docs/api/stats.md §3)
    "MultivariateNormal", "MultivariateT", "Multinomial", "Dirichlet",
    # T8: descriptive statistics (docs/api/stats.md §4)
    "describe", "skew", "kurtosis", "moment", "zscore", "rankdata", "sem", "iqr", "mode", "entropy",
    "differential_entropy", "DescribeResult", "ModeResult",
]
