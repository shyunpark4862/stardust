"""stardust.stats: probability distributions (scipy.stats' new-style API, a subset).

Usage: ``import stardust as sd`` then ``X = sd.stats.Normal(mu=0.0, sigma=2.0)`` and
``X.cdf(x)``. The public API is docs/api/stats.md; the decisions are
docs/design/stats.md §1.1. Every method is elementwise and accurate to 1e-14 relative
in each distribution's guaranteed region, like sd.special.
"""

from ._continuous import (Beta, Cauchy, ChiSquared, Exponential, F, Gamma, Laplace, LogNormal, Normal, StudentT,
                          Uniform, Weibull)
from ._discrete import (Bernoulli, BetaBinomial, Binomial, DiscreteUniform, Geometric, Hypergeometric, LogSeries,
                        NegativeBinomial, Poisson, Zipf)
from ._distribution import ContinuousDistribution, DiscreteDistribution

__all__ = [
    "ContinuousDistribution", "DiscreteDistribution",
    # T4: continuous (docs/api/stats.md §2.1)
    "Normal", "StudentT", "ChiSquared", "F", "Gamma", "Beta", "Exponential", "LogNormal", "Uniform", "Cauchy",
    "Laplace", "Weibull",
    # T5: discrete
    "Binomial", "Poisson", "Geometric", "NegativeBinomial", "Hypergeometric", "Bernoulli", "DiscreteUniform",
    "BetaBinomial", "LogSeries", "Zipf",
]
