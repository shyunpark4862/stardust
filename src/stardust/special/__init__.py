"""stardust.special: special functions on stardust arrays (scipy.special's names).

Usage: ``import stardust as sd`` then ``sd.special.erf(x)``. The public API is
docs/api/special.md; the functions arrive milestone by milestone (ROADMAP.md,
special-function area, S1-S8).
"""

from .. import _core as _sd_core

# The functions live in the submodule stardust._core.special (docs/design/special.md §1.4).
_core = _sd_core.special

__version__ = "0.0.1"

__all__ = [
    # S1: logistic and information (docs/api/special.md §5)
    "expit",
    "logit",
    "log_expit",
    "xlogy",
    "xlog1py",
    "entr",
    "rel_entr",
    # S2: the error function and the normal distribution (§2)
    "erf",
    "erfc",
    "erfcx",
    "erfinv",
    "erfcinv",
    "ndtr",
    "log_ndtr",
    "ndtri",
    # S3: gamma, beta and zeta (§3, §7)
    "gamma",
    "gammaln",
    "rgamma",
    "digamma",
    "polygamma",
    "beta",
    "betaln",
    "binom",
    "zeta",
    # S4: incomplete gamma and beta, and their inverses (§3)
    "gammainc",
    "gammaincc",
    "gammaincinv",
    "gammainccinv",
    "betainc",
    "betaincc",
    "betaincinv",
    "betainccinv",
    # S5: reductions, exponential integrals, elliptic integrals (§6, §7, §10)
    "logsumexp",
    "softmax",
    "log_softmax",
    "expi",
    "expn",
    "ellipk",
    "ellipe",
    # S6: Bessel and Airy (§8, §9)
    "i0",
    "i0e",
    "i1",
    "i1e",
    "j0",
    "j1",
    "y0",
    "y1",
    "k0",
    "k0e",
    "k1",
    "k1e",
    "jv",
    "yv",
    "iv",
    "kv",
    "jve",
    "yve",
    "ive",
    "kve",
    "airy",
    # S7: hypergeometric and Lambert W (§11)
    "hyp1f1",
    "hyp2f1",
    "lambertw",
    # S8: complex (§12; erf, erfc, erfcx and lambertw also take complex)
    "wofz",
    "loggamma",
    # T1: distribution functions (§13)
    "stdtr",
    "stdtrit",
    "chdtr",
    "chdtrc",
    "chdtri",
    "fdtr",
    "fdtrc",
    "fdtri",
    "pdtr",
    "pdtrc",
    "bdtr",
    "bdtrc",
    "nbdtr",
    "nbdtrc",
    "ndtri_exp",
    # T2: noncentral distributions (§14; x last, upper tails too)
    "chndtr",
    "chndtrc",
    "chndtrix",
    "nctdtr",
    "nctdtrc",
    "nctdtrit",
    "ncfdtr",
    "ncfdtrc",
    "ncfdtri",
]

globals().update({_name: getattr(_core, _name) for _name in __all__})
