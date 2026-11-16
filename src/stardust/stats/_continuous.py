"""The continuous distributions (docs/api/stats.md §2.1). Parameters are keyword-only;
location and scale parameters default to 0 and 1, shape parameters are required (T19)."""

from ._distribution import ContinuousDistribution


class Normal(ContinuousDistribution):
    """The normal distribution with mean mu and standard deviation sigma > 0."""

    __slots__ = ()
    _name = "normal"
    _params = ("mu", "sigma")

    def __init__(self, *, mu=0.0, sigma=1.0):
        self._set(mu, sigma)


class StudentT(ContinuousDistribution):
    """Student's t distribution with df > 0 degrees of freedom."""

    __slots__ = ()
    _name = "studentt"
    _params = ("df",)

    def __init__(self, *, df):
        self._set(df)


class ChiSquared(ContinuousDistribution):
    """The chi-square distribution with df > 0 degrees of freedom."""

    __slots__ = ()
    _name = "chisquared"
    _params = ("df",)

    def __init__(self, *, df):
        self._set(df)


class F(ContinuousDistribution):
    """The F distribution with dfn > 0 and dfd > 0 degrees of freedom."""

    __slots__ = ()
    _name = "f"
    _params = ("dfn", "dfd")

    def __init__(self, *, dfn, dfd):
        self._set(dfn, dfd)


class Gamma(ContinuousDistribution):
    """The gamma distribution with shape a > 0 and scale > 0 (density
    x^(a-1) e^(-x/scale) / (Gamma(a) scale^a); the rate is 1/scale)."""

    __slots__ = ()
    _name = "gamma"
    _params = ("a", "scale")

    def __init__(self, *, a, scale=1.0):
        self._set(a, scale)


class Beta(ContinuousDistribution):
    """The beta distribution on [0, 1] with shapes a > 0 and b > 0."""

    __slots__ = ()
    _name = "beta"
    _params = ("a", "b")

    def __init__(self, *, a, b):
        self._set(a, b)


class Exponential(ContinuousDistribution):
    """The exponential distribution with scale > 0 (mean scale; the rate is 1/scale)."""

    __slots__ = ()
    _name = "exponential"
    _params = ("scale",)

    def __init__(self, *, scale=1.0):
        self._set(scale)


class Uniform(ContinuousDistribution):
    """The uniform distribution on [a, b], a < b."""

    __slots__ = ()
    _name = "uniform"
    _params = ("a", "b")

    def __init__(self, *, a, b):
        self._set(a, b)


class Laplace(ContinuousDistribution):
    """The Laplace distribution with location mu and scale sigma > 0 (density
    exp(-|x - mu| / sigma) / (2 sigma); sigma is not the standard deviation)."""

    __slots__ = ()
    _name = "laplace"
    _params = ("mu", "sigma")

    def __init__(self, *, mu=0.0, sigma=1.0):
        self._set(mu, sigma)


class Cauchy(ContinuousDistribution):
    """The Cauchy distribution with location mu and scale sigma > 0 (density
    1 / (pi sigma (1 + ((x - mu) / sigma)^2)); it has no mean or variance)."""

    __slots__ = ()
    _name = "cauchy"
    _params = ("mu", "sigma")

    def __init__(self, *, mu=0.0, sigma=1.0):
        self._set(mu, sigma)


class LogNormal(ContinuousDistribution):
    """The log-normal distribution: log X is normal with mean mu and standard deviation
    sigma > 0."""

    __slots__ = ()
    _name = "lognormal"
    _params = ("mu", "sigma")

    def __init__(self, *, mu=0.0, sigma=1.0):
        self._set(mu, sigma)


class Weibull(ContinuousDistribution):
    """The Weibull distribution with shape c > 0 and scale > 0:
    P(X > x) = exp(-(x / scale)^c) for x >= 0."""

    __slots__ = ()
    _name = "weibull"
    _params = ("c", "scale")

    def __init__(self, *, c, scale=1.0):
        self._set(c, scale)


class Logistic(ContinuousDistribution):
    """The logistic distribution with location mu and scale sigma > 0:
    P(X <= x) = 1 / (1 + exp(-(x - mu) / sigma))."""

    __slots__ = ()
    _name = "logistic"
    _params = ("mu", "sigma")

    def __init__(self, *, mu=0.0, sigma=1.0):
        self._set(mu, sigma)


class Gumbel(ContinuousDistribution):
    """The Gumbel distribution of maxima with location mu and scale sigma > 0:
    P(X <= x) = exp(-exp(-(x - mu) / sigma)) (scipy's gumbel_r; T26)."""

    __slots__ = ()
    _name = "gumbel"
    _params = ("mu", "sigma")

    def __init__(self, *, mu=0.0, sigma=1.0):
        self._set(mu, sigma)


class GEV(ContinuousDistribution):
    """The generalized extreme value distribution with shape c, location mu and scale
    sigma > 0: P(X <= x) = exp(-(1 - c z)^(1/c)), z = (x - mu) / sigma; the Gumbel at c = 0.
    c is scipy's (genextreme), the negative of the usual xi: c > 0 bounds X above at
    mu + sigma / c, c < 0 below (T26)."""

    __slots__ = ()
    _name = "gev"
    _params = ("c", "mu", "sigma")

    def __init__(self, *, c, mu=0.0, sigma=1.0):
        self._set(c, mu, sigma)


class Pareto(ContinuousDistribution):
    """The Pareto distribution with shape b > 0 and scale > 0: P(X > x) = (scale / x)^b for
    x >= scale."""

    __slots__ = ()
    _name = "pareto"
    _params = ("b", "scale")

    def __init__(self, *, b, scale=1.0):
        self._set(b, scale)


class Rayleigh(ContinuousDistribution):
    """The Rayleigh distribution with scale > 0: P(X > x) = exp(-x^2 / (2 scale^2)) for x >= 0
    (T26: scale, as for the other distributions on [0, inf))."""

    __slots__ = ()
    _name = "rayleigh"
    _params = ("scale",)

    def __init__(self, *, scale=1.0):
        self._set(scale)


class InverseGamma(ContinuousDistribution):
    """The inverse gamma distribution with shape a > 0 and scale > 0: X = scale / Y with
    Y ~ Gamma(a) (density scale^a x^(-a-1) exp(-scale / x) / Gamma(a))."""

    __slots__ = ()
    _name = "inversegamma"
    _params = ("a", "scale")

    def __init__(self, *, a, scale=1.0):
        self._set(a, scale)


class NoncentralChiSquared(ContinuousDistribution):
    """The noncentral chi-square distribution with df > 0 degrees of freedom and
    noncentrality nc >= 0 (nc = 0: the chi-square; T26)."""

    __slots__ = ()
    _name = "noncentralchisquared"
    _params = ("df", "nc")

    def __init__(self, *, df, nc):
        self._set(df, nc)


class NoncentralT(ContinuousDistribution):
    """The noncentral t distribution with df > 0 degrees of freedom and noncentrality nc:
    (Z + nc) / sqrt(V / df), Z standard normal, V chi-square(df)."""

    __slots__ = ()
    _name = "noncentralt"
    _params = ("df", "nc")

    def __init__(self, *, df, nc):
        self._set(df, nc)


class NoncentralF(ContinuousDistribution):
    """The noncentral F distribution with dfn > 0 and dfd > 0 degrees of freedom and
    noncentrality nc >= 0 (nc = 0: the F distribution)."""

    __slots__ = ()
    _name = "noncentralf"
    _params = ("dfn", "dfd", "nc")

    def __init__(self, *, dfn, dfd, nc):
        self._set(dfn, dfd, nc)
