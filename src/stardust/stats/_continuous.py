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
