"""The discrete distributions (docs/api/stats.md §2.1, T25). Parameters are keyword-only and
required, except as noted."""

from ._distribution import DiscreteDistribution


class Binomial(DiscreteDistribution):
    """The number of successes in n independent trials (n a nonnegative integer) with success
    probability 0 <= p <= 1."""

    __slots__ = ()
    _name = "binomial"
    _params = ("n", "p")

    def __init__(self, *, n, p):
        self._set(n, p)


class Poisson(DiscreteDistribution):
    """The Poisson distribution with mean mu >= 0."""

    __slots__ = ()
    _name = "poisson"
    _params = ("mu",)

    def __init__(self, *, mu):
        self._set(mu)


class Geometric(DiscreteDistribution):
    """The number of trials up to and including the first success, success probability
    0 < p <= 1 (support k >= 1, as scipy's geom)."""

    __slots__ = ()
    _name = "geometric"
    _params = ("p",)

    def __init__(self, *, p):
        self._set(p)


class NegativeBinomial(DiscreteDistribution):
    """The number of failures before the n-th success (n > 0, real), success probability
    0 < p <= 1 (support k >= 0, as scipy's nbinom)."""

    __slots__ = ()
    _name = "negativebinomial"
    _params = ("n", "p")

    def __init__(self, *, n, p):
        self._set(n, p)


class Hypergeometric(DiscreteDistribution):
    """The number of successes in `draws` draws without replacement from `total` items of
    which `successes` are successes (scipy's hypergeom(M, n, N) = (total, successes, draws))."""

    __slots__ = ()
    _name = "hypergeometric"
    _params = ("total", "successes", "draws")

    def __init__(self, *, total, successes, draws):
        self._set(total, successes, draws)


class Bernoulli(DiscreteDistribution):
    """1 with probability p, 0 with probability 1 - p (0 <= p <= 1)."""

    __slots__ = ()
    _name = "bernoulli"
    _params = ("p",)

    def __init__(self, *, p):
        self._set(p)


class DiscreteUniform(DiscreteDistribution):
    """The integers low, low + 1, ..., high - 1 with equal probability (integers low < high;
    high excluded, as scipy's randint and sd.random's integers)."""

    __slots__ = ()
    _name = "discreteuniform"
    _params = ("low", "high")

    def __init__(self, *, low, high):
        self._set(low, high)


class BetaBinomial(DiscreteDistribution):
    """The binomial distribution with n trials and a success probability drawn from
    beta(a, b) (n a nonnegative integer, a, b > 0)."""

    __slots__ = ()
    _name = "betabinomial"
    _params = ("n", "a", "b")

    def __init__(self, *, n, a, b):
        self._set(n, a, b)


class LogSeries(DiscreteDistribution):
    """The logarithmic series distribution: P(X = k) = -p^k / (k log(1 - p)), k >= 1, 0 < p < 1."""

    __slots__ = ()
    _name = "logseries"
    _params = ("p",)

    def __init__(self, *, p):
        self._set(p)


class Zipf(DiscreteDistribution):
    """The Zipf (zeta) distribution: P(X = k) = k^-a / zeta(a), k >= 1, a > 1."""

    __slots__ = ()
    _name = "zipf"
    _params = ("a",)

    def __init__(self, *, a):
        self._set(a)
