"""Small seeded data shared by the rule tests of several fits (test_fit_rules.py and the fits'
own files): a two-column design X with a linear, a binary and a count response, the decay
model's data for nls, and a design whose column is a Unix time (a large offset)."""

import numpy as np

import stardust as sd


def a(x):
    return np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)


def gen():
    return sd.random.default_rng(3)


RNG = np.random.default_rng(20261119)
N = 60
X = RNG.normal(size=(N, 2))
Y = 1.0 + X @ [1.0, -2.0] + RNG.normal(size=N)
YB = (RNG.uniform(size=N) < 1 / (1 + np.exp(-(0.3 + X @ [1.0, -0.7])))).astype(float)
YP = RNG.poisson(np.exp(0.2 + 0.4 * X[:, 0])).astype(float)


def decay(x, a_, k, c):
    return a_ * sd.exp(-k * x) + c


def decay_jac(x, a_, k, c):
    e = sd.exp(-k * x)
    return sd.stack([e, -a_ * x * e, sd.ones(x.shape)], axis=1)


TD = np.linspace(0, 4, 40)
YD = 2.5 * np.exp(-1.3 * TD) + 0.5 + 0.01 * RNG.normal(size=40)


def unix_time_data(seed=30, n=100, scale=1.7e12, span=8.64e7, slope=1e-6, noise=1.0):
    """x a Unix time in ms over a day, y a trend of 86 units with noise: a large offset column
    and a large intercept that cancels it."""
    rg = np.random.default_rng(seed)
    x = scale + np.sort(rg.uniform(0, span, n))
    y = slope * (x - scale) + noise * rg.normal(size=n)
    return y, x
