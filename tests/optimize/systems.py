"""The 14 square systems F(x) = 0 of MINPACK's HYBRD and HYBRJ test drivers (netlib minpack/ex
file15, file16: VECFCN, VECJAC, INITPT) in their numbering and operation order, so that doubles
agree with the Fortran to the bit (tests/optimize/data/minpack_functions.json;
docs/design/optimize.md §7).

Most are the Moré-Garbow-Hillstrom residuals with m = n (mgh.py); four differ: Rosenbrock lists
its two equations the other way round, and the Wood (4), Watson (6) and variably dimensioned (12)
systems are the gradients of 1/2 sum F_i^2 of the paper's least squares problems.
"""

from .ad import FLOAT
from .mgh import f07, f13, f26, f27, f28, f29, f30, f31, f35, j07, j13, j27, j28, j29, j30, j31, j35


def _sq(t):
    return t * t


def _zeros(n):
    return [[0.0] * n for _ in range(n)]


def f1(x, M=FLOAT):
    return [1.0 - x[0], 10.0 * (x[1] - _sq(x[0]))]


def j1(x, M=FLOAT):
    return [[-1.0, 0.0], [-20.0 * x[0], 10.0]]


def f3(x, M=FLOAT):
    return [1.0e4 * x[0] * x[1] - 1.0, M.exp(-x[0]) + M.exp(-x[1]) - 1.0001]


def j3(x, M=FLOAT):
    return [[1.0e4 * x[1], 1.0e4 * x[0]], [-M.exp(-x[0]), -M.exp(-x[1])]]


def f4(x, M=FLOAT):
    t1 = x[1] - _sq(x[0])
    t2 = x[3] - _sq(x[2])
    return [
        -200.0 * x[0] * t1 - (1.0 - x[0]),
        200.0 * t1 + 20.2 * (x[1] - 1.0) + 19.8 * (x[3] - 1.0),
        -180.0 * x[2] * t2 - (1.0 - x[2]),
        180.0 * t2 + 20.2 * (x[3] - 1.0) + 19.8 * (x[1] - 1.0),
    ]


def j4(x, M=FLOAT):
    j = _zeros(4)
    t1 = x[1] - 3.0 * _sq(x[0])
    t2 = x[3] - 3.0 * _sq(x[2])
    j[0][0] = -200.0 * t1 + 1.0
    j[0][1] = -200.0 * x[0]
    j[1][0] = -2.0 * 200.0 * x[0]
    j[1][1] = 200.0 + 20.2
    j[1][3] = 19.8
    j[2][2] = -180.0 * t2 + 1.0
    j[2][3] = -180.0 * x[2]
    j[3][1] = 19.8
    j[3][2] = -2.0 * 180.0 * x[2]
    j[3][3] = 180.0 + 20.2
    return j


def f6(x, M=FLOAT):
    n = len(x)
    out = [0.0] * n
    for i in range(1, 30):
        ti = i / 29.0
        s1, temp = 0.0, 1.0
        for j in range(2, n + 1):
            s1 = s1 + (j - 1) * temp * x[j - 1]
            temp = ti * temp
        s2, temp = 0.0, 1.0
        for j in range(1, n + 1):
            s2 = s2 + temp * x[j - 1]
            temp = ti * temp
        t1 = s1 - _sq(s2) - 1.0
        t2 = 2.0 * ti * s2
        temp = 1.0 / ti
        for k in range(1, n + 1):
            out[k - 1] = out[k - 1] + temp * ((k - 1) - t2) * t1
            temp = ti * temp
    temp = x[1] - _sq(x[0]) - 1.0
    out[0] = out[0] + x[0] * (1.0 - 2.0 * temp)
    out[1] = out[1] + temp
    return out


def j6(x, M=FLOAT):
    n = len(x)
    j = _zeros(n)
    for i in range(1, 30):
        ti = i / 29.0
        s1, temp = 0.0, 1.0
        for c in range(2, n + 1):
            s1 = s1 + (c - 1) * temp * x[c - 1]
            temp = ti * temp
        s2, temp = 0.0, 1.0
        for c in range(1, n + 1):
            s2 = s2 + temp * x[c - 1]
            temp = ti * temp
        t1 = 2.0 * (s1 - _sq(s2) - 1.0)
        t2 = 2.0 * s2
        temp = _sq(ti)
        tk = 1.0
        for k in range(1, n + 1):
            tj = tk
            for c in range(k, n + 1):
                j[k - 1][c - 1] = j[k - 1][c - 1] + tj * (((k - 1) / ti - t2) * ((c - 1) / ti - t2) - t1)
                tj = ti * tj
            tk = temp * tk
    j[0][0] = j[0][0] + 6.0 * _sq(x[0]) - 2.0 * x[1] + 3.0
    j[0][1] = j[0][1] - 2.0 * x[0]
    j[1][1] = j[1][1] + 1.0
    for k in range(n):
        for c in range(k, n):
            j[c][k] = j[k][c]
    return j


def f11(x, M=FLOAT):
    # VECFCN's order: n + k - sin x_k - sum cos x_j - k cos x_k
    n = len(x)
    c = [M.cos(v) for v in x]
    s = 0.0
    for v in c:
        s = s + v
    return [float(n + k + 1) - M.sin(x[k]) - s - (k + 1) * c[k] for k in range(n)]


def j11(x, M=FLOAT):
    n = len(x)
    j = [[M.sin(x[c]) for c in range(n)] for _ in range(n)]
    for c in range(n):
        j[c][c] = (c + 2) * M.sin(x[c]) - M.cos(x[c])
    return j


def f12(x, M=FLOAT):
    s = 0.0
    for j, v in enumerate(x, 1):
        s = s + j * (v - 1.0)
    temp = s * (1.0 + 2.0 * _sq(s))
    return [v - 1.0 + k * temp for k, v in enumerate(x, 1)]


def j12(x, M=FLOAT):
    n = len(x)
    s = 0.0
    for j, v in enumerate(x, 1):
        s = s + j * (v - 1.0)
    temp = 1.0 + 6.0 * _sq(s)
    j = _zeros(n)
    for k in range(1, n + 1):
        for c in range(k, n + 1):
            j[k - 1][c - 1] = float(k * c) * temp
            j[c - 1][k - 1] = j[k - 1][c - 1]
        j[k - 1][k - 1] = j[k - 1][k - 1] + 1.0
    return j


def _square(f, jf):
    return (lambda x, M=FLOAT: f(x, len(x), M)), (lambda x, M=FLOAT: jf(x, len(x), M))


# number -> (name, F(x, M), J(x, M))
SYSTEMS = {
    1: ("Rosenbrock", f1, j1),
    2: ("Powell singular", *_square(f13, j13)),
    3: ("Powell badly scaled", f3, j3),
    4: ("Wood", f4, j4),
    5: ("Helical valley", *_square(f07, j07)),
    6: ("Watson", f6, j6),
    7: ("Chebyquad", *_square(f35, j35)),
    8: ("Brown almost-linear", *_square(f27, j27)),
    9: ("Discrete boundary value", *_square(f28, j28)),
    10: ("Discrete integral equation", *_square(f29, j29)),
    11: ("Trigonometric", f11, j11),
    12: ("Variably dimensioned", f12, j12),
    13: ("Broyden tridiagonal", *_square(f30, j30)),
    14: ("Broyden banded", *_square(f31, j31)),
}
# f26 (the paper's trigonometric residuals) is the same function written in the paper's order
PAPER_TRIGONOMETRIC = f26


def x0(number, n, factor=1.0):
    """INITPT of the HYBRD/HYBRJ drivers: the standard point times factor (Watson's zero
    point becomes the vector of the factor)."""
    if number == 1:
        x = [-1.2, 1.0]
    elif number == 2:
        x = [3.0, -1.0, 0.0, 1.0]
    elif number == 3:
        x = [0.0, 1.0]
    elif number == 4:
        x = [-3.0, -1.0, -3.0, -1.0]
    elif number == 5:
        x = [-1.0, 0.0, 0.0]
    elif number == 6:
        x = [0.0] * n
    elif number == 7:
        h = 1.0 / (n + 1)
        x = [j * h for j in range(1, n + 1)]
    elif number == 8:
        x = [0.5] * n
    elif number in (9, 10):
        h = 1.0 / (n + 1)
        x = [(j * h) * (j * h - 1.0) for j in range(1, n + 1)]
    elif number == 11:
        x = [1.0 / n] * n
    elif number == 12:
        h = 1.0 / n
        x = [1.0 - j * h for j in range(1, n + 1)]
    elif number in (13, 14):
        x = [-1.0] * n
    else:
        raise ValueError(number)
    if factor == 1.0:
        return x
    if number == 6:
        return [float(factor)] * n
    return [factor * v for v in x]


class SystemCase:
    """One system at one size and start: number, n, factor."""

    __slots__ = ("number", "n", "factor")

    def __init__(self, number, n, factor):
        self.number, self.n, self.factor = number, n, float(factor)

    @property
    def name(self):
        return SYSTEMS[self.number][0]

    @property
    def id(self):
        return f"hybrd{self.number:02d}-n{self.n}-x{int(self.factor)}"

    def x0(self):
        return x0(self.number, self.n, self.factor)

    def fun(self, x, M=FLOAT):
        return SYSTEMS[self.number][1](list(x), M)

    def jac(self, x, M=FLOAT):
        return SYSTEMS[self.number][2](list(x), M)

    def __repr__(self):
        return self.id


def _cases():
    # file21, the data of the HYBRD/HYBRJ drivers: (problem, n, number of starts)
    rows = [
        (1, 2, 3),
        (2, 4, 3),
        (3, 2, 2),
        (4, 4, 3),
        (5, 3, 3),
        (6, 6, 2),
        (6, 9, 2),
        (7, 5, 3),
        (7, 6, 3),
        (7, 7, 3),
        (7, 8, 1),
        (7, 9, 1),
        (8, 10, 3),
        (8, 30, 1),
        (8, 40, 1),
        (9, 10, 3),
        (10, 1, 3),
        (10, 10, 3),
        (11, 10, 3),
        (12, 10, 3),
        (13, 10, 3),
        (14, 10, 3),
    ]
    return [SystemCase(p, n, 10.0**k) for p, n, tries in rows for k in range(tries)]


# The HYBRD/HYBRJ driver runs, in their order (52 runs)
CASES = _cases()
BY_ID = {c.id: c for c in CASES}


def as_numpy(case):
    import numpy as np

    def F(x):
        return np.array(case.fun([np.float64(v) for v in x]))

    def J(x):
        return np.array(case.jac([np.float64(v) for v in x]), dtype=float).reshape(case.n, case.n)

    return F, J
