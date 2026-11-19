"""The 35 test problems of Moré, Garbow and Hillstrom, "Testing unconstrained optimization
software", ACM TOMS 7 (1981) 17-41 (docs/design/optimize.md §7), in the paper's numbering.

Each problem is a residual vector F: R^n -> R^m with its analytic Jacobian, so that it serves

  - least squares: minimize 1/2 sum F_i^2 (scipy's cost) or sum F_i^2 (the paper's f),
  - unconstrained minimization: f(x) = sum F_i^2, gradient 2 J^T F, Hessian by Dual2 (ad.py),
  - square systems F(x) = 0 where m = n (the systems of MINPACK's HYBRD/HYBRJ drivers, some of
    which differ from these residuals, are in systems.py).

The 18 problems of MINPACK's least squares drivers (netlib minpack/ex file17, file19) are
written in the operation order of their SSQFCN and SSQJAC, so that doubles agree with the
Fortran to the bit (tests/optimize/data/minpack_functions.json); the rest follow the paper.
Two choices the paper leaves open follow MINPACK: the helical valley's theta at x1 = 0 is
sign(x2)/4, and the multiples of a zero starting point (Watson) set every component to the
factor. The Gulf function's m, free in the paper (3 <= m <= 100), is 99 here.

Problems take x as a list of numbers and M, a namespace of elementary functions (ad.FLOAT,
ad.ARB, ad.AD). The functions here return lists; as_numpy() wraps them for scipy.
"""

import math

from .ad import FLOAT


def _sq(t):
    return t * t


def _cube(t):
    return t * t * t


# ---- the data of the paper's tables (MINPACK's DATA statements for the shared ones)

BARD_Y = (1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1, 3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1,
          1.34, 2.1, 4.39)
GAUSSIAN_Y = (0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521, 0.3989, 0.3521, 0.2420, 0.1295, 0.0540,
              0.0175, 0.0044, 0.0009)
MEYER_Y = (3.478e4, 2.861e4, 2.365e4, 1.963e4, 1.637e4, 1.372e4, 1.154e4, 9.744e3, 8.261e3, 7.03e3, 6.005e3,
           5.147e3, 4.427e3, 3.82e3, 3.307e3, 2.872e3)
KOWALIK_V = (4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2)
KOWALIK_Y = (1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2)
OSBORNE1_Y = (8.44e-1, 9.08e-1, 9.32e-1, 9.36e-1, 9.25e-1, 9.08e-1, 8.81e-1, 8.5e-1, 8.18e-1, 7.84e-1, 7.51e-1,
              7.18e-1, 6.85e-1, 6.58e-1, 6.28e-1, 6.03e-1, 5.8e-1, 5.58e-1, 5.38e-1, 5.22e-1, 5.06e-1, 4.9e-1,
              4.78e-1, 4.67e-1, 4.57e-1, 4.48e-1, 4.38e-1, 4.31e-1, 4.24e-1, 4.2e-1, 4.14e-1, 4.11e-1, 4.06e-1)
OSBORNE2_Y = (1.366, 1.191, 1.112, 1.013, 9.91e-1, 8.85e-1, 8.31e-1, 8.47e-1, 7.86e-1, 7.25e-1, 7.46e-1, 6.79e-1,
              6.08e-1, 6.55e-1, 6.16e-1, 6.06e-1, 6.02e-1, 6.26e-1, 6.51e-1, 7.24e-1, 6.49e-1, 6.49e-1, 6.94e-1,
              6.44e-1, 6.24e-1, 6.61e-1, 6.12e-1, 5.58e-1, 5.33e-1, 4.95e-1, 5.0e-1, 4.23e-1, 3.95e-1, 3.75e-1,
              3.72e-1, 3.91e-1, 3.96e-1, 4.05e-1, 4.28e-1, 4.29e-1, 5.23e-1, 5.62e-1, 6.07e-1, 6.53e-1, 6.72e-1,
              7.08e-1, 6.33e-1, 6.68e-1, 6.45e-1, 6.32e-1, 5.91e-1, 5.59e-1, 5.97e-1, 6.25e-1, 7.39e-1, 7.1e-1,
              7.29e-1, 7.2e-1, 6.36e-1, 5.81e-1, 4.28e-1, 2.92e-1, 1.62e-1, 9.8e-2, 5.4e-2)


def _zeros(m, n):
    return [[0.0] * n for _ in range(m)]


# ---- 1. Rosenbrock (MINPACK lsq 4)
def f01(x, m, M=FLOAT):
    return [10.0 * (x[1] - _sq(x[0])), 1.0 - x[0]]


def j01(x, m, M=FLOAT):
    return [[-20.0 * x[0], 10.0], [-1.0, 0.0]]


# ---- 2. Freudenstein and Roth (MINPACK lsq 7)
def f02(x, m, M=FLOAT):
    return [-13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1],
            -29.0 + x[0] + ((1.0 + x[1]) * x[1] - 14.0) * x[1]]


def j02(x, m, M=FLOAT):
    return [[1.0, x[1] * (10.0 - 3.0 * x[1]) - 2.0], [1.0, x[1] * (2.0 + 3.0 * x[1]) - 14.0]]


# ---- 3. Powell badly scaled
def f03(x, m, M=FLOAT):
    return [1.0e4 * x[0] * x[1] - 1.0, M.exp(-x[0]) + M.exp(-x[1]) - 1.0001]


def j03(x, m, M=FLOAT):
    return [[1.0e4 * x[1], 1.0e4 * x[0]], [-M.exp(-x[0]), -M.exp(-x[1])]]


# ---- 4. Brown badly scaled
def f04(x, m, M=FLOAT):
    return [x[0] - 1.0e6, x[1] - 2.0e-6, x[0] * x[1] - 2.0]


def j04(x, m, M=FLOAT):
    return [[1.0, 0.0], [0.0, 1.0], [x[1], x[0]]]


# ---- 5. Beale
def _beale_pow(t, i):
    return t if i == 1 else (t * t if i == 2 else t * t * t)


def f05(x, m, M=FLOAT):
    y = (1.5, 2.25, 2.625)
    return [y[i - 1] - x[0] * (1.0 - _beale_pow(x[1], i)) for i in (1, 2, 3)]


def j05(x, m, M=FLOAT):
    rows = []
    for i in (1, 2, 3):
        rows.append([-(1.0 - _beale_pow(x[1], i)), x[0] * i * (1.0 if i == 1 else _beale_pow(x[1], i - 1))])
    return rows


# ---- 6. Jennrich and Sampson (MINPACK lsq 13)
def f06(x, m, M=FLOAT):
    return [2.0 + 2.0 * i - M.exp(i * x[0]) - M.exp(i * x[1]) for i in range(1, m + 1)]


def j06(x, m, M=FLOAT):
    return [[-i * M.exp(i * x[0]), -i * M.exp(i * x[1])] for i in range(1, m + 1)]


# ---- 7. Helical valley (MINPACK lsq 5)
def _theta(x, M):
    tpi = 8.0 * M.atan(1.0)
    if x[0] > 0:
        return M.atan(x[1] / x[0]) / tpi
    if x[0] < 0:
        return M.atan(x[1] / x[0]) / tpi + 0.5
    return M.copysign(0.25, x[1])


def f07(x, m, M=FLOAT):
    t1 = _theta(x, M)
    t2 = M.sqrt(_sq(x[0]) + _sq(x[1]))
    return [10.0 * (x[2] - 10.0 * t1), 10.0 * (t2 - 1.0), x[2]]


def j07(x, m, M=FLOAT):
    tpi = 8.0 * M.atan(1.0)
    temp = _sq(x[0]) + _sq(x[1])
    t1 = tpi * temp
    t2 = M.sqrt(temp)
    return [[100.0 * x[1] / t1, -100.0 * x[0] / t1, 10.0], [10.0 * x[0] / t2, 10.0 * x[1] / t2, 0.0],
            [0.0, 0.0, 1.0]]


# ---- 8. Bard (MINPACK lsq 8)
def _bard_uvw(i):
    u, v = float(i), float(16 - i)
    return u, v, (u if i <= 8 else v)


def f08(x, m, M=FLOAT):
    out = []
    for i in range(1, 16):
        u, v, w = _bard_uvw(i)
        out.append(BARD_Y[i - 1] - (x[0] + u / (x[1] * v + x[2] * w)))
    return out


def j08(x, m, M=FLOAT):
    rows = []
    for i in range(1, 16):
        u, v, w = _bard_uvw(i)
        d = _sq(x[1] * v + x[2] * w)
        rows.append([-1.0, u * v / d, u * w / d])
    return rows


# ---- 9. Gaussian
def f09(x, m, M=FLOAT):
    out = []
    for i in range(1, 16):
        d = (8 - i) / 2.0 - x[2]
        out.append(x[0] * M.exp(-x[1] * _sq(d) / 2.0) - GAUSSIAN_Y[i - 1])
    return out


def j09(x, m, M=FLOAT):
    rows = []
    for i in range(1, 16):
        d = (8 - i) / 2.0 - x[2]
        e = M.exp(-x[1] * _sq(d) / 2.0)
        rows.append([e, -x[0] * e * _sq(d) / 2.0, x[0] * e * x[1] * d])
    return rows


# ---- 10. Meyer (MINPACK lsq 10)
def f10(x, m, M=FLOAT):
    out = []
    for i in range(1, 17):
        temp = 5.0 * i + 45.0 + x[2]
        out.append(x[0] * M.exp(x[1] / temp) - MEYER_Y[i - 1])
    return out


def j10(x, m, M=FLOAT):
    rows = []
    for i in range(1, 17):
        temp = 5.0 * i + 45.0 + x[2]
        t1 = x[1] / temp
        t2 = M.exp(t1)
        d2 = x[0] * t2 / temp
        rows.append([t2, d2, -t1 * d2])
    return rows


# ---- 11. Gulf research and development
def _gulf_ty(i):
    t = i / 100.0
    return t, 25.0 + (-50.0 * math.log(t)) ** (2.0 / 3.0)


def f11(x, m, M=FLOAT):
    out = []
    for i in range(1, m + 1):
        t, y = _gulf_ty(i)
        out.append(M.exp(-(abs(y - x[1]) ** x[2]) / x[0]) - t)
    return out


def j11(x, m, M=FLOAT):
    rows = []
    for i in range(1, m + 1):
        t, y = _gulf_ty(i)
        d = y - x[1]
        a = abs(d)
        p = a ** x[2]
        e = M.exp(-p / x[0])
        s = 1.0 if d > 0 else -1.0
        rows.append([e * p / _sq(x[0]), e * x[2] * a ** (x[2] - 1.0) * s / x[0], -e * p * M.log(a) / x[0]])
    return rows


# ---- 12. Box three-dimensional (MINPACK lsq 12)
def f12(x, m, M=FLOAT):
    out = []
    for i in range(1, m + 1):
        temp = float(i)
        t1 = temp / 10.0
        out.append(M.exp(-t1 * x[0]) - M.exp(-t1 * x[1]) + (M.exp(-temp) - M.exp(-t1)) * x[2])
    return out


def j12(x, m, M=FLOAT):
    rows = []
    for i in range(1, m + 1):
        temp = float(i)
        t1 = temp / 10.0
        rows.append([-t1 * M.exp(-t1 * x[0]), t1 * M.exp(-t1 * x[1]), M.exp(-temp) - M.exp(-t1)])
    return rows


# ---- 13. Powell singular (MINPACK lsq 6)
def f13(x, m, M=FLOAT):
    return [x[0] + 10.0 * x[1], M.sqrt(5.0) * (x[2] - x[3]), _sq(x[1] - 2.0 * x[2]),
            M.sqrt(10.0) * _sq(x[0] - x[3])]


def j13(x, m, M=FLOAT):
    j = _zeros(4, 4)
    j[0][0] = 1.0
    j[0][1] = 10.0
    j[1][2] = M.sqrt(5.0)
    j[1][3] = -j[1][2]
    j[2][1] = 2.0 * (x[1] - 2.0 * x[2])
    j[2][2] = -2.0 * j[2][1]
    j[3][0] = 2.0 * M.sqrt(10.0) * (x[0] - x[3])
    j[3][3] = -j[3][0]
    return j


# ---- 14. Wood
def f14(x, m, M=FLOAT):
    s90, s10 = M.sqrt(90.0), M.sqrt(10.0)
    return [10.0 * (x[1] - _sq(x[0])), 1.0 - x[0], s90 * (x[3] - _sq(x[2])), 1.0 - x[2],
            s10 * (x[1] + x[3] - 2.0), (x[1] - x[3]) / s10]


def j14(x, m, M=FLOAT):
    s90, s10 = M.sqrt(90.0), M.sqrt(10.0)
    return [[-20.0 * x[0], 10.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -2.0 * s90 * x[2], s90],
            [0.0, 0.0, -1.0, 0.0], [0.0, s10, 0.0, s10], [0.0, 1.0 / s10, 0.0, -1.0 / s10]]


# ---- 15. Kowalik and Osborne (MINPACK lsq 9)
def f15(x, m, M=FLOAT):
    out = []
    for i in range(11):
        v = KOWALIK_V[i]
        t1 = v * (v + x[1])
        t2 = v * (v + x[2]) + x[3]
        out.append(KOWALIK_Y[i] - x[0] * t1 / t2)
    return out


def j15(x, m, M=FLOAT):
    rows = []
    for i in range(11):
        v = KOWALIK_V[i]
        t1 = v * (v + x[1])
        t2 = v * (v + x[2]) + x[3]
        a = -t1 / t2
        b = -v * x[0] / t2
        c = a * b
        rows.append([a, b, c, c / v])
    return rows


# ---- 16. Brown and Dennis (MINPACK lsq 14)
def f16(x, m, M=FLOAT):
    out = []
    for i in range(1, m + 1):
        temp = i / 5.0
        t1 = x[0] + temp * x[1] - M.exp(temp)
        t2 = x[2] + M.sin(temp) * x[3] - M.cos(temp)
        out.append(_sq(t1) + _sq(t2))
    return out


def j16(x, m, M=FLOAT):
    rows = []
    for i in range(1, m + 1):
        temp = i / 5.0
        ti = M.sin(temp)
        t1 = x[0] + temp * x[1] - M.exp(temp)
        t2 = x[2] + ti * x[3] - M.cos(temp)
        a, c = 2.0 * t1, 2.0 * t2
        rows.append([a, temp * a, c, ti * c])
    return rows


# ---- 17. Osborne 1 (MINPACK lsq 17)
def f17(x, m, M=FLOAT):
    out = []
    for i in range(1, 34):
        temp = 10.0 * (i - 1)
        out.append(OSBORNE1_Y[i - 1] - (x[0] + x[1] * M.exp(-x[3] * temp) + x[2] * M.exp(-x[4] * temp)))
    return out


def j17(x, m, M=FLOAT):
    rows = []
    for i in range(1, 34):
        temp = 10.0 * (i - 1)
        t1 = M.exp(-x[3] * temp)
        t2 = M.exp(-x[4] * temp)
        rows.append([-1.0, -t1, -t2, temp * x[1] * t1, temp * x[2] * t2])
    return rows


# ---- 18. Biggs EXP6
def _biggs_ty(i):
    t = 0.1 * i
    return t, math.exp(-t) - 5.0 * math.exp(-10.0 * t) + 3.0 * math.exp(-4.0 * t)


def f18(x, m, M=FLOAT):
    out = []
    for i in range(1, m + 1):
        t, y = _biggs_ty(i)
        out.append(x[2] * M.exp(-t * x[0]) - x[3] * M.exp(-t * x[1]) + x[5] * M.exp(-t * x[4]) - y)
    return out


def j18(x, m, M=FLOAT):
    rows = []
    for i in range(1, m + 1):
        t, _ = _biggs_ty(i)
        e1, e2, e5 = M.exp(-t * x[0]), M.exp(-t * x[1]), M.exp(-t * x[4])
        rows.append([-t * x[2] * e1, t * x[3] * e2, e1, -e2, -t * x[5] * e5, e5])
    return rows


# ---- 19. Osborne 2 (MINPACK lsq 18)
def f19(x, m, M=FLOAT):
    out = []
    for i in range(1, 66):
        temp = (i - 1) / 10.0
        t1 = M.exp(-x[4] * temp)
        t2 = M.exp(-x[5] * _sq(temp - x[8]))
        t3 = M.exp(-x[6] * _sq(temp - x[9]))
        t4 = M.exp(-x[7] * _sq(temp - x[10]))
        out.append(OSBORNE2_Y[i - 1] - (x[0] * t1 + x[1] * t2 + x[2] * t3 + x[3] * t4))
    return out


def j19(x, m, M=FLOAT):
    rows = []
    for i in range(1, 66):
        temp = (i - 1) / 10.0
        t1 = M.exp(-x[4] * temp)
        t2 = M.exp(-x[5] * _sq(temp - x[8]))
        t3 = M.exp(-x[6] * _sq(temp - x[9]))
        t4 = M.exp(-x[7] * _sq(temp - x[10]))
        rows.append([-t1, -t2, -t3, -t4, temp * x[0] * t1, x[1] * _sq(temp - x[8]) * t2,
                     x[2] * _sq(temp - x[9]) * t3, x[3] * _sq(temp - x[10]) * t4,
                     -2.0 * x[1] * x[5] * (temp - x[8]) * t2, -2.0 * x[2] * x[6] * (temp - x[9]) * t3,
                     -2.0 * x[3] * x[7] * (temp - x[10]) * t4])
    return rows


# ---- 20. Watson (MINPACK lsq 11)
def f20(x, m, M=FLOAT):
    n = len(x)
    out = []
    for i in range(1, 30):
        div = i / 29.0
        s1, dx = 0.0, 1.0
        for j in range(2, n + 1):
            s1 = s1 + (j - 1) * dx * x[j - 1]
            dx = div * dx
        s2, dx = 0.0, 1.0
        for j in range(1, n + 1):
            s2 = s2 + dx * x[j - 1]
            dx = div * dx
        out.append(s1 - _sq(s2) - 1.0)
    out.append(x[0])
    out.append(x[1] - _sq(x[0]) - 1.0)
    return out


def j20(x, m, M=FLOAT):
    n = len(x)
    rows = []
    for i in range(1, 30):
        div = i / 29.0
        s2, dx = 0.0, 1.0
        for j in range(1, n + 1):
            s2 = s2 + dx * x[j - 1]
            dx = div * dx
        temp = 2.0 * div * s2
        dx = 1.0 / div
        row = []
        for j in range(1, n + 1):
            row.append(dx * ((j - 1) - temp))
            dx = div * dx
        rows.append(row)
    r30 = [0.0] * n
    r30[0] = 1.0
    r31 = [0.0] * n
    r31[0] = -2.0 * x[0]
    r31[1] = 1.0
    return rows + [r30, r31]


# ---- 21. Extended Rosenbrock
def f21(x, m, M=FLOAT):
    out = []
    for i in range(0, len(x), 2):
        out += [10.0 * (x[i + 1] - _sq(x[i])), 1.0 - x[i]]
    return out


def j21(x, m, M=FLOAT):
    n = len(x)
    j = _zeros(n, n)
    for i in range(0, n, 2):
        j[i][i], j[i][i + 1] = -20.0 * x[i], 10.0
        j[i + 1][i] = -1.0
    return j


# ---- 22. Extended Powell singular
def f22(x, m, M=FLOAT):
    out = []
    for i in range(0, len(x), 4):
        out += f13(x[i:i + 4], 4, M)
    return out


def j22(x, m, M=FLOAT):
    n = len(x)
    j = _zeros(n, n)
    for i in range(0, n, 4):
        b = j13(x[i:i + 4], 4, M)
        for r in range(4):
            for c in range(4):
                j[i + r][i + c] = b[r][c]
    return j


# ---- 23. Penalty I
def f23(x, m, M=FLOAT):
    a = M.sqrt(1.0e-5)
    s = 0.0
    for v in x:
        s = s + _sq(v)
    return [a * (v - 1.0) for v in x] + [s - 0.25]


def j23(x, m, M=FLOAT):
    n = len(x)
    a = M.sqrt(1.0e-5)
    j = _zeros(n + 1, n)
    for i in range(n):
        j[i][i] = a
        j[n][i] = 2.0 * x[i]
    return j


# ---- 24. Penalty II
def f24(x, m, M=FLOAT):
    n = len(x)
    a = M.sqrt(1.0e-5)
    e1 = M.exp(-0.1)
    out = [x[0] - 0.2]
    for i in range(2, n + 1):
        y = M.exp(i / 10.0) + M.exp((i - 1) / 10.0)
        out.append(a * (M.exp(x[i - 1] / 10.0) + M.exp(x[i - 2] / 10.0) - y))
    for i in range(n + 1, 2 * n):
        out.append(a * (M.exp(x[i - n] / 10.0) - e1))
    s = 0.0
    for j in range(1, n + 1):
        s = s + (n - j + 1) * _sq(x[j - 1])
    out.append(s - 1.0)
    return out


def j24(x, m, M=FLOAT):
    n = len(x)
    a = M.sqrt(1.0e-5)
    j = _zeros(2 * n, n)
    j[0][0] = 1.0
    for i in range(2, n + 1):
        j[i - 1][i - 1] = a * M.exp(x[i - 1] / 10.0) / 10.0
        j[i - 1][i - 2] = a * M.exp(x[i - 2] / 10.0) / 10.0
    for i in range(n + 1, 2 * n):
        j[i - 1][i - n] = a * M.exp(x[i - n] / 10.0) / 10.0
    for k in range(1, n + 1):
        j[2 * n - 1][k - 1] = 2.0 * (n - k + 1) * x[k - 1]
    return j


# ---- 25. Variably dimensioned
def f25(x, m, M=FLOAT):
    s = 0.0
    for j, v in enumerate(x, 1):
        s = s + j * (v - 1.0)
    return [v - 1.0 for v in x] + [s, _sq(s)]


def j25(x, m, M=FLOAT):
    n = len(x)
    s = 0.0
    for j, v in enumerate(x, 1):
        s = s + j * (v - 1.0)
    j = _zeros(n + 2, n)
    for i in range(n):
        j[i][i] = 1.0
        j[n][i] = float(i + 1)
        j[n + 1][i] = 2.0 * s * (i + 1)
    return j


# ---- 26. Trigonometric
def f26(x, m, M=FLOAT):
    n = len(x)
    c = [M.cos(v) for v in x]
    s = 0.0
    for v in c:
        s = s + v
    return [n - s + (i + 1) * (1.0 - c[i]) - M.sin(x[i]) for i in range(n)]


def j26(x, m, M=FLOAT):
    n = len(x)
    j = [[M.sin(x[k]) for k in range(n)] for _ in range(n)]
    for i in range(n):
        j[i][i] = (i + 2) * M.sin(x[i]) - M.cos(x[i])
    return j


# ---- 27. Brown almost-linear (MINPACK lsq 16)
def f27(x, m, M=FLOAT):
    n = len(x)
    s = -float(n + 1)
    prod = 1.0
    for v in x:
        s = s + v
        prod = v * prod
    out = [v + s for v in x]
    out[n - 1] = prod - 1.0
    return out


def j27(x, m, M=FLOAT):
    n = len(x)
    j = [[1.0] * n for _ in range(n)]
    prod = 1.0
    for c in range(n):
        prod = x[c] * prod
        j[c][c] = 2.0
    for c in range(n):
        temp = x[c]
        p = prod
        if temp == 0:
            temp, p = 1.0, 1.0
            for k in range(n):
                if k != c:
                    p = x[k] * p
        j[n - 1][c] = p / temp
    return j


# ---- 28. Discrete boundary value
def f28(x, m, M=FLOAT):
    n = len(x)
    h = 1.0 / (n + 1)
    out = []
    for k in range(n):
        t = (k + 1) * h
        lo = x[k - 1] if k > 0 else 0.0
        hi = x[k + 1] if k < n - 1 else 0.0
        out.append(2.0 * x[k] - lo - hi + _cube(x[k] + t + 1.0) * _sq(h) / 2.0)
    return out


def j28(x, m, M=FLOAT):
    n = len(x)
    h = 1.0 / (n + 1)
    j = _zeros(n, n)
    for k in range(n):
        t = (k + 1) * h
        j[k][k] = 2.0 + 3.0 * _sq(x[k] + t + 1.0) * _sq(h) / 2.0
        if k > 0:
            j[k][k - 1] = -1.0
        if k < n - 1:
            j[k][k + 1] = -1.0
    return j


# ---- 29. Discrete integral equation
def f29(x, m, M=FLOAT):
    n = len(x)
    h = 1.0 / (n + 1)
    out = []
    for k in range(1, n + 1):
        tk = k * h
        s1 = 0.0
        for j in range(1, k + 1):
            tj = j * h
            s1 = s1 + tj * _cube(x[j - 1] + tj + 1.0)
        s2 = 0.0
        for j in range(k + 1, n + 1):
            tj = j * h
            s2 = s2 + (1.0 - tj) * _cube(x[j - 1] + tj + 1.0)
        out.append(x[k - 1] + h * ((1.0 - tk) * s1 + tk * s2) / 2.0)
    return out


def j29(x, m, M=FLOAT):
    n = len(x)
    h = 1.0 / (n + 1)
    j = _zeros(n, n)
    for k in range(1, n + 1):
        tk = k * h
        for c in range(1, n + 1):
            tj = c * h
            temp = 3.0 * _sq(x[c - 1] + tj + 1.0)
            j[k - 1][c - 1] = h * min(tj * (1.0 - tk), tk * (1.0 - tj)) * temp / 2.0
        j[k - 1][k - 1] = j[k - 1][k - 1] + 1.0
    return j


# ---- 30. Broyden tridiagonal
def f30(x, m, M=FLOAT):
    n = len(x)
    out = []
    for k in range(n):
        lo = x[k - 1] if k > 0 else 0.0
        hi = x[k + 1] if k < n - 1 else 0.0
        out.append((3.0 - 2.0 * x[k]) * x[k] - lo - 2.0 * hi + 1.0)
    return out


def j30(x, m, M=FLOAT):
    n = len(x)
    j = _zeros(n, n)
    for k in range(n):
        j[k][k] = 3.0 - 4.0 * x[k]
        if k > 0:
            j[k][k - 1] = -1.0
        if k < n - 1:
            j[k][k + 1] = -2.0
    return j


# ---- 31. Broyden banded (ml = 5, mu = 1)
def f31(x, m, M=FLOAT):
    n = len(x)
    out = []
    for k in range(n):
        temp = 0.0
        for c in range(max(0, k - 5), min(k + 1, n - 1) + 1):
            if c != k:
                temp = temp + x[c] * (1.0 + x[c])
        out.append(x[k] * (2.0 + 5.0 * _sq(x[k])) + 1.0 - temp)
    return out


def j31(x, m, M=FLOAT):
    n = len(x)
    j = _zeros(n, n)
    for k in range(n):
        for c in range(max(0, k - 5), min(k + 1, n - 1) + 1):
            if c != k:
                j[k][c] = -(1.0 + 2.0 * x[c])
        j[k][k] = 2.0 + 15.0 * _sq(x[k])
    return j


# ---- 32. Linear function, full rank (MINPACK lsq 1)
def f32(x, m, M=FLOAT):
    n = len(x)
    s = 0.0
    for v in x:
        s = s + v
    temp = 2.0 * s / m + 1.0
    return [(-temp + x[i]) if i < n else -temp for i in range(m)]


def j32(x, m, M=FLOAT):
    n = len(x)
    temp = 2.0 / m
    j = [[-temp] * n for _ in range(m)]
    for c in range(n):
        j[c][c] = j[c][c] + 1.0
    return j


# ---- 33. Linear function, rank 1 (MINPACK lsq 2)
def f33(x, m, M=FLOAT):
    s = 0.0
    for c, v in enumerate(x, 1):
        s = s + c * v
    return [i * s - 1.0 for i in range(1, m + 1)]


def j33(x, m, M=FLOAT):
    return [[float(i) * float(c) for c in range(1, len(x) + 1)] for i in range(1, m + 1)]


# ---- 34. Linear function, rank 1 with zero columns and rows (MINPACK lsq 3)
def f34(x, m, M=FLOAT):
    n = len(x)
    s = 0.0
    for c in range(2, n):
        s = s + c * x[c - 1]
    out = [(i - 1) * s - 1.0 for i in range(1, m + 1)]
    out[m - 1] = -1.0
    return out


def j34(x, m, M=FLOAT):
    n = len(x)
    j = _zeros(m, n)
    for c in range(2, n):
        for i in range(2, m):
            j[i - 1][c - 1] = float(i - 1) * float(c)
    return j


# ---- 35. Chebyquad (MINPACK lsq 15)
def f35(x, m, M=FLOAT):
    n = len(x)
    out = [0.0] * m
    for c in range(n):
        t1 = 1.0
        t2 = 2.0 * x[c] - 1.0
        temp = 2.0 * t2
        for i in range(m):
            out[i] = out[i] + t2
            ti = temp * t2 - t1
            t1 = t2
            t2 = ti
    dx = 1.0 / n
    iev = -1
    for i in range(m):
        out[i] = dx * out[i]
        if iev > 0:
            out[i] = out[i] + 1.0 / (_sq(float(i + 1)) - 1.0)
        iev = -iev
    return out


def j35(x, m, M=FLOAT):
    n = len(x)
    dx = 1.0 / n
    j = _zeros(m, n)
    for c in range(n):
        t1 = 1.0
        t2 = 2.0 * x[c] - 1.0
        temp = 2.0 * t2
        t3 = 0.0
        t4 = 2.0
        for i in range(m):
            j[i][c] = dx * t4
            ti = 4.0 * t2 + temp * t4 - t3
            t3 = t4
            t4 = ti
            ti = temp * t2 - t1
            t1 = t2
            t2 = ti
    return j


# ---- starting points (x0 of the paper; MINPACK's INITPT for the shared ones)

def _x0(number, n):
    if number == 1:
        return [-1.2, 1.0]
    if number == 2:
        return [0.5, -2.0]
    if number == 3:
        return [0.0, 1.0]
    if number in (4, 5):
        return [1.0, 1.0]
    if number == 6:
        return [0.3, 0.4]
    if number == 7:
        return [-1.0, 0.0, 0.0]
    if number == 8:
        return [1.0, 1.0, 1.0]
    if number == 9:
        return [0.4, 1.0, 0.0]
    if number == 10:
        return [0.02, 4000.0, 250.0]
    if number == 11:
        return [5.0, 2.5, 0.15]
    if number == 12:
        return [0.0, 10.0, 20.0]
    if number == 13:
        return [3.0, -1.0, 0.0, 1.0]
    if number == 14:
        return [-3.0, -1.0, -3.0, -1.0]
    if number == 15:
        return [0.25, 0.39, 0.415, 0.39]
    if number == 16:
        return [25.0, 5.0, -5.0, -1.0]
    if number == 17:
        return [0.5, 1.5, -1.0, 0.01, 0.02]
    if number == 18:
        return [1.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    if number == 19:
        return [1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5]
    if number == 20:
        return [0.0] * n
    if number == 21:
        return [-1.2, 1.0] * (n // 2)
    if number == 22:
        return [3.0, -1.0, 0.0, 1.0] * (n // 4)
    if number == 23:
        return [float(j) for j in range(1, n + 1)]
    if number == 24:
        return [0.5] * n
    if number == 25:
        h = 1.0 / n
        return [1.0 - j * h for j in range(1, n + 1)]
    if number == 26:
        return [1.0 / n] * n
    if number == 27:
        return [0.5] * n
    if number in (28, 29):
        h = 1.0 / (n + 1)
        return [(j * h) * (j * h - 1.0) for j in range(1, n + 1)]
    if number in (30, 31):
        return [-1.0] * n
    if number in (32, 33, 34):
        return [1.0] * n
    if number == 35:
        h = 1.0 / (n + 1)
        return [j * h for j in range(1, n + 1)]
    raise ValueError(number)


def x0(number, n, factor=1.0):
    """The standard starting point times factor (MINPACK's INITPT rule: a zero starting
    point becomes the vector of the factor)."""
    x = _x0(number, n)
    if factor == 1.0:
        return x
    if number == 20:
        return [float(factor)] * n
    return [factor * v for v in x]


class Problem:
    """number, name, fun(x, m, M), jac(x, m, M), the allowed (n, m), and the paper's optima."""

    def __init__(self, number, name, fun, jac, n, m, minpack=None):
        self.number, self.name, self.fun, self.jac = number, name, fun, jac
        self._n, self._m = n, m          # an int, or None when the dimension is a parameter
        self.minpack = minpack           # MINPACK's least squares problem number, or None

    def m_for(self, n):
        m = self._m
        return m(n) if callable(m) else m

    def __repr__(self):
        return f"Problem({self.number}, {self.name!r})"


def _same(n):
    return n


PROBLEMS = {p.number: p for p in [
    Problem(1, "Rosenbrock", f01, j01, 2, 2, minpack=4),
    Problem(2, "Freudenstein and Roth", f02, j02, 2, 2, minpack=7),
    Problem(3, "Powell badly scaled", f03, j03, 2, 2),
    Problem(4, "Brown badly scaled", f04, j04, 2, 3),
    Problem(5, "Beale", f05, j05, 2, 3),
    Problem(6, "Jennrich and Sampson", f06, j06, 2, 10, minpack=13),
    Problem(7, "Helical valley", f07, j07, 3, 3, minpack=5),
    Problem(8, "Bard", f08, j08, 3, 15, minpack=8),
    Problem(9, "Gaussian", f09, j09, 3, 15),
    Problem(10, "Meyer", f10, j10, 3, 16, minpack=10),
    Problem(11, "Gulf research and development", f11, j11, 3, 99),
    Problem(12, "Box three-dimensional", f12, j12, 3, 10, minpack=12),
    Problem(13, "Powell singular", f13, j13, 4, 4, minpack=6),
    Problem(14, "Wood", f14, j14, 4, 6),
    Problem(15, "Kowalik and Osborne", f15, j15, 4, 11, minpack=9),
    Problem(16, "Brown and Dennis", f16, j16, 4, 20, minpack=14),
    Problem(17, "Osborne 1", f17, j17, 5, 33, minpack=17),
    Problem(18, "Biggs EXP6", f18, j18, 6, 13),
    Problem(19, "Osborne 2", f19, j19, 11, 65, minpack=18),
    Problem(20, "Watson", f20, j20, None, 31, minpack=11),
    Problem(21, "Extended Rosenbrock", f21, j21, None, _same),
    Problem(22, "Extended Powell singular", f22, j22, None, _same),
    Problem(23, "Penalty I", f23, j23, None, lambda n: n + 1),
    Problem(24, "Penalty II", f24, j24, None, lambda n: 2 * n),
    Problem(25, "Variably dimensioned", f25, j25, None, lambda n: n + 2),
    Problem(26, "Trigonometric", f26, j26, None, _same),
    Problem(27, "Brown almost-linear", f27, j27, None, _same, minpack=16),
    Problem(28, "Discrete boundary value", f28, j28, None, _same),
    Problem(29, "Discrete integral equation", f29, j29, None, _same),
    Problem(30, "Broyden tridiagonal", f30, j30, None, _same),
    Problem(31, "Broyden banded", f31, j31, None, _same),
    Problem(32, "Linear function - full rank", f32, j32, None, None, minpack=1),
    Problem(33, "Linear function - rank 1", f33, j33, None, None, minpack=2),
    Problem(34, "Linear function - rank 1 with zero columns and rows", f34, j34, None, None, minpack=3),
    Problem(35, "Chebyquad", f35, j35, None, None, minpack=15),
]}

# MINPACK's least squares numbering (file17, file19) -> the paper's
MINPACK_LSQ = {p.minpack: p.number for p in PROBLEMS.values() if p.minpack}


class Case:
    """One problem at one size and starting point: number, n, m, factor."""

    __slots__ = ("number", "n", "m", "factor")

    def __init__(self, number, n, m, factor):
        self.number, self.n, self.m, self.factor = number, n, m, float(factor)

    @property
    def problem(self):
        return PROBLEMS[self.number]

    @property
    def id(self):
        return f"mgh{self.number:02d}-n{self.n}-m{self.m}-x{int(self.factor)}"

    def x0(self):
        return x0(self.number, self.n, self.factor)

    def fun(self, x, M=FLOAT):
        return self.problem.fun(list(x), self.m, M)

    def jac(self, x, M=FLOAT):
        return self.problem.jac(list(x), self.m, M)

    def f(self, x, M=FLOAT):
        """The paper's objective sum F_i^2 (in order)."""
        s = 0.0
        for r in self.fun(x, M):
            s = s + r * r
        return s

    def __repr__(self):
        return self.id


def _minpack_lsq_cases():
    # file22, the data of the LMDER/LMDIF drivers: (MINPACK problem, n, m, number of starts)
    rows = [(1, 5, 10, 1), (1, 5, 50, 1), (2, 5, 10, 1), (2, 5, 50, 1), (3, 5, 10, 1), (3, 5, 50, 1),
            (4, 2, 2, 3), (5, 3, 3, 3), (6, 4, 4, 3), (7, 2, 2, 3), (8, 3, 15, 3), (9, 4, 11, 3), (10, 3, 16, 2),
            (11, 6, 31, 3), (11, 9, 31, 3), (11, 12, 31, 3), (12, 3, 10, 1), (13, 2, 10, 1), (14, 4, 20, 3),
            (15, 1, 8, 3), (15, 8, 8, 1), (15, 9, 9, 1), (15, 10, 10, 1), (16, 10, 10, 3), (16, 30, 30, 1),
            (16, 40, 40, 1), (17, 5, 33, 1), (18, 11, 65, 1)]
    out = []
    for mp, n, m, tries in rows:
        for k in range(tries):
            out.append(Case(MINPACK_LSQ[mp], n, m, 10.0 ** k))
    return out


# MINPACK's least squares driver runs, in its order (53 runs)
MINPACK_LSQ_CASES = _minpack_lsq_cases()

# The problems MINPACK's drivers leave out, at the paper's sizes, from x0, 10 x0 and 100 x0
_OTHERS = [(3, 2), (4, 2), (5, 2), (9, 3), (11, 3), (14, 4), (18, 6), (21, 10), (22, 12), (23, 4), (23, 10),
           (24, 4), (24, 10), (25, 10), (26, 10), (28, 10), (29, 10), (30, 10), (31, 10)]
OTHER_CASES = [Case(num, n, PROBLEMS[num].m_for(n), f) for num, n in _OTHERS for f in (1.0, 10.0, 100.0)]

# The whole collection: MINPACK's runs and the rest of the 35 problems (110 cases)
CASES = MINPACK_LSQ_CASES + OTHER_CASES
BY_ID = {c.id: c for c in CASES}

# The paper's optimal values of f = sum F_i^2 (6 significant digits), by (number, n, m); a
# problem with several local minima lists them all
KNOWN = {
    (1, 2, 2): (0.0,), (2, 2, 2): (0.0, 48.9842), (3, 2, 2): (0.0,), (4, 2, 3): (0.0,), (5, 2, 3): (0.0,),
    (6, 2, 10): (124.362,), (7, 3, 3): (0.0,), (8, 3, 15): (8.21487e-3, 17.4286), (9, 3, 15): (1.12793e-8,),
    (10, 3, 16): (87.9458,), (11, 3, 99): (0.0,), (12, 3, 10): (0.0,), (13, 4, 4): (0.0,), (14, 4, 6): (0.0,),
    (15, 4, 11): (3.07505e-4, 1.02734e-3), (16, 4, 20): (85822.2,), (17, 5, 33): (5.46489e-5,),
    (18, 6, 13): (0.0, 5.65565e-3), (19, 11, 65): (4.01377e-2,),
    (20, 6, 31): (2.28767e-3,), (20, 9, 31): (1.39976e-6,), (20, 12, 31): (4.72238e-10,),
    (21, 10, 10): (0.0,), (22, 12, 12): (0.0,), (23, 4, 5): (2.24997e-5,), (23, 10, 11): (7.08765e-5,),
    (24, 4, 8): (9.37629e-6,), (24, 10, 20): (2.93660e-4,), (25, 10, 12): (0.0,), (26, 10, 10): (0.0,),
    (27, 10, 10): (0.0, 1.0), (27, 30, 30): (0.0, 1.0), (27, 40, 40): (0.0, 1.0), (28, 10, 10): (0.0,),
    (29, 10, 10): (0.0,), (30, 10, 10): (0.0,), (31, 10, 10): (0.0,),
    (35, 8, 8): (3.51687e-3,), (35, 9, 9): (0.0,), (35, 10, 10): (6.50395e-3,),
}
for _m in (10, 50):
    KNOWN[(32, 5, _m)] = (float(_m - 5),)
    KNOWN[(33, 5, _m)] = (_m * (_m - 1) / (2.0 * (2 * _m + 1)),)
    KNOWN[(34, 5, _m)] = ((_m * _m + 3 * _m - 6) / (2.0 * (2 * _m - 3)),)

# Minimizers the paper gives exactly, where F is 0 (or the known value) to rounding
MINIMIZERS = {
    (1, 2, 2): [1.0, 1.0], (2, 2, 2): [5.0, 4.0], (4, 2, 3): [1.0e6, 2.0e-6], (5, 2, 3): [3.0, 0.5],
    (7, 3, 3): [1.0, 0.0, 0.0], (11, 3, 99): [50.0, 25.0, 1.5], (12, 3, 10): [1.0, 10.0, 1.0],
    (13, 4, 4): [0.0, 0.0, 0.0, 0.0], (14, 4, 6): [1.0, 1.0, 1.0, 1.0],
    (18, 6, 13): [1.0, 10.0, 1.0, 5.0, 4.0, 3.0], (21, 10, 10): [1.0] * 10, (22, 12, 12): [0.0] * 12,
    (25, 10, 12): [1.0] * 10, (27, 10, 10): [1.0] * 10, (32, 5, 10): [-1.0] * 5, (32, 5, 50): [-1.0] * 5,
}


def as_numpy(case):
    """(F, J, f, g) of a case for numpy: residuals (m,), Jacobian (m, n), f = sum F^2 and its
    gradient 2 J^T F."""
    import numpy as np

    def F(x):
        return np.array(case.fun([np.float64(v) for v in x]))

    def J(x):
        return np.array(case.jac([np.float64(v) for v in x]), dtype=float).reshape(case.m, case.n)

    def f(x):
        return case.f([np.float64(v) for v in x])

    def g(x):
        xl = [np.float64(v) for v in x]
        return 2.0 * (np.array(case.jac(xl), dtype=float).reshape(case.m, case.n).T @ np.array(case.fun(xl)))

    return F, J, f, g


def hessian(case, x):
    """The Hessian of f = sum F_i^2 at x, exact to rounding (Dual2)."""
    from .ad import value_grad_hess
    return value_grad_hess(lambda v, M: _sumsq(case.problem.fun(v, case.m, M)), x)[2]


def _sumsq(rs):
    s = 0.0
    for r in rs:
        s = s + r * r
    return s
