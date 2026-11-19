"""The NIST StRD nonlinear least squares problems (tests/regression/data/nls/*.dat, from
https://www.itl.nist.gov/div898/strd/nls/): data, the two starting points and the certified
values (docs/design/regression.md §3.7), with each model and its Jacobian written for any
array namespace with exp, log, cos, sin, atan (sd and numpy)."""

import math
import os
import re

DATA = os.path.join(os.path.dirname(__file__), "data", "nls")

# name -> (difficulty, number of parameters); the order of NIST's table
PROBLEMS = {
    "Misra1a": ("lower", 2),
    "Chwirut2": ("lower", 3),
    "Chwirut1": ("lower", 3),
    "Lanczos3": ("lower", 6),
    "Gauss1": ("lower", 8),
    "Gauss2": ("lower", 8),
    "DanWood": ("lower", 2),
    "Misra1b": ("lower", 2),
    "Kirby2": ("average", 5),
    "Hahn1": ("average", 7),
    "Nelson": ("average", 3),
    "MGH17": ("average", 5),
    "Lanczos1": ("average", 6),
    "Lanczos2": ("average", 6),
    "Gauss3": ("average", 8),
    "Misra1c": ("average", 2),
    "Misra1d": ("average", 2),
    "Roszman1": ("average", 4),
    "ENSO": ("average", 9),
    "MGH09": ("higher", 4),
    "Thurber": ("higher", 7),
    "BoxBOD": ("higher", 2),
    "Rat42": ("higher", 3),
    "MGH10": ("higher", 3),
    "Eckerle4": ("higher", 3),
    "Rat43": ("higher", 4),
    "Bennett5": ("higher", 3),
}
NAMES = list(PROBLEMS)


def load(name):
    """dict: y and x as lists of decimal strings (x: rows; Nelson has two columns and its
    response is log y), start1, start2, the certified params, bse, rsd and ssr (strings)."""
    with open(os.path.join(DATA, name + ".dat"), encoding="ascii") as fh:
        lines = fh.read().replace("\r", "").splitlines()
    head = "\n".join(lines[:12])
    a, b = map(int, re.search(r"Data\s+\(lines (\d+) to (\d+)\)", head).groups())
    rows = [ln.split() for ln in lines[a - 1 : b] if ln.strip()]
    start1, start2, params, bse = [], [], [], []
    ssr = rsd = None
    for ln in lines[: a - 1]:
        m = re.match(r"\s*b(\d+)\s*=\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", ln)
        if m:
            start1.append(m.group(2))
            start2.append(m.group(3))
            params.append(m.group(4))
            bse.append(m.group(5))
        m = re.match(r"\s*Residual Sum of Squares:\s+(\S+)", ln)
        if m:
            ssr = m.group(1)
        m = re.match(r"\s*Residual Standard Deviation:\s+(\S+)", ln)
        if m:
            rsd = m.group(1)
    assert len(params) == PROBLEMS[name][1], name
    return {
        "name": name,
        "y": [r[0] for r in rows],
        "x": [r[1:] for r in rows],
        "start1": start1,
        "start2": start2,
        "params": params,
        "bse": bse,
        "ssr": ssr,
        "rsd": rsd,
    }


def data(d, xp):
    """(x, y) as float64 arrays of namespace xp: x (n,) or (n, 2) for Nelson, y (log y for Nelson)."""
    if d["name"] == "Nelson":
        x = xp.asarray([[float(v) for v in r] for r in d["x"]])
        y = xp.asarray([math.log(float(v)) for v in d["y"]])
    else:
        x = xp.asarray([float(r[0]) for r in d["x"]])
        y = xp.asarray([float(v) for v in d["y"]])
    return x, y


def _rational(x, b, k, xp):
    """(N, D, powers): N = b1 + ... + b_{k} x^(k-1), D = 1 + b_{k+1} x + ..."""
    pw = [x**0 * 1.0]
    for _ in range(max(k, len(b) - k + 1)):
        pw.append(pw[-1] * x)
    N = sum(b[i] * pw[i] for i in range(k))
    D = 1.0 + sum(b[k + i] * pw[i + 1] for i in range(len(b) - k))
    return N, D, pw


def model(name, xp):
    """(f, jac): f(x, *b) and jac(x, *b) -> (n, p), written with xp's functions."""
    exp, log, cos, sin, atan, stack = xp.exp, xp.log, xp.cos, xp.sin, xp.atan, xp.stack
    J = lambda *cols: stack(list(cols), axis=1)
    one = lambda x: x * 0.0 + 1.0
    pi = math.pi

    if name in ("Misra1a", "BoxBOD"):
        f = lambda x, b1, b2: b1 * (1 - exp(-b2 * x))
        jac = lambda x, b1, b2: J(1 - exp(-b2 * x), b1 * x * exp(-b2 * x))
    elif name in ("Chwirut1", "Chwirut2"):
        f = lambda x, b1, b2, b3: exp(-b1 * x) / (b2 + b3 * x)

        def jac(x, b1, b2, b3):
            v = exp(-b1 * x) / (b2 + b3 * x)
            return J(-x * v, -v / (b2 + b3 * x), -x * v / (b2 + b3 * x))
    elif name.startswith("Lanczos"):
        f = lambda x, b1, b2, b3, b4, b5, b6: b1 * exp(-b2 * x) + b3 * exp(-b4 * x) + b5 * exp(-b6 * x)

        def jac(x, b1, b2, b3, b4, b5, b6):
            e2, e4, e6 = exp(-b2 * x), exp(-b4 * x), exp(-b6 * x)
            return J(e2, -b1 * x * e2, e4, -b3 * x * e4, e6, -b5 * x * e6)
    elif name.startswith("Gauss"):

        def f(x, b1, b2, b3, b4, b5, b6, b7, b8):
            return b1 * exp(-b2 * x) + b3 * exp(-((x - b4) ** 2) / b5**2) + b6 * exp(-((x - b7) ** 2) / b8**2)

        def jac(x, b1, b2, b3, b4, b5, b6, b7, b8):
            e1 = exp(-b2 * x)
            g1 = exp(-((x - b4) ** 2) / b5**2)
            g2 = exp(-((x - b7) ** 2) / b8**2)
            return J(
                e1,
                -b1 * x * e1,
                g1,
                b3 * g1 * 2 * (x - b4) / b5**2,
                b3 * g1 * 2 * (x - b4) ** 2 / b5**3,
                g2,
                b6 * g2 * 2 * (x - b7) / b8**2,
                b6 * g2 * 2 * (x - b7) ** 2 / b8**3,
            )
    elif name == "DanWood":
        f = lambda x, b1, b2: b1 * x**b2
        jac = lambda x, b1, b2: J(x**b2, b1 * x**b2 * log(x))
    elif name == "Misra1b":
        f = lambda x, b1, b2: b1 * (1 - (1 + b2 * x / 2) ** -2)
        jac = lambda x, b1, b2: J(1 - (1 + b2 * x / 2) ** -2, b1 * x * (1 + b2 * x / 2) ** -3)
    elif name in ("Kirby2", "Hahn1", "Thurber"):
        k = 3 if name == "Kirby2" else 4

        def f(x, *b):
            N, D, _ = _rational(x, b, k, xp)
            return N / D

        def jac(x, *b):
            N, D, pw = _rational(x, b, k, xp)
            return J(*([pw[i] / D for i in range(k)] + [-N * pw[i + 1] / D**2 for i in range(len(b) - k)]))
    elif name == "Nelson":
        f = lambda x, b1, b2, b3: b1 - b2 * x[:, 0] * exp(-b3 * x[:, 1])

        def jac(x, b1, b2, b3):
            e = exp(-b3 * x[:, 1])
            return J(one(e), -x[:, 0] * e, b2 * x[:, 0] * x[:, 1] * e)
    elif name == "MGH17":
        f = lambda x, b1, b2, b3, b4, b5: b1 + b2 * exp(-x * b4) + b3 * exp(-x * b5)

        def jac(x, b1, b2, b3, b4, b5):
            e4, e5 = exp(-x * b4), exp(-x * b5)
            return J(one(x), e4, e5, -b2 * x * e4, -b3 * x * e5)
    elif name == "Misra1c":
        f = lambda x, b1, b2: b1 * (1 - (1 + 2 * b2 * x) ** -0.5)
        jac = lambda x, b1, b2: J(1 - (1 + 2 * b2 * x) ** -0.5, b1 * x * (1 + 2 * b2 * x) ** -1.5)
    elif name == "Misra1d":
        f = lambda x, b1, b2: b1 * b2 * x * (1 + b2 * x) ** -1
        jac = lambda x, b1, b2: J(b2 * x / (1 + b2 * x), b1 * x / (1 + b2 * x) ** 2)
    elif name == "Roszman1":
        f = lambda x, b1, b2, b3, b4: b1 - b2 * x - atan(b3 / (x - b4)) / pi

        def jac(x, b1, b2, b3, b4):
            a = b3 / (x - b4)
            return J(one(x), -x, -(1 / (x - b4)) / (1 + a * a) / pi, -(b3 / (x - b4) ** 2) / (1 + a * a) / pi)
    elif name == "ENSO":

        def f(x, b1, b2, b3, b4, b5, b6, b7, b8, b9):
            t = 2 * pi * x
            return (
                b1
                + b2 * cos(t / 12)
                + b3 * sin(t / 12)
                + b5 * cos(t / b4)
                + b6 * sin(t / b4)
                + b8 * cos(t / b7)
                + b9 * sin(t / b7)
            )

        def jac(x, b1, b2, b3, b4, b5, b6, b7, b8, b9):
            t = 2 * pi * x
            c4, s4, c7, s7 = cos(t / b4), sin(t / b4), cos(t / b7), sin(t / b7)
            return J(
                one(x),
                cos(t / 12),
                sin(t / 12),
                (t / b4**2) * (b5 * s4 - b6 * c4),
                c4,
                s4,
                (t / b7**2) * (b8 * s7 - b9 * c7),
                c7,
                s7,
            )
    elif name == "MGH09":
        f = lambda x, b1, b2, b3, b4: b1 * (x**2 + x * b2) / (x**2 + x * b3 + b4)

        def jac(x, b1, b2, b3, b4):
            N, D = x**2 + x * b2, x**2 + x * b3 + b4
            return J(N / D, b1 * x / D, -b1 * N * x / D**2, -b1 * N / D**2)
    elif name == "Rat42":
        f = lambda x, b1, b2, b3: b1 / (1 + exp(b2 - b3 * x))

        def jac(x, b1, b2, b3):
            e = exp(b2 - b3 * x)
            return J(1 / (1 + e), -b1 * e / (1 + e) ** 2, b1 * x * e / (1 + e) ** 2)
    elif name == "MGH10":
        f = lambda x, b1, b2, b3: b1 * exp(b2 / (x + b3))

        def jac(x, b1, b2, b3):
            e = exp(b2 / (x + b3))
            return J(e, b1 * e / (x + b3), -b1 * e * b2 / (x + b3) ** 2)
    elif name == "Eckerle4":
        f = lambda x, b1, b2, b3: (b1 / b2) * exp(-0.5 * ((x - b3) / b2) ** 2)

        def jac(x, b1, b2, b3):
            z = (x - b3) / b2
            g = exp(-0.5 * z * z)
            return J(g / b2, (b1 / b2**2) * g * (z * z - 1), b1 * g * z / b2**2)
    elif name == "Rat43":
        f = lambda x, b1, b2, b3, b4: b1 / ((1 + exp(b2 - b3 * x)) ** (1 / b4))

        def jac(x, b1, b2, b3, b4):
            e = exp(b2 - b3 * x)
            u = 1 + e
            v = u ** (-1 / b4)
            return J(v, -b1 / b4 * v / u * e, b1 / b4 * v / u * e * x, b1 * v * log(u) / b4**2)
    elif name == "Bennett5":
        f = lambda x, b1, b2, b3: b1 * (b2 + x) ** (-1 / b3)

        def jac(x, b1, b2, b3):
            v = (b2 + x) ** (-1 / b3)
            return J(v, -b1 / b3 * v / (b2 + x), b1 * v * log(b2 + x) / b3**2)
    else:
        raise KeyError(name)
    return f, jac


def lre(est, cert):
    """NIST's log relative error -log10(|est - cert| / |cert|) (absolute when cert is 0),
    capped at 11 (the certified digits); 0 for a non-finite estimate."""
    c = float(cert)
    if not math.isfinite(est):
        return 0.0
    err = abs(est - c) / abs(c) if c != 0 else abs(est)
    if err == 0:
        return 11.0
    return max(0.0, min(11.0, -math.log10(err)))
