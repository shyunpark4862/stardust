"""Quotients and remainders of the three division conventions (Table tab:div-conventions)
and the staircase data of Figure fig:div-steps (b = 3, a = -7..7)."""
import math


def T(a, b):  # truncate toward zero (C, C++)
    q = abs(a) // abs(b) * (1 if (a >= 0) == (b > 0) else -1)
    return q, a - q * b


def F(a, b):  # floor (Python, numpy, stardust)
    return a // b, a % b


def E(a, b):  # Euclidean: 0 <= r < |b|
    r = a % abs(b)
    return (a - r) // b, r


for a, b in ((7, 2), (-7, 2), (7, -2), (-7, -2)):
    print(a, b, "T", T(a, b), "F", F(a, b), "E", E(a, b))
for a in range(-7, 8):
    print(a, T(a, 3), F(a, 3))
