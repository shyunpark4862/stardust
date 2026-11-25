"""Shared pieces of the Hurwitz zeta checks: the exact reference (integer s) and Cephes' constants."""
from fractions import Fraction as F
import math

EPS = 2.0**-53
A = [12.0, -720.0, 30240.0, -1209600.0, 47900160.0, -1.8924375803183791606e9, 7.47242496e10,
     -2.950130727918164224e12, 1.1646782814350067249e14, -4.5979787224074726105e15,
     1.8152105401943546773e17, -7.1661652561756670113e18]


def _bernoulli(nmax):
    B = [F(1)]
    for n in range(1, nmax + 1):
        B.append(-sum(math.comb(n + 1, k) * B[k] for k in range(n)) / (n + 1))
    return B


BN = _bernoulli(70)


def zeta_ref(s, q):
    Q = F(q)
    a = Q + 60
    t = sum((Q + k) ** (-s) for k in range(60)) + a ** (1 - s) / (s - 1) + F(1, 2) * a ** (-s)
    for j in range(1, 31):
        t += BN[2 * j] / math.factorial(2 * j) * math.prod(range(s, s + 2 * j - 1)) * a ** (-s - 2 * j + 1)
    return t
