"""Condition of Neumann's normalizing sum (x/2)^mu / Gamma(mu+1) = J_mu + sum_k (mu+2k)(mu+1)_{k-1}/k! J_{mu+2k}:
kappa = sum |terms| / |sum| over x, mu = 0.3 and -0.4 (Decimal series). Output: neumann.dat."""
import math
from decimal import Decimal as D
from dref import jv as jref, dec, gamma

def kappa(mu, x):
    M = dec(mu); tot = abs(jref(M, x)); s = jref(M, x); b = D(1); k = 1
    top = x + 12 * x ** (1 / 3) + 20
    while mu + 2 * k < top:
        if k > 1: b = b * (M + k - 1) / k
        t = (M + 2 * k) * b * jref(M + 2 * k, x)
        tot += abs(t); s += t; k += 1
    lhs = (dec(x) / 2) ** M / gamma(M + 1)
    return float(tot / abs(s)), float(abs(s - lhs) / abs(lhs))

with open("neumann.dat", "w") as f:
    f.write("x k03 km04\n")
    for x in (0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100):
        a, ea = kappa(0.3, x); b, eb = kappa(-0.4, x)
        f.write("%g %.4f %.4f\n" % (x, a, b)); print(x, "%.2f %.2f   identity residual %.1e %.1e" % (a, b, ea, eb))
