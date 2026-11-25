"""Terms of CF1 (J'_nu/J_nu) until Lentz's test |delta - 1| <= eps (the forward count of jy_temme),
over x for nu = 0.3, 10, 100. Output: cf1terms.dat."""
import math, numpy as np
EPS = 2.0 ** -52; TINY = 1e-300
def terms(nu, x):
    xi2 = 2.0 / x; b = xi2 * nu; d = 0.0; c = max(nu / x, TINY)
    for i in range(1, 10 ** 7):
        b += xi2; d = b - d
        if abs(d) < TINY: d = TINY
        c = b - 1.0 / c
        if abs(c) < TINY: c = TINY
        d = 1.0 / d; de = c * d
        if abs(de - 1.0) <= EPS: return i
    return None
with open("cf1terms.dat", "w") as f:
    f.write("x t0 t10 t100\n")
    for x in np.geomspace(0.1, 3000, 60):
        f.write("%.4f %d %d %d\n" % (x, terms(0.3, x), terms(10.3, x), terms(100.3, x)))
for x in (1, 10, 100, 1000):
    print(x, terms(0.3, x), terms(10.3, x), terms(100.3, x))
