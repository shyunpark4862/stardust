"""Sterbenz 그림: [1/4,4]^2의 무작위 binary64 쌍에서 x - y의 뺄셈이 정확한지(Fraction으로 확인)."""
import random
from fractions import Fraction as F
random.seed(10)
rows = []; inside_bad = 0; n_in = 0; n_out_exact = 0; n_out = 0
for _ in range(4000):
    x = 2 ** random.uniform(-2, 2); y = 2 ** random.uniform(-2, 2)
    exact = F(x - y) == F(x) - F(y)
    inside = y / 2 <= x <= 2 * y
    if inside:
        n_in += 1; inside_bad += not exact
    else:
        n_out += 1; n_out_exact += exact
    rows.append((x, y, int(exact)))
with open('sterbenz.dat', 'w') as f:
    f.write('x y exact\n')
    for r in rows: f.write('%.6f %.6f %d\n' % r)
print('inside', n_in, 'inexact', inside_bad, '| outside', n_out, 'exact', n_out_exact)
