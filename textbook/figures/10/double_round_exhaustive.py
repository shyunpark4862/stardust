from fractions import Fraction as F
from math import isqrt
import sys
def rnd(x, p):
    # round positive Fraction x to p bits RNE, unbounded exponent
    if x == 0: return x
    s = 1 if x > 0 else -1; x = abs(x)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if F(2)**e > x: e -= 1
    # x in [2^e, 2^{e+1})
    sc = F(2)**(p-1-e)
    y = x*sc
    n = y.numerator // y.denominator; r = y - n
    if r > F(1,2) or (r == F(1,2) and n % 2): n += 1
    return s*F(n)/sc
def rnd_sqrt(a, p):
    # a Fraction>0; round sqrt(a) to p bits RNE
    # scale so that sqrt known to many bits: use integer sqrt with large scale
    K = 400
    num = a.numerator * 4**K
    t = num // a.denominator
    r = isqrt(t)
    exact = (r*r*a.denominator == num)
    v = F(r, 2**K)
    if not exact: v += F(1, 2**(K+1))  # sticky: strictly between r and r+1
    return rnd(v, p)
def check(p, qs):
    nums = []
    for m in range(2**(p-1), 2**p):
        for e in range(-p-2, p+3):
            nums.append(F(m)*F(2)**e)
    bad = {op:set() for op in '+-*/s'}
    for a in nums:
        for b in nums:
            for op in '+-*/':
                x = {'+':a+b,'-':a-b,'*':a*b,'/':a/b}[op]
                if x == 0: continue
                d = rnd(x,p)
                for q in qs:
                    if rnd(rnd(x,q),p) != d: bad[op].add(q)
        for q in qs:
            pass
    for a in nums:
        d = rnd_sqrt(a,p)
        for q in qs:
            if rnd(rnd_sqrt(a,q),p) != d: bad['s'].add(q)
    return bad
p = int(sys.argv[1])
qs = list(range(p+1, 2*p+4))
b = check(p, qs)
for op in b: print(p, op, 'failing q:', sorted(b[op]))
