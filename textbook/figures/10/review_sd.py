"""10장 검토 반영용 측정 (복제본 stardust, numpy). 결과는 review_sd.txt."""
import struct, time, math, numpy as np, stardust as sd
def bits(x): return '0x%016x' % struct.unpack('<Q', struct.pack('<d', float(x)))[0]
def fb(b): return struct.unpack('<d', struct.pack('<Q', b))[0]
def one(f, *a): return f(*[sd.asarray([x]) for x in a]).tolist()[0]
def arr_from_bits(bs): return sd.asarray(np.array(bs, dtype=np.uint64).view(np.float64))
out = []
def P(*a): out.append(' '.join(str(x) for x in a)); print(*a)
pa, pb = 0x7ff8000000000aaa, 0x7ff8000000000bbb
snan = 0x7ff0000000000001
P('add(pa,pb)', bits(sd.add(arr_from_bits([pa]), arr_from_bits([pb])).tolist()[0]))
P('add(pb,pa)', bits(sd.add(arr_from_bits([pb]), arr_from_bits([pa])).tolist()[0]))
P('add(snan,0)', bits(sd.add(arr_from_bits([snan]), sd.asarray([0.0])).tolist()[0]))
z = sd.asarray([0.0]); inf = sd.asarray([math.inf])
P('0/0', bits(sd.divide(z, z).tolist()[0]), 'inf-inf', bits(sd.subtract(inf, inf).tolist()[0]), 'sqrt(-1)', bits(sd.sqrt(sd.asarray([-1.0])).tolist()[0]))
P('sd.log(-1)', bits(sd.log(sd.asarray([-1.0])).tolist()[0]), 'np.log(-1)', bits(np.log(np.array([-1.0]))[0]) if True else '')
P('negative(nan)', bits(sd.negative(sd.asarray([math.nan])).tolist()[0]))
P('signbit(-nan)', sd.signbit(sd.negative(sd.asarray([math.nan]))).tolist()[0])
# maximum with snan, scalar vs vector
for n in (3, 5):
    xb = [snan] + [0x3ff0000000000000] * (n - 1)
    r = sd.maximum(arr_from_bits(xb), sd.asarray([0.0] * n)).tolist()
    P('maximum(snan..) n=%d' % n, bits(r[0]))
r = sd.maximum(arr_from_bits([snan, 0, 0, 0, snan]), sd.asarray([0.0] * 5)).tolist()
P('maximum n=5 elem0', bits(r[0]), 'elem4', bits(r[4]))
# signed zero reductions
P('sum([-0,-0])', sd.sum(sd.asarray([-0.0, -0.0])).tolist() if hasattr(sd.sum(sd.asarray([-0.0])), 'tolist') else sd.sum(sd.asarray([-0.0, -0.0])))
for name, f, x in [('cumulative_sum', sd.cumulative_sum, [-0.0, -0.0]), ('prod', sd.prod, [-0.0]), ('min', sd.min, [0.0, -0.0]),
                   ('max', sd.max, [-0.0, 0.0]), ('abs', sd.abs, [-0.0]), ('sign', sd.sign, [-0.0])]:
    r = f(sd.asarray(x)); P(name, x, r.tolist() if hasattr(r, 'tolist') else r)
for name in ('round', 'floor', 'ceil', 'trunc'):
    f = getattr(sd, name)
    P(name, f(sd.asarray([2.5, -2.5, -0.4, -0.0, -0.5, 4503599627370495.5])).tolist())
# complex
for name, a, b in [('mul', complex(math.inf, 0), complex(math.inf, 0)), ('mul', complex(math.inf, math.nan), 2 + 0j),
                   ('mul', 1e300 + 1e300j, 1e300 + 1e300j), ('div', 1 + 1j, 0j), ('div', 0j, 0j)]:
    f = sd.multiply if name == 'mul' else sd.divide
    P('complex', name, a, b, f(sd.asarray([a]), sd.asarray([b])).tolist()[0])
P('abs(inf+nanj)', sd.abs(sd.asarray([complex(math.inf, math.nan)])).tolist()[0], 'abs(3e-310+4e-310j)', sd.abs(sd.asarray([3e-310 + 4e-310j])).tolist()[0])
# cmul fma
rng = np.random.default_rng(3); n = 20000
a = rng.standard_normal(n) + 1j * rng.standard_normal(n); b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
plain = a.real * b.real - a.imag * b.imag
fm = np.array([math.fma(x, y, -(u * v)) for x, y, u, v in zip(a.real, b.real, a.imag, b.imag)])
s = np.asarray(sd.multiply(sd.asarray(a), sd.asarray(b))).real; npv = (a * b).real
P('cmul: plain!=fma %.4f' % np.mean(plain != fm), 'np==fma %.4f' % np.mean(npv == fm), 'sd==np %.4f' % np.mean(s == npv), 'sd==fma %.4f' % np.mean(s == fm))
# import narrower types, bool
P('float32 0.1 ->', sd.asarray(np.array([0.1], dtype=np.float32)).tolist()[0])
P('float16 1e-7 ->', sd.asarray(np.array([1e-7], dtype=np.float16)).tolist()[0])
for dt in (np.uint64, np.longdouble):
    try: sd.asarray(np.array([1], dtype=dt)); P(dt.__name__, 'ok')
    except Exception as e: P(dt.__name__, type(e).__name__, e)
P('bool', sd.asarray([math.nan, -0.0, 0.0]).astype(sd.bool).tolist())
P('nextafter(inf,0)', sd.nextafter(sd.asarray([math.inf]), sd.asarray([0.0])).tolist()[0])
P('int->float 2^62+2^9', sd.asarray([2**62 + 2**9]).astype(sd.float64).tolist()[0] == 2.0**62, '2^62+2^9+1', sd.asarray([2**62 + 2**9 + 1]).astype(sd.float64).tolist()[0] == 2.0**62 + 2**10)
P('mean [2^53+1,0,0]', repr(sd.mean(sd.asarray([2**53 + 1, 0, 0]))), 'true', (2**53 + 1) // 3, (2**53 + 1) % 3)
# timing
def best(f, k=30):
    t = []
    for _ in range(k):
        s0 = time.perf_counter(); f(); t.append(time.perf_counter() - s0)
    return min(t) * 1e6
x = rng.uniform(-1e6, 1e6, 10**6); X = sd.asarray(x); Y = sd.asarray(x + 1)
for th in (1, None):
    if th: sd.set_num_threads(1)
    lab = '1thr' if th else 'default'
    P(lab, 'nextafter sd %.0f np %.0f' % (best(lambda: sd.nextafter(X, Y)), best(lambda: np.nextafter(x, x + 1))))
    P(lab, 'astype sd %.0f np %.0f' % (best(lambda: X.astype(sd.int64)), best(lambda: x.astype(np.int64))))
    P(lab, 'isinf sd %.1f np %.1f isnan sd %.1f' % (best(lambda: sd.isinf(X)), best(lambda: np.isinf(x)), best(lambda: sd.isnan(X))))
    if th: sd.set_num_threads(0) if False else None
sd.set_num_threads(1)
m = 16384
nrm = sd.asarray(rng.uniform(1, 2, m)); sub = sd.asarray(rng.uniform(1, 2, m) * 1e-310); h = sd.asarray([0.5] * m); o = sd.asarray([0.0] * m)
P('L1 1thr multiply normal %.2f subnormal %.2f us' % (best(lambda: sd.multiply(nrm, h, out=o), 2000), best(lambda: sd.multiply(sub, h, out=o), 2000)))
open(__file__.replace('.py', '.txt'), 'w').write('\n'.join(out) + '\n')
