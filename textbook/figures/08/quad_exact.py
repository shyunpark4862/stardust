import stardust as sd, math
for c in (1.0, 1.3, 0.7, 2.9, math.pi):
    f = lambda x, c=c: (x - c) ** 2 + 1.0
    out = []
    for meth, kw in (("brent", {}), ("golden", {}), ("bounded", {"bounds": (0.0, 3.5)}), ("brent", {"bracket": (0.0, 0.5)})):
        r = sd.optimize.minimize_scalar(f, method=meth, **kw)
        out.append(f"{meth}{'(br 0,.5)' if 'bracket' in kw else ''} {(float(r.x)-c)/c:+.1e}")
    print(f"c={c:.4f}: " + "; ".join(out))
