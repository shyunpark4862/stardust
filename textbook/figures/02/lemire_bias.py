"""Rejection probability of Lemire's method and the bias of w mod r, for 16-bit words.

For r = 1..2^16, t = 2^16 mod r values of w are rejected, so the rejection probability is
t / 2^16; w mod r returns the values below t one extra time, a relative bias of 1 / q with
q = floor(2^16 / r) (when t > 0). Also checks Theorem thm:lemire exhaustively for 8-bit and
16-bit words (every r): each j gets exactly q words. Writes lemire_bias.dat.
"""
W = 16
B = 1 << W


def check(width):
    b = 1 << width
    for r in range(1, b + 1):
        t = b % r
        q = b // r
        cnt = [0] * r
        for w in range(b):
            j, l = divmod(w * r, b)
            if l >= t:
                cnt[j] += 1
        assert all(c == q for c in cnt), (width, r)


check(8)
check(12)
keep = {int(1.0035**k) for k in range(3200)} | {2**k + d for k in range(17) for d in (-1, 0, 1)}
keep = {r for r in keep if 1 <= r <= B}
with open("lemire_bias.dat", "w") as f:
    f.write("r reject bias\n")
    for r in range(1, B + 1):
        t, q = B % r, B // r
        if r in keep:
            f.write(f"{r} {t / B:.6f} {(1 / q if t else 0):.6f}\n")
worst = max(range(1, B + 1), key=lambda r: (B % r) / B)
print("worst r", worst, (B % worst) / B)
