"""Euclid step counts for the figure in Section 2.2 (Lame's bound).

For each b, S(b) = max over a > b of the number of division steps of Euclid's
algorithm on (a, b); since the first step maps (a, b) to (b, a mod b), S(b) =
1 + max_{0 <= r < b} steps(b, r). A(b) is the mean over 0 <= r < b of
1 + steps(b, r) (a uniform residue a mod b). Writes euclid_steps.dat.
"""
import math

N = 2000
steps = {}


def st(a, b):
    k = 0
    while b:
        a, b = b, a % b
        k += 1
    return k


phi = (1 + 5 ** 0.5) / 2
fib = [0, 1, 1]
while fib[-1] < 10 * N:
    fib.append(fib[-1] + fib[-2])
with open("euclid_steps.dat", "w") as f:
    f.write("b max mean lame\n")
    for b in range(1, N + 1):
        s = [1 + st(b, r) for r in range(b)]
        lame = max(k for k in range(1, len(fib) - 1) if fib[k + 1] <= b)  # F_{k+1} <= b
        f.write(f"{b} {max(s)} {sum(s) / b:.4f} {lame}\n")
        assert max(s) <= lame
# where the bound is reached
print([(b, max(1 + st(b, r) for r in range(b))) for b in fib[2:18] if b <= N])
print("mean at N:", sum(1 + st(N, r) for r in range(N)) / N, "12ln2/pi^2 ln N =", 12 * math.log(2) / math.pi ** 2 * math.log(N))
