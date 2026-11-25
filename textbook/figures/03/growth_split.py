"""Split growth.dat (from growth.c) into sort.dat and chase.dat, and fit log-log slopes
(least squares on the largest sizes) for the table of Section 3.1."""
import math

lines = open("growth.dat").read().split("\n")
sort, chase, cur = [], [], None
for ln in lines:
    if ln.startswith("# sort"):
        cur = sort
    elif ln.startswith("# chase"):
        cur = chase
    elif ln.strip():
        cur.append([float(v) for v in ln.split()])
with open("sort.dat", "w") as f:
    f.write("n ins merge\n")
    for n, i, m in sort:
        f.write(f"{int(n)} {i if i > 0 else 'nan'} {m}\n")
with open("chase.dat", "w") as f:
    f.write("bytes ns\n")
    for b, t in chase:
        f.write(f"{int(b)} {t}\n")


def slope(pts):
    xs = [math.log(p[0]) for p in pts]
    ys = [math.log(p[1]) for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


ins = [(n, i) for n, i, m in sort if i > 0 and n >= 8192]
mer = [(n, m) for n, i, m in sort if n >= 4096]
print("insertion slope (n >= 8192):", round(slope(ins), 3))
print("merge slope (n >= 4096):", round(slope(mer), 3))
print("n log n local slope at n = 2^16: 1 + 1/ln n =", round(1 + 1 / math.log(2**16), 3))
print("merge time / (n log2 n) at 2^12 and 2^20 (ns):", [round(m / (n * math.log2(n)) * 1e9, 3) for n, i, m in sort if n in (4096, 1048576)])
