"""Chapter 1: a tie-blind argmax (keeps the right operand on equal values when merging chunks)
gives different answers for different chunkings; the (value, index) total order does not."""
import random
random.seed(1)
x = [random.choice([1, 2, 3, 5, 5, 5]) for _ in range(40)]
def local(lo, hi):  # first maximum inside a chunk
    b = lo
    for i in range(lo, hi):
        if x[i] > x[b]: b = i
    return b
def merge_naive(a, b): return b if x[b] >= x[a] else a      # ties: right wins
def merge_total(a, b): return b if (x[b], -b) > (x[a], -a) else a  # (value, -index) order
for chunks in (1, 2, 3, 4, 5, 8):
    size = -(-len(x) // chunks)
    parts = [local(s, min(len(x), s + size)) for s in range(0, len(x), size)]
    na, tt = parts[0], parts[0]
    for p in parts[1:]: na, tt = merge_naive(na, p), merge_total(tt, p)
    print(chunks, 'naive', na, 'total-order', tt)
print('positions of 5:', [i for i, v in enumerate(x) if v == 5])
