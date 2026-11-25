"""The quote mask of a CSV fragment and its prefix xor (Section 2.6; csrc/frame/io/csv.cpp
prefix_xor, which multiplies the mask by all ones without carries, PMULL). Prints the rows of
the figure: bytes, quote mask q, prefix_xor(q), separators and line feeds, those outside quotes.
"""

text = 'x,"a,b",y,"c""d"\n'


def clmul(a, b, width=64):
    r = 0
    for i in range(width):
        if (b >> i) & 1:
            r ^= a << i
    return r


def mask(ch):
    return sum(1 << k for k, c in enumerate(text) if c == ch)


q = mask('"')
p = clmul(q, (1 << 64) - 1) & ((1 << 64) - 1)
# prefix xor by the shift-and-xor ladder (the portable alternative)
r = q
for s in (1, 2, 4, 8, 16, 32):
    r ^= (r << s) & ((1 << 64) - 1)
assert r == p
ends = mask(",") | mask("\n")
n = len(text)
row = lambda m: " ".join(str((m >> k) & 1) for k in range(n))
print("bytes ", " ".join(c if c != "\n" else "LF" for c in text))
print("q     ", row(q))
print("in    ", row(p))
print("ends  ", row(ends))
print("field ", row(ends & ~p))
