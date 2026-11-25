"""Chapter 1: the float64 -> uint64 sort key (sign bit set for +, all bits flipped for -),
checked on the special values; prints a table (value, bits, key) in totalOrder."""
import struct, math
def bits(x): return struct.unpack('<Q', struct.pack('<d', x))[0]
def key(x):
    b = bits(x)
    return (~b) & (2**64 - 1) if b >> 63 else b | (1 << 63)
neg_nan = struct.unpack('<d', struct.pack('<Q', 0xFFF8000000000000))[0]
vals = [neg_nan, -math.inf, -1.0, -5e-324, -0.0, 0.0, 5e-324, 1.0, 2.0, math.inf, math.nan]
names = ['-NaN', '-inf', '-1', '-min sub', '-0', '+0', '+min sub', '1', '2', '+inf', '+NaN']
prev = -1
for n, v in zip(names, vals):
    k = key(v); assert k > prev; prev = k
    print(f"{n:9s} {bits(v):016X} {k:016X}")
print("keys strictly increase in totalOrder: ok")
