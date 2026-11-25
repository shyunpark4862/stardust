import random
random.seed(5)
def rnd(M, keep):  # round positive integer M to 'keep' significant bits, RNE, return integer value
    k = M.bit_length()
    if k <= keep: return M
    d = k-keep; q = M >> d; r = M & ((1<<d)-1); h = 1<<(d-1)
    if r > h or (r == h and q & 1): q += 1
    return q << d
N=200000
pairs=[(random.getrandbits(52)|1<<52, random.getrandbits(52)|1<<52) for _ in range(N)]
for q in (54,56,60,64,70,80,100,105,106):
    bad=sum(1 for a,b in pairs if rnd(rnd(a*b,q),53)!=rnd(a*b,53))
    print("mul q=%d bad=%d frac=%.2e"%(q,bad,bad/N))
# addition with exponent gaps
adds=[]
for _ in range(N):
    a=random.getrandbits(52)|1<<52; b=random.getrandbits(52)|1<<52; s=random.randint(0,80)
    adds.append((a<<s)+b)
for q in (54,60,64,80,100,105,106):
    bad=sum(1 for S in adds if rnd(rnd(S,q),53)!=rnd(S,53))
    print("add q=%d bad=%d frac=%.2e"%(q,bad,bad/N))
