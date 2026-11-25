from fractions import Fraction as F
def rnd(x,p,mode):
    if x==0: return x
    s=1 if x>0 else -1; x=abs(x)
    e=x.numerator.bit_length()-x.denominator.bit_length()
    if F(2)**e>x: e-=1
    sc=F(2)**(p-1-e); y=x*sc; n=y.numerator//y.denominator; r=y-n
    if r>F(1,2) or (r==F(1,2) and (mode=='up' or n%2)): n+=1
    return s*F(n)/sc
p=4
nums=[s*F(m)*F(2)**e for m in range(8,16) for e in range(-6,7) for s in (1,-1)]
for mode in ('even','up'):
    bad=0; drift=None
    for u in nums:
        for v in nums:
            w=u; seq=[]
            for k in range(6):
                w=rnd(rnd(w+v,p,mode)-v,p,mode); seq.append(w)
            if len(set(seq))>1:
                bad+=1; drift=drift or (u,v,seq)
    print(mode,bad,drift)
