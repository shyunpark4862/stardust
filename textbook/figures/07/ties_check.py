import numpy as np, itertools, math
from fractions import Fraction as F
import stardust as sd
A=sd.asarray
rng=np.random.default_rng(3)
# 1. MWU auto at c=d=20: which method?
x=rng.normal(size=20); y=rng.normal(size=20)+0.5
r=sd.htest.mannwhitneyu(A(x),A(y)); print("mwu auto 20,20:", r.method, float(r.pvalue))
r=sd.htest.mannwhitneyu(A(x),A(y),method='exact'); print("mwu exact 20,20:", r.method, float(r.pvalue))
# 2. wilcoxon with ties exact vs brute force
d=np.array([1,1,2,2,2,3,4,4,5,6,-1,-2,-3,-3,-4.0])
r=sd.htest.wilcoxon(A(d),method='exact',alternative='less')
ad=np.abs(d); 
from collections import Counter
order=np.argsort(ad,kind='stable'); ranks2=np.empty(len(d))
# doubled midranks
vals=sorted(set(ad)); rk={}
pos=0
for v in vals:
    t=int((ad==v).sum()); rk[v]=2*pos+t+1; pos+=t
a=[rk[v] for v in ad]
T2=sum(rk[v] for v in d if v>0)
cnt=0; tot=0
# brute: 2^15
le=0
for bits in itertools.product([0,1],repeat=len(a)):
    s=sum(ai for ai,b in zip(a,bits) if b)
    le+= s<=T2
print("wilcoxon ties exact:", float(r.pvalue), le/2**len(a), r.method)
# symmetric?
from collections import Counter
C=Counter(sum(ai for ai,b in zip(a,bits) if b) for bits in itertools.product([0,1],repeat=len(a)))
S=sum(a); print("symmetric with ties:", all(C[s]==C[S-s] for s in C))
# 3. MWU ties exact vs brute
x=np.array([1,1,2,3,3,5.0]); y=np.array([2,2,3,4,4,4,6,7.0])
r=sd.htest.mannwhitneyu(A(x),A(y),method='exact',alternative='less')
z=np.concatenate([x,y]); 
vals=sorted(set(z)); rk={}; pos=0
for v in vals:
    t=int((z==v).sum()); rk[v]=2*pos+t+1; pos+=t
sc=[rk[v] for v in z]; R1=sum(rk[v] for v in x)
le=0;tot=0
for S_ in itertools.combinations(range(len(z)),len(x)):
    tot+=1; le+= sum(sc[i] for i in S_)<=R1
print("mwu ties exact:", float(r.pvalue), le/tot)
# 4. ansari no ties exact vs brute
x=rng.normal(size=7); y=rng.normal(size=6)*2
r=sd.htest.ansari(A(x),A(y),method='exact',alternative='two-sided')
N=13; z=np.concatenate([x,y]); rkk=np.argsort(np.argsort(z))+1
scr=np.minimum(rkk,N+1-rkk); AB=scr[:7].sum()
vals=[sum(scr[list(S_)]) if False else None for S_ in []]
allsc=list(range(1,N+1)); allsc=[min(r_,N+1-r_) for r_ in allsc]
dist=Counter(sum(allsc[i] for i in S_) for S_ in itertools.combinations(range(N),7))
tot=sum(dist.values()); cdf=sum(v for k,v in dist.items() if k<=AB)/tot; sf=sum(v for k,v in dist.items() if k>=AB)/tot
print("ansari exact:", float(r.pvalue), min(1,2*min(cdf,sf)))
# 5. ks ties
x=np.array([1,2,2,3,5,5,5,8.0]); y=np.array([2,3,3,4,6,7,7,9,9,10.0])
r=sd.htest.ks_2samp(A(x),A(y),method='exact')
z=np.concatenate([x,y]); m,n=8,10
L=math.lcm(m,n); al,be=L//m,L//n
def stat(lab):
    xs=z[lab]; ys=z[~lab]; h=0
    for v in sorted(set(z)):
        i=(xs<=v).sum(); j=(ys<=v).sum(); h=max(h,abs(i*al-j*be))
    return h
lab=np.zeros(18,bool); lab[:8]=True; h0=stat(lab)
cnt=0;tot=0
for S_ in itertools.combinations(range(18),8):
    lab=np.zeros(18,bool); lab[list(S_)]=True; tot+=1; cnt+= stat(lab)>=h0
print("ks ties exact:", float(r.pvalue), cnt/tot)
