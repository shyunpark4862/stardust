import numpy as np, math, sys
from fractions import Fraction as F
sys.path.insert(0,'/Users/sanghyunpark/Desktop/stardust/textbook/figures/07')
from exact import mahonian, qbinom
import stardust as sd
A=sd.asarray
rng=np.random.default_rng(8)
for n,rho in ((200,-0.6),(200,0.0),(400,-1.0)):
    x=rng.normal(size=n); y=x*rho+rng.normal(size=n)
    r=sd.htest.kendalltau(A(x),A(y),method='exact',alternative='less')
    o=np.argsort(x); yy=y[o]
    K=int(sum((yy[i]>yy[i+1:]).sum() for i in range(n)))
    # tail via truncated mahonian to b=max(K, tot-K)?
    tot=n*(n-1)//2
    c=min(K,tot-K)
    M=[1]
    for j in range(2,n+1):
        new=[0]*(c+1); run=0
        for k in range(c+1):
            run+= M[k] if k<len(M) else 0
            if 0<=k-j<len(M): run-=M[k-j]
            new[k]=run
        M=new
    f=math.factorial(n)
    le=F(sum(M),f)
    # alternative 'less' : tau small => K large; p = P(K>=K_obs) = P(K<=tot-K)
    p_ex = le if K>=tot-K else 1-le+F(M[c],f)
    p=float(r.pvalue)
    print(n, "K",K,"p",p, "rel", float(abs(F(p)-p_ex)/p_ex))
