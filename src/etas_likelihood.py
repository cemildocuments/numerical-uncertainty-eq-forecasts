"""Conditional temporal ETAS likelihood with analytic derivatives.

Parameters: log(mu), log(n), log(c), log(p-1), r=alpha/b.
Times are days; input history includes all retained events before window end.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def nll_gradient(theta,times,excess,beta,start,end):
    mu=math.exp(theta[0]);n=math.exp(theta[1]);c=math.exp(theta[2])
    pm1=math.exp(theta[3]);p=1+pm1;r=theta[4]
    N=len(times);kappa=np.empty(N);dr=np.empty(N)
    for j in range(N):
        kappa[j]=n*(1-r)*math.exp(r*beta*excess[j])
        dr[j]=-1/(1-r)+beta*excess[j]
    value=mu*(end-start);grad=np.zeros(5);grad[0]=value;count=0
    for j in range(N):
        if times[j]>=end:continue
        age0=max(start-times[j],0.0);age1=end-times[j]
        z0=math.log1p(age0/c);z1=math.log1p(age1/c)
        s0=math.exp(-pm1*z0);s1=math.exp(-pm1*z1)
        mass=s0-s1;term=kappa[j]*mass
        value+=term;grad[1]+=term;grad[4]+=term*dr[j]
        grad[2]+=kappa[j]*pm1*(s0*age0/(c+age0)-s1*age1/(c+age1))
        grad[3]+=kappa[j]*pm1*(-s0*z0+s1*z1)
        if times[j]<start:continue
        count+=1;lam=mu;dlam=np.zeros(5);dlam[0]=mu
        for i in range(j):
            lag=times[j]-times[i]
            if lag<=0:continue
            z=math.log1p(lag/c)
            term=kappa[i]*pm1/c*math.exp(-p*z)
            lam+=term;dlam[1]+=term;dlam[2]+=term*(-1+p*lag/(c+lag))
            dlam[3]+=term*(1-pm1*z);dlam[4]+=term*dr[i]
        value-=math.log(lam)
        grad-=dlam/lam
    if count==0:raise ValueError('Empty likelihood period')
    return value/count,grad/count
