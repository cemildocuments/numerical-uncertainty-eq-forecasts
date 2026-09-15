"""Exact Brier interval propagation and paired dependence-aware resampling."""
from fractions import Fraction
import numpy as np
from flint import arb,ctx
from validate_ball_zero import encode


def rational(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator))
def decode(x):return Fraction(int(x['numerator']),int(x['denominator']))
def endpoint(x,side):
    e=x[side];return Fraction(int(e['mantissa']))*Fraction(2)**e['exponent']


def probabilities(method):
    if method['status']=='numerical_failure':return None
    if 'exceedance_lower' in method:
        lo=decode(method['exceedance_lower']);hi=decode(method['exceedance_upper'])
    elif 'poisson_lower' in method:
        lo=endpoint(method['poisson_lower'],'lower');hi=endpoint(method['poisson_upper'],'upper')
    elif 'probability' in method:
        lo=hi=Fraction(str(method['probability']))
    else:return None
    if not 0<=lo<=hi<=1:raise ValueError('Invalid probability bounds')
    return lo,hi


def brier(lo,hi,y):
    if y not in (0,1) or not 0<=lo<=hi<=1:raise ValueError('Invalid score inputs')
    return ((1-hi)**2,(1-lo)**2) if y else (lo**2,hi**2)


def log_loss(lo,hi,y):
    if y not in (0,1) or not 0<=lo<=hi<=1:raise ValueError('Invalid score inputs')
    # Event probability of the observed outcome, increasing from a to b.
    a,b=(lo,hi) if y else (1-hi,1-lo)
    old=ctx.prec;ctx.prec=160
    try:
        lower='infinity' if b==0 else encode((-(arb(b.numerator)/b.denominator).log()).lower())
        upper='infinity' if a==0 else encode((-(arb(a.numerator)/a.denominator).log()).upper())
        return dict(lower=lower,upper=upper)
    finally:ctx.prec=old


def paired_brier(a,b,y):
    alo,ahi=brier(*a,y);blo,bhi=brier(*b,y)
    am=(a[0]+a[1])/2;bm=(b[0]+b[1])/2
    return alo-bhi,(am-y)**2-(bm-y)**2,ahi-blo


def block_bootstrap(contrasts,block_length=13,replicates=10000,seed=20260913):
    values=np.asarray(contrasts,dtype=float)
    if values.ndim!=2 or values.shape[1]!=3 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('Require finite lower/midpoint/upper contrast triples')
    if np.any(values[:,0]>values[:,1]) or np.any(values[:,1]>values[:,2]):raise ValueError('Unordered contrast bounds')
    if type(block_length) is not int or block_length<1 or type(replicates) is not int or replicates<1:raise ValueError('Invalid resampling plan')
    N=len(values);blocks=(N+block_length-1)//block_length;rng=np.random.default_rng(seed)
    samples=np.empty((replicates,3));offsets=np.arange(block_length)
    for start in range(0,replicates,250):
        count=min(250,replicates-start)
        starts=rng.integers(0,N,size=(count,blocks))
        indices=((starts[:,:,None]+offsets)%N).reshape(count,-1)[:,:N]
        samples[start:start+count]=values[indices].mean(axis=1)
    percentiles=np.quantile(samples,[.025,.975],axis=0)
    return dict(block_length=block_length,replicates=replicates,seed=seed,origins=N,
                sample_mean=values.mean(axis=0).tolist(),percentile_025=percentiles[0].tolist(),percentile_975=percentiles[1].tolist(),
                numerical_envelope=[float(percentiles[0,0]),float(percentiles[1,2])],
                interpretation='Dependence-sensitive empirical bootstrap uncertainty; not an exact finite-sample coverage guarantee')
