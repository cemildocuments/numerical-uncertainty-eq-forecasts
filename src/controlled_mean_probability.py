"""Time-discretization bounds for ETAS mean count and Poisson conversion.

Future marks are iid unbounded GR; b>alpha, n<1. The returned Poisson
probability encloses a model approximation, not the actual ETAS probability.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import hashlib
import time
from flint import arb, ctx
from validated_probability import decimal_value, positive_integer
from validate_ball_zero import encode


def validate(config):
    c=deepcopy(config)
    q={key:decimal_value(c[key],key) for key in
       ('horizon','mu','c','p','m0','threshold','b','alpha','productivity')}
    if any(q[x]<=0 for x in ('horizon','c','b')) or q['p']<=1:
        raise ValueError('Require positive horizon,c,b and p>1')
    if any(q[x]<0 for x in ('mu','alpha','productivity')) or q['threshold']<q['m0']:
        raise ValueError('Require nonnegative rates, alpha and threshold>=m0')
    if q['b']<=q['alpha'] or q['productivity']*q['b']/(q['b']-q['alpha'])>=1:
        raise ValueError('Require b>alpha and branching mean<1')
    positive_integer(c['time_bins'],'time_bins')
    positive_integer(c['precision_bits'],'precision_bits')
    if c['precision_bits']<64:raise ValueError('Require at least64 precision bits')
    if not isinstance(c['history'],list):raise ValueError('history must be a list')
    for event in c['history']:
        if decimal_value(event['time'],'history time')>=0 or decimal_value(event['magnitude'],'history magnitude')<q['m0']:
            raise ValueError('History must precede origin with magnitude>=m0')
    return c


def enclose(config):
    c=validate(config);old=ctx.prec
    try:
        ctx.prec=c['precision_bits'];K=c['time_bins']
        H=arb(c['horizon']);dt=H/K;scale=arb(c['c']);p=arb(c['p'])
        b=arb(c['b']);alpha=arb(c['alpha']);base=arb(c['productivity'])
        n=base*b/(b-alpha)
        if not n<1:raise ArithmeticError('Cannot certify subcriticality at this precision')
        survival=lambda t:(1+t/scale)**(1-p)
        w=[survival(j*dt)-survival((j+1)*dt) for j in range(K+1)]
        seed=[]
        history=[(-arb(e['time']),base*(alpha*arb(10).log()*(arb(e['magnitude'])-arb(c['m0']))).exp()) for e in c['history']]
        for j in range(K):
            seed.append(arb(c['mu'])*dt+sum((k*(survival(age+j*dt)-survival(age+(j+1)*dt)) for age,k in history),arb(0)))
        early=[];late=[];den=1-n*w[0]
        if not den>0:raise ArithmeticError('Cannot certify early-grid contraction')
        for j in range(K+1):
            early.append((1+n*sum((w[l]*early[j-l] for l in range(1,j+1)),arb(0)))/den)
            late.append(1+n*sum((w[l-1]*late[j-l] for l in range(1,j+1)),arb(0)))
        low=sum((seed[j]*late[K-j-1] for j in range(K)),arb(0))
        high=sum((seed[j]*early[K-j] for j in range(K)),arb(0))
        # Independent nonnegative-generation bounds also enclose the total mean.
        B=sum(seed,arb(0));low_endpoint=max(low.lower(),B.lower(),arb(0))
        high_endpoint=min(high.upper(),(B/(1-n)).upper())
        q=(-b*arb(10).log()*(arb(c['threshold'])-arb(c['m0']))).exp()
        mean_lo=(q*low_endpoint).lower();mean_hi=(q*high_endpoint).upper()
        if not 0<=mean_lo<=mean_hi:raise ArithmeticError('Inconsistent mean enclosure')
        prob_lo=max(arb(0),(1-(-mean_lo).exp()).lower())
        prob_hi=min(arb(1),(1-(-mean_hi).exp()).upper())
        return dict(parameters=c,mean_lower=encode(mean_lo),mean_upper=encode(mean_hi),
                    poisson_lower=encode(prob_lo),poisson_upper=encode(prob_hi),
                    branching_mean=encode(n),target_mark_probability=encode(q),immigrant_mass=encode(B),
                    source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                   for name in ('controlled_mean_probability.py','validated_probability.py','validate_ball_zero.py')})
    finally:ctx.prec=old


def endpoint(record,side):
    item=record[side]
    return Fraction(int(item['mantissa']))*Fraction(2)**int(item['exponent'])


def solve(config, policy):
    c=validate(config)
    for name in ('max_time_bins','max_evaluations'):positive_integer(policy[name],name)
    tol=decimal_value(policy['absolute_probability_tolerance'],'tolerance')
    wall=decimal_value(policy['wall_seconds'],'wall_seconds')
    if not 0<tol<1 or wall<=0 or c['time_bins']>policy['max_time_bins']:
        raise ValueError('Invalid adaptive mean policy')
    trace=[];bestlo=besthi=None;records=None;error=None;start=time.perf_counter()
    while c['time_bins']<=policy['max_time_bins'] and len(trace)<policy['max_evaluations'] and time.perf_counter()-start<float(wall):
        tick=time.perf_counter()
        try:
            result=enclose(c)
            lo=endpoint(result['poisson_lower'],'lower');hi=endpoint(result['poisson_upper'],'upper')
            if bestlo is None:
                bestlo,besthi=lo,hi;records=[result['poisson_lower'],result['poisson_upper']]
            else:
                if lo>bestlo:bestlo=lo;records[0]=result['poisson_lower']
                if hi<besthi:besthi=hi;records[1]=result['poisson_upper']
            if bestlo>besthi:raise ArithmeticError('Disjoint mean-based probability intervals')
            trace.append(dict(status='enclosed',seconds=time.perf_counter()-tick,result=result))
            if besthi-bestlo<=2*tol:break
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            trace.append(dict(status='numerical_failure',error=error,time_bins=c['time_bins'],seconds=time.perf_counter()-tick));break
        c['time_bins']*=2
    achieved=error is None and bestlo is not None and besthi-bestlo<=2*tol
    answer=dict(status='numerical_failure' if error else ('tolerance_achieved' if achieved else 'resource_limit'),
                error=error,trace=trace,evaluations=len(trace),wall_seconds=time.perf_counter()-start,policy=deepcopy(policy))
    answer['wall_budget_exceeded']=answer['wall_seconds']>float(wall)
    if records is not None:answer.update(poisson_lower=records[0],poisson_upper=records[1])
    return answer
