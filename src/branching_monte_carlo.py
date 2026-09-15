"""Event-driven Monte Carlo indicators with conservative unfinished accounting.

Uses float64 random sampling; no deterministic arithmetic certificate is claimed.
"""
import math,time,hashlib
from pathlib import Path
import numpy as np
from controlled_mean_probability import validate


def prepare(config):
    c=validate(config);H=float(c['horizon']);scale=float(c['c']);p=float(c['p'])
    beta=float(c['b'])*math.log(10);alpha=float(c['alpha'])*math.log(10);base=float(c['productivity']);m0=float(c['m0'])
    age=np.array([-float(x['time']) for x in c['history']])
    k=np.array([base*math.exp(alpha*(float(x['magnitude'])-m0)) for x in c['history']])
    s0=np.exp((1-p)*np.log1p(age/scale))
    fraction=-np.expm1((1-p)*np.log1p(H/(scale+age)))
    mass=np.concatenate(([float(c['mu'])*H],k*s0*fraction))
    return dict(H=H,c=scale,p=p,beta=beta,alpha=alpha,base=base,m0=m0,threshold=float(c['threshold']),
                age=age,fraction=fraction,mass=mass,cdf=np.cumsum(mass),B=float(mass.sum()))


def batch(model,size,rng,event_limit,generation_limit):
    success=np.zeros(size,dtype=bool);unknown=np.zeros(size,dtype=bool)
    number=rng.poisson(model['B'],size);total=int(number.sum())
    unknown[number>0]=True
    if total>event_limit:return success,unknown,total,'event_limit'
    ids=np.repeat(np.arange(size),number)
    component=np.searchsorted(model['cdf'],rng.random(total)*model['B'],side='right')
    u=rng.random(total);times=u*model['H']
    historic=component>0
    j=component[historic]-1
    times[historic]=(model['c']+model['age'][j])*np.expm1(np.log1p(-u[historic]*model['fraction'][j])/(1-model['p']))
    generated=total
    for generation in range(generation_limit):
        if not len(ids):return success,np.zeros(size,dtype=bool),generated,'completed'
        marks=model['m0']+rng.exponential(1/model['beta'],len(ids))
        success[ids[marks>=model['threshold']]]=True
        keep=~success[ids];ids=ids[keep];times=times[keep];marks=marks[keep]
        if not len(ids):return success,np.zeros(size,dtype=bool),generated,'completed'
        remaining=np.maximum(0,model['H']-times)
        fraction=-np.expm1((1-model['p'])*np.log1p(remaining/model['c']))
        mean=model['base']*np.exp(model['alpha']*(marks-model['m0']))*fraction
        number=rng.poisson(mean);children=int(number.sum());generated+=children
        unknown=np.zeros(size,dtype=bool);unknown[ids[number>0]]=True
        if generated>event_limit:return success,unknown,generated,'event_limit'
        childids=np.repeat(ids,number);parenttimes=np.repeat(times,number);fraction=np.repeat(fraction,number)
        lag=model['c']*np.expm1(np.log1p(-rng.random(children)*fraction)/(1-model['p']))
        ids=childids;times=parenttimes+lag
    return success,unknown,generated,'generation_limit'


def solve(config,plan,case_index,total_cases=306):
    N=plan['samples_per_case'];batch_size=plan['batch_size'];alpha=plan['familywise_alpha']
    if type(N) is not int or N<=0 or type(batch_size) is not int or batch_size<=0 or not 0<alpha<1:
        raise ValueError('Invalid Monte Carlo budget')
    for key in ('maximum_events_per_batch','maximum_generations'):
        if type(plan[key]) is not int or plan[key]<=0:raise ValueError('Invalid simulation limit')
    if type(case_index) is not int or not 0<=case_index<total_cases:raise ValueError('Invalid case index')
    start=time.perf_counter();model=prepare(config);known_success=0;known_failure=0;attempted=0;trace=[]
    while attempted<N and time.perf_counter()-start<plan['wall_seconds_per_case']:
        size=min(batch_size,N-attempted)
        rng=np.random.default_rng(np.random.SeedSequence([plan['seed'],case_index,len(trace)]))
        try:
            success,unknown,events,status=batch(model,size,rng,plan['maximum_events_per_batch'],plan['maximum_generations'])
        except Exception as error:
            attempted+=size
            trace.append(dict(status='numerical_failure',error=f'{type(error).__name__}: {error}',size=size));break
        known_success+=int(success.sum());known_failure+=int((~success & ~unknown).sum())
        attempted+=size;trace.append(dict(status=status,size=size,unknown=int(unknown.sum()),generated_events=events))
    unknown=N-known_success-known_failure
    allowance=math.sqrt(math.log(2*total_cases/alpha)/(2*N))
    failed=any(x['status']=='numerical_failure' for x in trace)
    return dict(status='numerical_failure' if failed else ('completed' if unknown==0 else 'incomplete'),planned_replicates=N,attempted_replicates=attempted,
                known_success=known_success,known_failure=known_failure,unknown=unknown,
                empirical_lower=known_success/N,empirical_upper=(known_success+unknown)/N,
                probability_lower=max(0,known_success/N-allowance),probability_upper=min(1,(known_success+unknown)/N+allowance),
                hoeffding_allowance=allowance,total_cases=total_cases,familywise_alpha=alpha,
                trace=trace,wall_seconds=time.perf_counter()-start,plan=plan,case_index=case_index,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
