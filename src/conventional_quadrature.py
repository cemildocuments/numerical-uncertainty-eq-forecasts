"""Uncertified float64 Volterra comparator, separate from interval solvers."""
import math,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
from controlled_mean_probability import validate


def calculate(config,time_bins,mark_nodes):
    c=validate(dict(config,time_bins=time_bins))
    H=float(c['horizon']);scale=float(c['c']);p=float(c['p']);mu=float(c['mu'])
    m0=float(c['m0']);h=float(c['threshold']);beta=float(c['b'])*math.log(10)
    alpha=float(c['alpha'])*math.log(10);base=float(c['productivity']);dt=H/time_bins
    times=np.linspace(0,H,time_bins+1)
    g=(p-1)/scale*np.exp(-p*np.log1p(times/scale))
    immigrants=np.full(time_bins+1,mu)
    ages=np.array([-float(e['time']) for e in c['history']])
    productivity=np.array([base*math.exp(alpha*(float(e['magnitude'])-m0)) for e in c['history']])
    for j,t in enumerate(times):
        immigrants[j]+=np.sum(productivity*(p-1)/scale*np.exp(-p*np.log1p((ages+t)/scale)))
    f=np.zeros(time_bins+1)
    if h>m0:
        nodes,weights=leggauss(mark_nodes);x=(nodes+1)*(h-m0)/2
        masses=weights*(h-m0)/2*beta*np.exp(-beta*x)
        kappa=base*np.exp(alpha*x);f[0]=-math.expm1(-beta*(h-m0))
        for j in range(1,time_bins+1):
            past=dt*(np.dot(g[1:j],f[j-1:0:-1]-1)+.5*g[j]*(f[0]-1))
            def equation(value):
                return np.dot(masses,np.exp(kappa*(past+.5*dt*g[0]*(value-1))))-value
            f[j]=brentq(equation,0.,1.,xtol=1e-13,rtol=1e-13)
    integrand=immigrants*(f[::-1]-1)
    log_zero=dt*(np.sum(integrand[1:-1])+.5*(integrand[0]+integrand[-1]))
    probability=-math.expm1(float(log_zero))
    if not 0<=probability<=1:raise ArithmeticError('Quadrature probability outside [0,1]')
    return probability


def solve(config,plan):
    trace=[];previous=None;status='resource_limit';start=time.perf_counter()
    for K,M in plan['grids']:
        if time.perf_counter()-start>=plan['wall_seconds']:break
        tick=time.perf_counter()
        try:
            value=calculate(config,K,M)
            difference=None if previous is None else abs(value-previous)
            trace.append(dict(time_bins=K,mark_nodes=M,probability=value,successive_difference=difference,seconds=time.perf_counter()-tick))
            if difference is not None and difference<=float(plan['successive_probability_difference']):
                status='heuristic_convergence';break
            previous=value
        except Exception as e:
            status='numerical_failure';trace.append(dict(error=f'{type(e).__name__}: {e}',time_bins=K,mark_nodes=M));break
    return dict(status=status,trace=trace,wall_seconds=time.perf_counter()-start,certified=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
