"""Continuous bounded-GR marks and fixed-history zero-count enclosures."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from flint import arb,ctx
from validate_ball_zero import encode
ROOT=Path(__file__).resolve().parents[1]

def bound(c,bins,early):
    K=c['time_bins'];dt=arb(c['horizon'])/K;p=arb(c['p']);cc=arb(c['c'])
    low=arb(c['m0']);span=arb(c['mmax'])-low;threshold=arb(c['threshold'])
    beta=arb(c['b'])*arb(10).log();alpha=arb(c['alpha'])*arb(10).log();base=arb(c['productivity'])
    edges=[low+span*i/bins for i in range(bins+1)]
    den=1-(-beta*span).exp()
    probs=[((-beta*(edges[i]-low)).exp()-(-beta*(edges[i+1]-low)).exp())/den for i in range(bins)]
    kappas=[base*(alpha*(edges[i+int(early)]-low)).exp() for i in range(bins)]
    # Threshold coincides with a bin boundary; selection is exact within bins.
    threshold_index=c['threshold_bin_fraction_numerator']*bins//c['threshold_bin_fraction_denominator']
    assert threshold_index*c['threshold_bin_fraction_denominator']==c['threshold_bin_fraction_numerator']*bins
    assert edges[threshold_index].overlaps(threshold)
    nbar=sum((a*b for a,b in zip(probs,kappas)),arb(0));assert nbar<1
    def survival(t):return (1+t/cc)**(1-p)
    w=[survival(i*dt)-survival((i+1)*dt) for i in range(K+1)]
    seed=[]
    for j in range(K):
        value=arb(c['mu'])*dt
        for event in c['history']:
            age=-arb(event['time']);k=base*(alpha*(arb(event['magnitude'])-low)).exp()
            value+=k*(survival(age+j*dt)-survival(age+(j+1)*dt))
        seed.append(value)
    f=[]
    for n in range(K+1):
        past=sum(((w[l] if early else w[l-1])*(f[n-l]-1) for l in range(1,n+1)),arb(0))
        value=arb('0.5','0.5')
        for _ in range(c['max_iterations']):
            argument=past+(w[0]*(value-1) if early else 0)
            value=sum((probs[m]*(kappas[m]*argument).exp() for m in range(threshold_index)),arb(0))
            if value.rad()<arb(c['radius_tolerance']):break
        else:raise RuntimeError('Interval width failed to converge')
        f.append(value)
    g=sum((seed[j]*(f[K-j if early else K-j-1]-1) for j in range(K)),arb(0)).exp()
    return g,nbar

def simulate(c):
    rng=np.random.default_rng(c['seed']);n=c['simulations'];H=float(c['horizon']);cc=float(c['c']);p=float(c['p'])
    m0=float(c['m0']);span=float(c['mmax'])-m0;beta=float(c['b'])*math.log(10);alpha=float(c['alpha'])*math.log(10)
    ids=np.arange(n);numbers=rng.poisson(float(c['mu'])*H,n)
    event_ids=[np.repeat(ids,numbers)];times=[rng.uniform(0,H,int(numbers.sum()))]
    for event in c['history']:
        age=-float(event['time']);k=float(c['productivity'])*math.exp(alpha*(float(event['magnitude'])-m0))
        slo=(1+age/cc)**(1-p);shi=(1+(age+H)/cc)**(1-p)
        numbers=rng.poisson(k*(slo-shi),n)
        event_ids.append(np.repeat(ids,numbers))
        s=slo-rng.random(int(numbers.sum()))*(slo-shi)
        times.append(cc*(s**(1/(1-p))-1)-age)
    ids=np.concatenate(event_ids);t=np.concatenate(times);counts=np.zeros(n,dtype=int)
    while len(ids):
        marks=m0-np.log1p(-rng.random(len(ids))*(-math.expm1(-beta*span)))/beta
        counts+=np.bincount(ids[marks>=float(c['threshold'])],minlength=n)
        mass=-np.expm1((1-p)*np.log1p((H-t)/cc))
        children=rng.poisson(float(c['productivity'])*np.exp(alpha*(marks-m0))*mass)
        childids=np.repeat(ids,children);parenttimes=np.repeat(t,children);childmass=np.repeat(mass,children)
        lag=cc*np.expm1(np.log1p(-rng.random(len(childids))*childmass)/(1-p))
        ids=childids;t=parenttimes+lag
    return float(np.mean(counts==0))

def main():
    path=ROOT/'experiments/synthetic_continuous_marks.json';c=json.loads(path.read_text());ctx.prec=c['precision_bits']
    empirical=simulate(c);tolerance=math.sqrt(math.log(2/c['simulation_alpha'])/(2*c['simulations']))
    records=[]
    for bins in c['mark_bins']:
        early,nu=bound(c,bins,True);late,nl=bound(c,bins,False)
        assert early.lower()<=late.upper()
        assert float(early.lower())-tolerance<=empirical<=float(late.upper())+tolerance
        records.append(dict(mark_bins=bins,early=encode(early),late=encode(late),
            width_approx=float(late.upper()-early.lower()),upper_branching_mean=str(nu),lower_branching_mean=str(nl)))
    out=ROOT/'results/synthetic_continuous_marks';out.mkdir(parents=True,exist_ok=True)
    result=dict(status='passed',scope='Synthetic bounded continuous GR magnitudes,fixed synthetic prehistory,zero-count target only. Not empirical earthquake forecasting.',
        simulated_zero_probability=empirical,monte_carlo_hoeffding_allowance=tolerance,records=records,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
