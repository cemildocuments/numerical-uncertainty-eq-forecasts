"""Interval-arithmetic check of zero-count probabilities for rounded trees."""
from pathlib import Path
import json,hashlib
from flint import arb,ctx
import flint
ROOT=Path(__file__).resolve().parents[1]

def zero(c,selected,early):
    grid=c['grid'];dt=arb(c['horizon'])/grid
    mu=arb(c['seed_rate']);shape=arb(c['kernel_p']);scale=arb(c['kernel_c'])
    probs=[arb(x) for x in c['mark_probabilities']];kappa=[arb(x) for x in c['productivities']]
    weights=[(1+i*dt/scale)**(1-shape)-(1+(i+1)*dt/scale)**(1-shape) for i in range(grid+1)]
    f=[];maximum_iterations=0
    for n in range(grid+1):
        past=sum(((weights[l] if early else weights[l-1])*(f[n-l]-1) for l in range(1,n+1)),arb(0))
        value=arb('0.5','0.5')
        for it in range(c['max_iterations']):
            argument=past+(weights[0]*(value-1) if early else 0)
            value=sum((probs[m]*(kappa[m]*argument).exp() for m in range(len(probs)) if not selected[m]),arb(0))
            if value.rad()<arb(c['radius_tolerance']):break
        else:raise RuntimeError('Interval iteration width did not converge')
        f.append(value);maximum_iterations=max(maximum_iterations,it+1)
    g=(mu*dt*sum((x-1 for x in (f[1:] if early else f[:-1])),arb(0))).exp()
    return g,maximum_iterations

def encode(x):
    # Exact dyadic endpoints, not decimal-rounded bounds.
    def endpoint(y):
        m,e=y.man_exp();return dict(mantissa=str(m),exponent=int(e))
    return dict(display=x.str(35),lower=endpoint(x.lower()),upper=endpoint(x.upper()),radius_display=x.rad().str(8))

def main():
    p=ROOT/'experiments/synthetic_ball_zero.json';c=json.loads(p.read_text());ctx.prec=c['precision_bits']
    results=[]
    for target,selection in [('all',[1,1]),('mark_1',[0,1])]:
        late,il=zero(c,selection,False);early,ie=zero(c,selection,True)
        assert early.lower()<=late.upper()
        if target=='all':
            exact=(-arb(c['seed_rate'])*arb(c['horizon'])).exp()
            assert late.overlaps(exact) and early.overlaps(exact)
        enclosure=early.union(late)
        results.append(dict(target=target,early=encode(early),late=encode(late),
            continuous_zero_enclosure=encode(enclosure),maximum_iterations=max(il,ie)))
    out=ROOT/'results/synthetic_ball_zero';out.mkdir(parents=True,exist_ok=True)
    result=dict(status='passed',scope='Two-mark homogeneous-seed model;zero-count only. Ball arithmetic plus previously derived time coupling. Not a certification of the full count solver.',
        precision_bits=ctx.prec,python_flint=flint.__version__,results=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),config_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
