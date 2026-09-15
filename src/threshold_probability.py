"""Zero-exceedance bounds by absorbing target marks; no numerical cutoff."""
from pathlib import Path
import hashlib,json
from flint import arb,ctx
from controlled_mean_probability import validate
from validated_probability import positive_integer,decimal_value
from validate_ball_zero import encode


def enclose(config,mark_bins,*,_seed_cache=None):
    c=validate(config);positive_integer(mark_bins,'mark_bins')
    positive_integer(c['max_iterations'],'max_iterations')
    radius=decimal_value(c['radius_tolerance'],'radius_tolerance')
    if not 0<radius<1:raise ValueError('radius_tolerance must be in (0,1)')
    old=ctx.prec
    try:
        ctx.prec=c['precision_bits'];K=c['time_bins'];dt=arb(c['horizon'])/K
        scale=arb(c['c']);p=arb(c['p']);base=arb(c['productivity'])
        beta=arb(c['b'])*arb(10).log();alpha=arb(c['alpha'])*arb(10).log()
        span=arb(c['threshold'])-arb(c['m0'])
        survival=lambda t:(1+t/scale)**(1-p)
        # A cache is local to an adaptive call. The content fingerprint prevents
        # reuse across different histories, model inputs, or arithmetic precision.
        fingerprint=hashlib.sha256(json.dumps({k:v for k,v in c.items() if k!='time_bins'},sort_keys=True).encode()).hexdigest()
        cache_key=(fingerprint,K)
        if _seed_cache is not None and cache_key in _seed_cache:
            seed=_seed_cache[cache_key]
        else:
            history=[(-arb(e['time']),base*(alpha*(arb(e['magnitude'])-arb(c['m0']))).exp()) for e in c['history']]
            seed=[arb(c['mu'])*dt+sum((k*(survival(age+j*dt)-survival(age+(j+1)*dt)) for age,k in history),arb(0)) for j in range(K)]
            if _seed_cache is not None:_seed_cache[cache_key]=seed
        if decimal_value(c['threshold'],'threshold')==decimal_value(c['m0'],'m0'):
            value=(-sum(seed,arb(0))).exp();lo=value.lower();hi=value.upper()
        else:
            w=[survival(j*dt)-survival((j+1)*dt) for j in range(K+1)]
            edges=[span*j/mark_bins for j in range(mark_bins+1)]
            masses=[(-beta*edges[j]).exp()-(-beta*edges[j+1]).exp() for j in range(mark_bins)]
            upper_mean=sum((masses[j]*base*(alpha*edges[j+1]).exp() for j in range(mark_bins)),arb(0))
            if not upper_mean<1:raise ValueError('upper non-target branching mean is not certified below one')
            answers=[]
            for early in (True,False):
                productivity=[base*(alpha*edges[j+int(early)]).exp() for j in range(mark_bins)]
                f=[]
                for n in range(K+1):
                    past=sum(((w[l] if early else w[l-1])*(f[n-l]-1) for l in range(1,n+1)),arb(0))
                    value=arb('0.5','0.5')
                    for _ in range(c['max_iterations']):
                        exponent=past+(w[0]*(value-1) if early else 0)
                        value=sum((mass*(k*exponent).exp() for mass,k in zip(masses,productivity)),arb(0))
                        if value.rad()<arb(c['radius_tolerance']):break
                    else:raise ArithmeticError('Threshold interval iteration did not converge')
                    f.append(value)
                answers.append(sum((seed[j]*(f[K-j if early else K-j-1]-1) for j in range(K)),arb(0)).exp())
            lo=answers[0].lower();hi=answers[1].upper()
        lo=max(arb(0),lo);hi=min(arb(1),hi)
        if not lo<=hi:raise ArithmeticError('Empty threshold enclosure')
        return dict(parameters=c,mark_bins=mark_bins,lower_endpoint=encode(lo),upper_endpoint=encode(hi),
                    method='Absorbing target marks; unconditional subthreshold GR mass',
                    source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                   for name in ('threshold_probability.py','controlled_mean_probability.py','validated_probability.py','validate_ball_zero.py')})
    finally:ctx.prec=old
