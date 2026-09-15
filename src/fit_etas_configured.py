"""Run a prespecified pre-test fit; never access test-period marks or scores."""
import csv,hashlib,itertools,json,math,platform,time,argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy,numba
from scipy.optimize import minimize
from etas_likelihood import nll_gradient
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    planpath=(ROOT/args.config).resolve()
    if not planpath.is_relative_to(ROOT):raise ValueError('Configuration outside project')
    plan=json.loads(planpath.read_text())
    protocol=json.loads((ROOT/'experiments/empirical_protocol_v1.json').read_text())
    source=ROOT/protocol['data']['path']
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=protocol['data']['sha256']:raise ValueError('Source hash mismatch')
    epoch=datetime.fromisoformat(plan['auxiliary_start'])
    startdate,enddate=[datetime.fromisoformat(x) for x in plan['likelihood_interval']]
    test_start=datetime.fromisoformat(protocol['partitions']['test_start'].replace('Z','+00:00')).replace(tzinfo=None)
    if not epoch<=startdate<enddate<=test_start:raise ValueError('Invalid or test-overlapping fitting interval')
    start=(startdate-epoch).total_seconds()/86400;end=(enddate-epoch).total_seconds()/86400
    m0=float(plan['cutoff']);times=[];magnitudes=[];previous=None
    with source.open() as stream:
        for row in csv.DictReader(stream):
            date=datetime.fromisoformat(row['time'])
            if date>=enddate:break
            if previous is not None and date<previous:raise ValueError('Unsorted source')
            previous=date
            if date<epoch:continue
            magnitude=float(row['magnitude'])
            if magnitude>=m0:
                times.append((date-epoch).total_seconds()/86400);magnitudes.append(magnitude)
    t=np.asarray(times);m=np.asarray(magnitudes);selection=(t>=start)&(t<end)
    count=int(selection.sum())
    if count==0 or np.mean(m[selection]-m0)<=0:raise ValueError('Insufficient fitting marks')
    b=1/(math.log(10)*float(np.mean(m[selection]-m0)));beta=b*math.log(10)
    rate=count/(end-start);bounds=plan['bounds']
    optimizer_bounds=[tuple(math.log(x) for x in bounds[key]) for key in ('mu','n','c')]
    optimizer_bounds.append(tuple(math.log(x-1) for x in bounds['p']))
    optimizer_bounds.append(tuple(bounds['alpha_over_b']))
    out=(ROOT/plan['output_directory']).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output must be in project results')
    out.mkdir(exist_ok=False)
    record=dict(status='running',scope=plan['scope'],likelihood_interval=plan['likelihood_interval'],cutoff=plan['cutoff'],branching_ratio_cap=plan['bounds']['n'][1],
                likelihood_events=count,auxiliary_events=int((t<start).sum()),b=format(b,'.17g'),
                source_sha256=digest,configuration_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),
                code_sha256={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in ('fit_etas_configured.py','etas_likelihood.py')},
                versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__),fits=[])
    def save():
        tmp=out/'fit.json.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out/'fit.json')
    save();print(json.dumps({k:record[k] for k in ('likelihood_events','auxiliary_events','b')}),flush=True)
    starts=plan['starts'];clock=time.perf_counter()
    for index,(n,c,p,r) in enumerate(itertools.product(starts['n'],starts['c'],starts['p'],starts['alpha_over_b'])):
        theta=np.array([math.log(rate*(1-n)),math.log(n),math.log(c),math.log(p-1),r])
        tick=time.perf_counter()
        result=minimize(nll_gradient,theta,args=(t,m-m0,beta,start,end),jac=True,
                        method=plan['optimizer']['method'],bounds=optimizer_bounds,
                        options={k:v for k,v in plan['optimizer'].items() if k!='method'})
        finite=bool(np.isfinite(result.fun) and np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.jac)))
        fit=dict(index=index,initial_theta=theta.tolist(),theta=result.x.tolist(),
                 success=bool(result.success and finite),message=str(result.message),
                 objective=float(result.fun),gradient=result.jac.tolist(),iterations=int(result.nit),
                 evaluations=int(result.nfev),seconds=time.perf_counter()-tick,
                 near_bound=[bool(min(abs(x-lo),abs(x-hi))<1e-5) for x,(lo,hi) in zip(result.x,optimizer_bounds)])
        record['fits'].append(fit);record['elapsed_seconds']=time.perf_counter()-clock;save()
        print(json.dumps({k:fit[k] for k in ('index','success','objective','iterations','seconds')}),flush=True)
    accepted=[x for x in record['fits'] if x['success']]
    if accepted:
        best=min(accepted,key=lambda x:x['objective']);a=best['theta'];n=math.exp(a[1]);r=a[4]
        parameters=dict(mu=math.exp(a[0]),n=n,c=math.exp(a[2]),p=1+math.exp(a[3]),b=b,alpha=r*b,productivity=n*(1-r),m0=m0)
        record.update(status='completed',selected_start=best['index'],parameters={k:format(v,'.17g') for k,v in parameters.items()})
    else:record['status']='no_accepted_fit'
    save();print(json.dumps(dict(status=record['status'],selected_start=record.get('selected_start'))),flush=True)

if __name__=='__main__':main()
