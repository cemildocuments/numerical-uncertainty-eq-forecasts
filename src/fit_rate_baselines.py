"""Select simple-rate smoothing using validation only; final rates pre-test."""
import json,csv,math,hashlib
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    path=ROOT/'experiments/rate_baselines_v1.json';c=json.loads(path.read_text())
    protocol=json.loads((ROOT/'experiments/empirical_protocol_v1.json').read_text());source=ROOT/protocol['data']['path']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=protocol['data']['sha256']:raise ValueError('Source hash mismatch')
    start=datetime.fromisoformat(c['training_start']);valid=datetime.fromisoformat(c['validation_start']);end=datetime.fromisoformat(c['test_start'])
    events=[]
    with source.open() as stream:
        for row in csv.DictReader(stream):
            t=datetime.fromisoformat(row['time'])
            if t>=end:break
            events.append(((t-start).total_seconds()/86400,float(row['magnitude'])))
    data=np.array(events);v=(valid-start).total_seconds()/86400;T=(end-start).total_seconds()/86400
    H=c['validation_horizon_days'];W=c['trailing_window_days'];origins=np.arange(v,T-H+1e-10,c['validation_cadence_days'])
    records=[]
    for threshold in c['thresholds']:
        times=data[data[:,1]>=float(threshold),0]
        count=lambda lo,hi:np.searchsorted(times,hi,side='left')-np.searchsorted(times,lo,side='left')
        train=int(count(0,v));total=int(count(0,T));r0=(train+.5)/v
        past=count(origins-W,origins);y=(count(origins,origins+H)>0).astype(int)
        choices=[]
        for tau in c['prior_exposure_candidates_days']:
            rate=(past+tau*r0)/(W+tau);x=rate*H
            losses=np.where(y==1,-np.log(-np.expm1(-x)),x)
            choices.append(dict(prior_exposure_days=tau,validation_mean_log_loss=float(np.mean(losses))))
        selected=min(choices,key=lambda x:x['validation_mean_log_loss'])
        records.append(dict(threshold=threshold,training_target_count=train,pretest_target_count=total,
                            validation_origins=len(origins),validation_positive_windows=int(y.sum()),
                            choices=choices,selected_prior_exposure_days=selected['prior_exposure_days'],
                            historical_rate_per_day=total/T,prior_rate_per_day=(total+.5)/T,
                            pretest_exposure_days=T))
    out=ROOT/'results/rate_baselines_v1';out.mkdir(exist_ok=False)
    result=dict(status='completed',test_outcomes_read=False,records=records,
                source_sha256=protocol['data']['sha256'],configuration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'fit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',selected_exposures={x['threshold']:x['selected_prior_exposure_days'] for x in records},validation_origins=len(origins))),flush=True)

if __name__=='__main__':main()
