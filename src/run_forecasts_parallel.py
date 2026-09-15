"""Chronological forecast execution with bounded worker inputs and resumption."""
import argparse,csv,hashlib,json,math,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from forecast_context import load_catalog,timestamp_ns,past_events,model_history,count_window,history_hash,DAY_NS
from adaptive_threshold_probability import solve as branching
from controlled_mean_probability import solve as mean
from conventional_quadrature import solve as quadrature
from branching_monte_carlo import solve as simulation
ROOT=Path(__file__).resolve().parents[1]


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def compact(result,reference):
    result.pop('input_config',None)
    result.pop('parameters',None)
    for row in result.get('trace',[]):
        if 'result' in row:row['result'].pop('parameters',None)
    result['input_reference']=reference
    return result


def worker(job):
    c,meta,policy,numerical,baseline=job
    begin=time.perf_counter();methods={}
    reference=dict(history_sha256=history_hash(c['history']),history_events=len(c['history']),
                   parameters={k:v for k,v in c.items() if k!='history'},origin=meta['origin_utc'])
    for name,fn in [('branching',branching),('matched_mean_poisson',mean)]:
        try:methods[name]=compact(fn(c,policy),reference)
        except Exception as error:methods[name]=dict(status='numerical_failure',error=f'{type(error).__name__}: {error}',input_reference=reference)
    H=float(c['horizon']);tau=baseline['selected_prior_exposure_days']
    recent_rate=(meta['past_year_targets']+tau*baseline['prior_rate_per_day'])/(365+tau)
    methods['historical_rate']=dict(status='completed',probability=-math.expm1(-H*baseline['historical_rate_per_day']))
    methods['recent_rate']=dict(status='completed',probability=-math.expm1(-H*recent_rate),past_year_targets=meta['past_year_targets'])
    if int(meta['numerical_check']) and meta.get('run_numerical_comparators',True):
        for name,fn in [('quadrature',lambda:quadrature(c,numerical['quadrature'])),
                        ('monte_carlo',lambda:simulation(c,numerical['monte_carlo'],meta['numerical_case_index']))]:
            try:methods[name]=fn()
            except Exception as error:methods[name]=dict(status='numerical_failure',error=f'{type(error).__name__}: {error}')
    return dict(case=meta,methods=methods,worker_seconds=time.perf_counter()-begin,labels_generated=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--execution-config',default='experiments/forecast_execution_v1.json');args=parser.parse_args()
    execution_path=(ROOT/args.execution_config).resolve()
    if not execution_path.is_relative_to(ROOT):raise ValueError('Execution configuration outside project')
    execution=json.loads(execution_path.read_text())
    protocol=json.loads((ROOT/'experiments/empirical_protocol_v1.json').read_text())
    fitpath=ROOT/execution['primary_fit'];fit=json.loads(fitpath.read_text())
    if fit['status']!='completed' or len(fit['fits'])!=12:raise RuntimeError('Pre-test fit must complete all12 starts')
    if fit['likelihood_interval'][1]!='2007-01-01':raise ValueError('Unexpected fit endpoint')
    baseline_path=ROOT/execution['baseline_fit'];baseline=json.loads(baseline_path.read_text())
    rates={x['threshold']:x for x in baseline['records']}
    numerical_path=ROOT/'experiments/numerical_comparators_v1.json';numerical=json.loads(numerical_path.read_text())
    calendar_path=ROOT/execution['origin_manifest'];calendar_summary=json.loads((ROOT/'experiments/forecast_origins_v1_summary.json').read_text())
    if hashlib.sha256(calendar_path.read_bytes()).hexdigest()!=calendar_summary['manifest_sha256']:raise ValueError('Frozen calendar changed')
    rows=list(csv.DictReader(calendar_path.open()))
    if execution.get('case_selection','all')=='primary':rows=[row for row in rows if row['primary']=='1']
    elif execution.get('case_selection','all')!='all':raise ValueError('Unknown case selection')
    expected_numerical=sum(int(row['numerical_check']) for row in rows)
    events=load_catalog(ROOT/protocol['data']['path'],protocol['data']['sha256'])
    sources=['run_forecasts_parallel.py','run_forecasts.py','forecast_context.py','adaptive_threshold_probability.py','threshold_probability.py','controlled_mean_probability.py','validated_probability.py','validate_ball_zero.py','conventional_quadrature.py','branching_monte_carlo.py']
    evidence={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [execution_path,fitpath,baseline_path,numerical_path,calendar_path]+[ROOT/'src'/name for name in sources]}
    evidence['catalog_sha256']=protocol['data']['sha256']
    signature=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()
    out=(ROOT/execution.get('output_directory','results/test_primary_v1')).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output outside project results')
    if out.exists():
        if not args.resume:raise FileExistsError('Use explicit --resume for existing forecast run')
        if json.loads((out/'manifest.json').read_text())['signature']!=signature:raise ValueError('Cannot resume changed code or inputs')
    else:out.mkdir()
    state=dict(status='running',signature=signature,evidence=evidence,total_cases=len(rows),labels_generated=False)
    atomic(out/'manifest.json',state)
    counter=0;done=0
    from concurrent.futures import wait, FIRST_COMPLETED
    def jobs():
        nonlocal counter,done
        for row in rows:
            origin=timestamp_ns(row['origin_utc']);past=past_events(events,origin)
            history=model_history(past,origin,fit['parameters']['m0'])
            meta=dict(row);meta['run_numerical_comparators']=execution.get('run_numerical_comparators',True)
            meta['past_year_targets']=count_window(past,origin-365*DAY_NS,origin,row['magnitude_threshold'])
            if int(row['numerical_check']):meta['numerical_case_index']=counter;counter+=1
            path=out/f"{int(row['origin_index']):04d}_h{row['horizon_days']}_m{row['magnitude_threshold']}.json"
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['signature']!=signature or any(saved['case'].get(k)!=v for k,v in row.items()):raise ValueError('Case signature or identity mismatch')
                done+=1;continue
            c=dict(fit['parameters'],horizon=row['horizon_days'],threshold=row['magnitude_threshold'],history=history,
                   time_bins=execution['initial_time_bins'],precision_bits=execution['precision_bits'],
                   radius_tolerance=execution['radius_tolerance'],max_iterations=execution['max_iterations'])
            yield (c,meta,execution['policy'],numerical,rates[row['magnitude_threshold']]),path
    with ProcessPoolExecutor(max_workers=execution['workers']) as pool:
        iterator=iter(jobs());pending={};exhausted=False
        while pending or not exhausted:
            while not exhausted and len(pending)<execution['workers']:
                try:job,path=next(iterator)
                except StopIteration:exhausted=True;break
                pending[pool.submit(worker,job)]=path
            if not pending:break
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                path=pending.pop(future);value=future.result();value['signature']=signature;atomic(path,value);done+=1
                state['completed_cases']=done;atomic(out/'manifest.json',state)
                print(json.dumps(dict(completed_cases=done,total_cases=len(rows))),flush=True)
    if done!=len(rows) or counter!=expected_numerical:raise RuntimeError('Incomplete forecast calendar')
    state['completed_cases']=done;state['status']='completed';atomic(out/'manifest.json',state)

if __name__=='__main__':main()
