"""Verify a minimal submission archive without rerunning fitted models."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,argparse,csv,bisect
from decimal import Decimal
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def dec(x):return F(int(x['numerator']),int(x['denominator']))
def bounds(pair,y):
 a,b=map(dec,pair)
 if not 0<=a<=b<=1 or y not in [0,1]:raise ValueError('Invalid probability/outcome')
 return ((1-b)**2,((a+b)/2-y)**2,(1-a)**2) if y else (a*a,((a+b)/2-y)**2,b*b)
def contrast(a,b,y):
 al,am,au=bounds(a,y);bl,bm,bu=bounds(b,y)
 return al-bu,am-bm,au-bl

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',action='store_true');parser.add_argument('--catalog',type=Path,help='Optional external pinned catalog for independent binary-label verification');args=parser.parse_args()
 manifest=json.loads((ROOT/'CONTENTS.json').read_text())
 for rel,digest in manifest['files'].items():
  p=ROOT/rel
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('File mismatch '+rel)
 administrative={'LICENSE','LICENSE.md','LICENSE.txt','.gitignore'}
 actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.relative_to(ROOT).parts and p.name!='CONTENTS.json' and not (p.parent==ROOT and p.name in administrative and p.name not in manifest['files'])}
 if actual!=set(manifest['files']):raise ValueError('Unexpected or missing archive files')
 groups={};positive_counts={};primary=[];identities=set();primary_labels={};primary_pairs={}
 calendar=list(csv.DictReader((ROOT/'experiments/forecast_origins_v1.csv').open()))
 expected_calendar={(r['origin_index'],r['horizon_days'],r['magnitude_threshold']):r['origin_utc'] for r in calendar}
 target_times={}
 def nanos(s):
  s=s.rstrip('Z');base,sep,fraction=s.partition('.')
  seconds=int((datetime.fromisoformat(base).replace(tzinfo=timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds())
  return seconds*10**9+int(fraction.ljust(9,'0') or '0')
 if args.catalog:
  protocol=json.loads((ROOT/'experiments/empirical_protocol_v1.json').read_text())
  if hashlib.sha256(args.catalog.read_bytes()).hexdigest()!=protocol['data']['sha256']:raise ValueError('External catalog digest mismatch')
  thresholds=sorted({k[2] for k in expected_calendar});target_times={h:[] for h in thresholds}
  for event in csv.DictReader(args.catalog.open()):
   t=nanos(event['time']);m=Decimal(event['magnitude'])
   for h in thresholds:
    if m>=Decimal(h):target_times[h].append(t)
  for values in target_times.values():values.sort()
 for line in (ROOT/'data/primary_probabilities.jsonl').open():
  x=json.loads(line);ident=(x['origin_index'],x['horizon_days'],x['magnitude_threshold'])
  if ident in identities:raise ValueError('Duplicate case')
  identities.add(ident)
  if expected_calendar.get(ident)!=x['origin_utc']:raise ValueError('Calendar identity mismatch')
  if args.catalog:
   t=nanos(x['origin_utc']);events=target_times[ident[2]];end=t+int(ident[1])*86400*10**9
   y=int(bisect.bisect_left(events,end)>bisect.bisect_left(events,t))
   if y!=x['outcome']:raise ValueError('Catalog outcome mismatch '+str(ident))
  if ident[1:]==('7','4.5'):primary_labels[int(ident[0])]=x['outcome']
  primary_pairs[ident]=x['probabilities']['branching']
  if set(x['probabilities'])!={'branching','matched_mean_poisson','historical_rate','recent_rate'}:raise ValueError('Missing method')
  if dec(x['probabilities']['branching'][0])>dec(x['probabilities']['matched_mean_poisson'][1]):raise ValueError('Structural probability ordering violated')
  for model,pair in x['probabilities'].items():
   k=ident[1:]+(model,);groups.setdefault(k,[]).append(bounds(pair,x['outcome']));positive_counts[k]=positive_counts.get(k,0)+x['outcome']
  if ident[1:]==('7','4.5'):primary.append((int(x['origin_index']),contrast(x['probabilities']['branching'],x['probabilities']['matched_mean_poisson'],x['outcome'])))
 if identities!=set(expected_calendar) or len(identities)!=6093 or len(primary)!=677:raise ValueError('Incomplete primary calendar')
 report=json.loads((ROOT/'results/primary_summary.json').read_text())
 for g in report['model_summaries']:
  values=groups[(g['horizon_days'],g['magnitude_threshold'],g['model'])]
  if len(values)!=g['cases'] or positive_counts[(g['horizon_days'],g['magnitude_threshold'],g['model'])]!=g['positive_windows']:raise ValueError('Group accounting')
  for j,key in enumerate(['brier_mean_lower','brier_midpoint_mean','brier_mean_upper']):
   if sum((v[j] for v in values),F(0))/len(values)!=dec(g[key]):raise ValueError('Brier mismatch')
 contrasts={'primary':[v for _,v in sorted(primary)]};expected={'primary':report['primary']}
 sensitivity=json.loads((ROOT/'results/sensitivity_summary.json').read_text())
 for g in sensitivity['sensitivity_results']:expected[g['variant']]=g['scores']
 by={}
 for line in (ROOT/'data/sensitivity_probabilities.jsonl').open():
  x=json.loads(line);k=(x['variant'],int(x['origin_index']))
  if k in by:raise ValueError('Duplicate sensitivity')
  if x['outcome']!=primary_labels.get(k[1]):raise ValueError('Sensitivity outcome mismatch')
  if dec(x['probabilities']['branching'][0])>dec(x['probabilities']['matched_mean_poisson'][1]):raise ValueError('Sensitivity structural ordering violated')
  by[k]=contrast(x['probabilities']['branching'],x['probabilities']['matched_mean_poisson'],x['outcome'])
 if set(by)!={(v,i) for v in ['n099','m025','m035','tighter'] for i in primary_labels}:raise ValueError('Sensitivity calendar mismatch')
 for variant in ['n099','m025','m035','tighter']:
  values=[v for (name,i),v in sorted(by.items()) if name==variant]
  if len(values)!=677:raise ValueError('Incomplete sensitivity')
  contrasts[variant]=values
 for name,values in contrasts.items():
  for j,x in enumerate(expected[name]['exact_mean_contrast']):
   if sum((v[j] for v in values),F(0))/len(values)!=dec(x):raise ValueError('Paired contrast mismatch '+name)
 comparison=json.loads((ROOT/'results/numerical_comparisons.json').read_text());seen=set()
 for item in comparison['cases']:
  c=item['case'];q=item['comparison'];ident=(c['origin_index'],c['horizon_days'],c['magnitude_threshold'])
  if ident in seen:raise ValueError('Duplicate numerical comparison')
  seen.add(ident);a,b=map(dec,primary_pairs[ident]);v=dec(q['quadrature_last_probability']);lo,hi=map(dec,q['monte_carlo_probability_bounds'])
  if b-a!=dec(q['branching_width']) or hi-lo!=dec(q['monte_carlo_width']):raise ValueError('Numerical width mismatch')
  if not 0<=lo<=hi<=1:raise ValueError('Invalid Monte Carlo interval')
  if (a<=v<=b)!=q['quadrature_inside_enclosure'] or max(a-v,v-b,F(0))!=dec(q['quadrature_distance_from_enclosure']):raise ValueError('Quadrature comparison mismatch')
  if (max(lo,a)<=min(hi,b))!=q['monte_carlo_interval_overlaps_enclosure']:raise ValueError('Monte Carlo overlap mismatch')
  if q['monte_carlo_known_success']+q['monte_carlo_known_failure']+q['unknown_replicates']!=q['planned_replicates']:raise ValueError('Monte Carlo accounting mismatch')
 expected_numerical={(r['origin_index'],r['horizon_days'],r['magnitude_threshold']) for r in calendar if r['numerical_check']=='1'}
 if seen!=expected_numerical or len(seen)!=306:raise ValueError('Incomplete numerical comparisons')
 if args.bootstrap:
  from score_intervals import block_bootstrap
  for name,values in contrasts.items():
   for saved in expected[name]['bootstrap']:
    computed=block_bootstrap(values,saved['block_length'],saved['replicates'],saved['seed'])
    if any(abs(a-b)>1e-13 for a,b in zip(computed['numerical_envelope'],saved['numerical_envelope'])):raise ValueError('Bootstrap mismatch '+name)
 if args.catalog:print('All 6,093 binary targets independently checked against the pinned catalog.')
 print('Verified archive hashes, complete calendars, every mean Brier score and all primary/sensitivity contrasts, and 306 numerical comparisons.'+(' Bootstrap envelopes also verified.' if args.bootstrap else ''))
if __name__=='__main__':main()
