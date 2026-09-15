"""Report all completed sensitivity scores and effective tolerance attainment."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import json,csv,hashlib,math,argparse
ROOT=Path(__file__).resolve().parents[1]
def value(x):return float(Fraction(int(x['numerator']),int(x['denominator'])))
def endpoint(x,lower):return f'{(math.floor(x*1e6) if lower else math.ceil(x*1e6))/1e6:.6f}'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--submission-layout',action='store_true');args=parser.parse_args()
 source=ROOT/('results/sensitivity_summary.json' if args.submission_layout else 'results/scoring_all_sensitivities_final_v1/summary.json');s=json.loads(source.read_text())
 compact=[json.loads(line) for line in (ROOT/'data/sensitivity_probabilities.jsonl').open()] if args.submission_layout else None
 if s['status']!='completed':raise ValueError('Complete sensitivity scores required')
 labels={'n099':r'$n\le0.99$','m025':r'$m_0=2.5$','m035':r'$m_0=3.5$','tighter':r'$\varepsilon=0.0005$'};rows=[]
 for g in s['sensitivity_results']:
  variant=g['variant'];counts={'branching':Counter(),'matched_mean_poisson':Counter()}
  if compact is not None:
   for x in compact:
    if x['variant']==variant:
     for model in counts:counts[model][x['statuses'][model]]+=1
  else:
   for p in (ROOT/f'results/test_{variant}_v1').glob('0*.json'):
    repair=ROOT/f'results/sensitivity_repairs_v1/{variant}'/p.name
    if repair.exists():p=repair
    x=json.loads(p.read_text())
    for model in counts:counts[model][x['methods'][model]['status']]+=1
  if any(sum(c.values())!=677 or c['numerical_failure'] for c in counts.values()):raise ValueError('Incomplete effective forecasts')
  score=g['scores'];ci=score['bootstrap'][0]['numerical_envelope'];obs=score['exact_mean_contrast']
  rows.append(dict(variant=variant,cases=677,branching_tolerance_achieved=counts['branching']['tolerance_achieved'],poisson_tolerance_achieved=counts['matched_mean_poisson']['tolerance_achieved'],branching_resource_limit=counts['branching']['resource_limit'],poisson_resource_limit=counts['matched_mean_poisson']['resource_limit'],mean_contrast=value(obs[1]),numerical_lower=value(obs[0]),numerical_upper=value(obs[2]),bootstrap_envelope_lower=ci[0],bootstrap_envelope_upper=ci[1]))
 p=ROOT/'tables/sensitivity_scores.csv'
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=[r'\begin{table}[t]',r'\centering\small',r'\caption{Primary-target sensitivity comparisons, each with 677 paired origins. Tolerance counts are branching/Poisson and $\varepsilon$ denotes the requested midpoint tolerance; valid wider intervals remain in all scores. Envelopes combine numerical bounds with block-length-13 resampling. These are secondary comparisons without multiplicity adjustment.}',r'\label{tab:sensitivity}',r'\begin{tabular}{lccc}\toprule',r'Perturbation & Tolerance achieved & Brier contrast & 95\% envelope \\\midrule']
 for r in rows:lines.append(f"{labels[r['variant']]} & {r['branching_tolerance_achieved']}/{r['poisson_tolerance_achieved']} & {r['mean_contrast']:.6f} & [{endpoint(r['bootstrap_envelope_lower'],True)}, {endpoint(r['bootstrap_envelope_upper'],False)}] "+'\\\\')
 lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(ROOT/'tables/sensitivity_scores.tex').write_text('\n'.join(lines)+'\n')
 (ROOT/'tables/sensitivity_scores_provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Completed empirical sensitivity forecasts, with separate documented repairs.'),indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
