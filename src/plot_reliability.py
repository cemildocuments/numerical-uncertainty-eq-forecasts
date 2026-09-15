"""Render reliability summaries; horizontal bars show numerical uncertainty only."""
import argparse, hashlib, json
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
MODELS={'branching':'Branching probability','matched_mean_poisson':'Matched-mean Poisson','historical_rate':'Historical rate','recent_rate':'Trailing-year rate'}


def value(record):
    return float(Fraction(int(record['numerator']),int(record['denominator'])))


def render(summary,output,*,synthetic=False):
    if summary.get('status')!='completed':raise ValueError('Completed diagnostics required')
    groups=[g for g in summary['groups'] if str(g['horizon_days'])=='7' and str(g['magnitude_threshold'])=='4.5']
    if len(groups)!=4 or {g['model'] for g in groups}!=set(MODELS):raise ValueError('Four unique primary model groups required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.1),layout='constrained',sharex=True,sharey=True)
    by={g['model']:g for g in groups}
    for i,(model,title) in enumerate(MODELS.items()):
        ax=axes.flat[i];group=by[model];d=group.get('diagnostics')
        ax.plot([0,1],[0,1],color='#777777',linestyle='--',linewidth=.8)
        ax.set_title(f'({chr(97+i)}) {title}',loc='left',fontsize=9)
        if d:
            bins=d['reliability']
            if len(bins)!=10 or [b['bin_index'] for b in bins]!=list(range(10)):raise ValueError('Expected ten ordered fixed bins')
            if sum(b['cases'] for b in bins)!=d['cases']:raise ValueError('Bin counts do not match diagnostic cases')
            total_positive=sum((Fraction(int(b['observed_fraction']['numerator']),int(b['observed_fraction']['denominator']))*b['cases'] for b in bins if b['cases']),Fraction(0))
            if total_positive!=d['positives']:raise ValueError('Bin outcomes do not match positive count')
            for b in bins:
                if not b['cases']:continue
                lo,mid,hi=[value(b['mean_probability_'+k]) for k in ['lower','midpoint','upper']]
                obs=value(b['observed_fraction'])
                if not 0<=lo<=mid<=hi<=1 or not 0<=obs<=1:raise ValueError('Invalid reliability values')
                ax.errorbar(mid,obs,xerr=[[mid-lo],[hi-mid]],fmt='o',color='#0072B2',markersize=4,capsize=2,linewidth=.9)
                ax.annotate(str(b['cases']),(mid,obs),xytext=(4,4),textcoords='offset points',fontsize=6.5)
            ambiguous=sum(b.get('interval_crosses_bin_boundary',0) for b in bins)
            note=f"n={d['cases']}; positives={d['positives']}\nMissing={group['missing']}; bin crossings={ambiguous}"
        else:note=f"No valid probabilities\nMissing={group['missing']}"
        ax.text(.03,.97,note,transform=ax.transAxes,ha='left',va='top',fontsize=7,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
        ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='Mean forecast probability',ylabel='Observed event fraction')
        ax.grid(alpha=.15)
    if synthetic:fig.suptitle('SYNTHETIC LAYOUT CHECK — NOT RESEARCH RESULTS',color='#9c2222',fontsize=10)
    output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(output.with_suffix('.'+ext),dpi=220)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics',default='results/diagnostics_primary_v1/summary.json');parser.add_argument('--scores',default='results/scoring_primary_v1');args=parser.parse_args()
    path=(ROOT/args.diagnostics).resolve()
    if not path.is_relative_to(ROOT/'results'):raise ValueError('Diagnostics must come from project results')
    summary=json.loads(path.read_text())
    scores=ROOT/args.scores
    for filename,key in [('case_scores.jsonl','source_scores_sha256'),('summary.json','source_summary_sha256')]:
        if hashlib.sha256((scores/filename).read_bytes()).hexdigest()!=summary[key]:raise ValueError('Diagnostics source hash mismatch')
    output=ROOT/'figures/reliability_primary'
    if any(output.with_suffix('.'+x).exists() for x in ['pdf','svg','png']):raise FileExistsError('Refusing to overwrite reliability figure')
    render(summary,output)
    evidence={'diagnostics_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,
              'interpretation':'Midpoint-defined bins; horizontal bars are numerical enclosures of mean probability, not sampling confidence intervals. Point labels give bin counts.'}
    (ROOT/'figures/reliability_primary_provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')

if __name__=='__main__':main()
