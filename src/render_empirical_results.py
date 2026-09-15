"""Tables and vector figures from immutable completed empirical summaries."""
from pathlib import Path
from fractions import Fraction
import json,csv,hashlib,statistics,math,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
MODELS={'branching':'Branching','matched_mean_poisson':'Matched-mean Poisson','historical_rate':'Historical rate','recent_rate':'Trailing-year rate'}
def val(x):return float(Fraction(int(x['numerator']),int(x['denominator']))) if isinstance(x,dict) else float(x)
def floorf(x,n):return f'{math.floor(x*10**n)/10**n:.{n}f}'
def ceilf(x,n):return f'{math.ceil(x*10**n)/10**n:.{n}f}'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submission-layout',action='store_true');args=parser.parse_args()
    (ROOT/'figures').mkdir(exist_ok=True);(ROOT/'tables').mkdir(exist_ok=True)
    paths=[ROOT/('results/'+p+'/summary.json') for p in ['scoring_final_v1','diagnostics_final_v1','log_loss_final_v1','numerical_comparison_v1']]
    if args.submission_layout:paths=[ROOT/('results/'+name+'.json') for name in ['primary_summary','diagnostics','log_loss','numerical_comparisons']]
    s,d,l,n=[json.loads(p.read_text()) for p in paths]
    if any(x['status']!='completed' for x in [s,d,l,n]):raise ValueError('Incomplete empirical results')
    select=lambda x:str(x['horizon_days'])=='7' and str(x['magnitude_threshold'])=='4.5'
    dg={x['model']:x['diagnostics'] for x in d['groups'] if select(x)};lg={x['model']:x for x in l['groups'] if select(x)}
    rows=[]
    for g in s['model_summaries']:
        if not select(g):continue
        if g['missing'] or g['cases']!=677 or g['positive_windows']!=113:raise ValueError('Unexpected primary accounting')
        q=dg[g['model']];z=lg[g['model']]['mean_log_loss']
        rows.append(dict(model=g['model'],cases=g['cases'],positives=g['positive_windows'],brier=val(g['brier_midpoint_mean']),brier_lower=val(g['brier_mean_lower']),brier_upper=val(g['brier_mean_upper']),log_loss_lower=val(z['lower']),log_loss_upper=val(z['upper']),auc=val(q['roc_auc']['midpoint']),auc_lower=val(q['roc_auc']['lower']),auc_upper=val(q['roc_auc']['upper']),average_precision=val(q['average_precision_midpoint']),ece=val(q['ece_midpoint'])))
    out=ROOT/'tables/primary_scores.csv'
    with out.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tex=[r'\begin{table}[t]',r'\centering\small',r'\caption{Primary seven-day $M\ge4.5$ scores for 677 origins (113 positive windows). Brackets enclose numerical uncertainty; AUC and AP use probability midpoints and are descriptive. Lower Brier and log loss are better.}',r'\label{tab:primary}',r'\begin{tabular}{lcccc}\toprule',r'Model & Brier [bounds] & Log loss [bounds] & AUC & AP \\\midrule']
    for r in rows:tex.append(f"{MODELS[r['model']]} & {r['brier']:.6f} [{floorf(r['brier_lower'],6)}, {ceilf(r['brier_upper'],6)}] & [{floorf(r['log_loss_lower'],4)}, {ceilf(r['log_loss_upper'],4)}] & {r['auc']:.3f} & {r['average_precision']:.3f} " + r"\\")
    tex += [r'\bottomrule\end{tabular}',r'\end{table}'];(ROOT/'tables/primary_scores.tex').write_text('\n'.join(tex)+'\n')
    c=[x['comparison'] for x in n['cases']];numeric=dict(cases=len(c),quadrature_inside=sum(x['quadrature_inside_enclosure'] for x in c),monte_carlo_overlap=sum(x['monte_carlo_interval_overlaps_enclosure'] for x in c),monte_carlo_unknown=sum(x['unknown_replicates'] for x in c),branching_width_median=statistics.median(val(x['branching_width']) for x in c),monte_carlo_width_median=statistics.median(val(x['monte_carlo_width']) for x in c),quadrature_max_distance=max(val(x['quadrature_distance_from_enclosure']) for x in c))
    for model in ['branching','quadrature','monte_carlo']:
        times=[x[model+'_wall_seconds'] for x in c];numeric[model+'_median_seconds']=statistics.median(times);numeric[model+'_max_seconds']=max(times)
    (ROOT/'tables/numerical_summary.json').write_text(json.dumps(numeric,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.1,3.8),layout='constrained')
    labels=[]
    for i,h in enumerate(['1','7','30']):
        for j,m in enumerate(['3.5','4.5','5.5']):
            by={x['model']:x for x in s['model_summaries'] if (x['horizon_days'],x['magnitude_threshold'])==(h,m)};a=by['branching'];b=by['matched_mean_poisson'];mid=val(a['brier_midpoint_mean'])-val(b['brier_midpoint_mean']);lo=val(a['brier_mean_lower'])-val(b['brier_mean_upper']);hi=val(a['brier_mean_upper'])-val(b['brier_mean_lower']);y=i*3+j
            ax.errorbar(mid,y,xerr=[[mid-lo],[hi-mid]],fmt='o',color='#0072B2' if (h,m)==('7','4.5') else '#555555',capsize=3,markersize=5)
            labels.append(f'{h} d, M ≥ {m}')
    ax.axvline(0,color='#999999',linestyle='--',linewidth=1);ax.set(yticks=range(9),yticklabels=labels,xlabel='Mean Brier difference: branching − matched-mean Poisson');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/('figures/brier_grid.'+ext),dpi=220)
    plt.close(fig)
    evidence={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]}
    (ROOT/'tables/empirical_render_provenance.json').write_text(json.dumps(dict(evidence=evidence,matplotlib=matplotlib.__version__,scope='Completed empirical outputs only; plot bars are numerical bounds, not sampling intervals.'),indent=2)+'\n')
    print(json.dumps(numeric,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
