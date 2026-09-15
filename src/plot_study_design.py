"""Generate the study design and catalog-size figure from audited artifacts."""
import csv, hashlib, json, platform
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[1]

def main():
    table=ROOT/'tables/catalog_partitions.csv'
    provenance=ROOT/'tables/catalog_partitions_provenance.json'
    evidence=json.loads(provenance.read_text())
    if hashlib.sha256(table.read_bytes()).hexdigest()!=evidence['table_sha256']:
        raise ValueError('Table differs from audited source')
    rows=list(csv.DictReader(table.open()))
    calendar=ROOT/'experiments/forecast_origins_v1_summary.json'
    dates=json.loads(calendar.read_text())
    colors=['#767676','#0072B2','#E69F00','#009E73']
    parts=list(dict.fromkeys(r['partition'] for r in rows))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                         'axes.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(2,1,figsize=(7.1,5.0),layout='constrained',gridspec_kw={'height_ratios':[1,1.35]})
    for i,label in enumerate(parts):
        row=next(r for r in rows if r['partition']==label)
        a,b=[mdates.date2num(datetime.fromisoformat(row[k])) for k in ('start_inclusive','end_exclusive')]
        ax.barh(i,b-a,left=a,height=.55,color=colors[i])
    first,last=[mdates.date2num(datetime.fromisoformat(dates[k].replace('Z','+00:00'))) for k in ('first_origin','last_origin')]
    ax.plot([first,last],[3,3],color='black',linewidth=1.2,marker='|',markersize=9)
    ax.set_yticks(range(4),['Auxiliary history','Training likelihood','Validation','Test catalog'])
    ax.invert_yaxis();ax.xaxis.set_major_locator(mdates.YearLocator(10));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Calendar year (UTC; half-open partitions)');ax.set_title('(a) Fixed chronological design',loc='left',fontweight='bold')
    ax.text(.99,.97,f"{dates['origins']:,} weekly forecast origins\n{dates['forecast_cases']:,} planned horizon–threshold cases",transform=ax.transAxes,ha='right',va='top',fontsize=7.5)
    width=.22
    for j,cutoff in enumerate(['2.5','3.0','3.5']):
        vals=[int(next(r['events'] for r in rows if r['partition']==label and r['magnitude_cutoff']==cutoff)) for label in parts]
        bars=bx.bar([i+(j-1)*width for i in range(4)],vals,width,label=f'M ≥ {cutoff}',color=['#56B4E9','#0072B2','#333333'][j])
        for bar,n in zip(bars,vals):
            bx.annotate(f'{n:,}',(bar.get_x()+bar.get_width()/2,n),xytext=(0,3),textcoords='offset points',ha='center',fontsize=6.6,rotation=90)
    bx.set_xticks(range(4),['Auxiliary','Training','Validation','Test catalog']);bx.set_ylim(0,53000)
    bx.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:g}k'))
    bx.set_ylabel('Catalog records');bx.legend(frameon=False,ncol=3,loc='upper right')
    bx.set_title('(b) Records retained at each modeling cutoff',loc='left',fontweight='bold')
    bx.grid(axis='y',alpha=.2);bx.set_axisbelow(True)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'study_design.{ext}',dpi=240)
    plt.close(fig)
    report={'source_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [table,provenance,calendar,Path(__file__)]},
            'outputs':{f'study_design.{ext}':hashlib.sha256((out/f'study_design.{ext}').read_bytes()).hexdigest() for ext in ['pdf','svg','png']},
            'python':platform.python_version(),'matplotlib':matplotlib.__version__,
            'forecast_outcomes_read':False,'interpretation':'Descriptive design and catalog counts; no forecast skill claim.'}
    (out/'study_design_provenance.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
