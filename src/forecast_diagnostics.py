"""Descriptive discrimination/calibration; exact arithmetic for probability ranks.

Reliability groups are fixed decimal bins of interval midpoints. Observed fractions
are descriptive: no independent-binomial confidence intervals are implied.
"""
from fractions import Fraction as F
import json,hashlib
from pathlib import Path
from score_intervals import decode,rational


def diagnostics(intervals,outcomes,bins=10):
    if len(intervals)!=len(outcomes) or not intervals or bins<1:
        raise ValueError('Nonempty aligned inputs and positive bins required')
    if any(y not in (0,1) for y in outcomes) or any(not 0<=a<=b<=1 for a,b in intervals):
        raise ValueError('Invalid outcome or interval')
    mids=[(a+b)/2 for a,b in intervals];n=len(mids);npos=sum(outcomes);nneg=n-npos
    result={'cases':n,'positives':npos,'prevalence':rational(F(npos,n)),
            'interpretation':'Descriptive; dependent forecast windows; midpoint AP is not an interval guarantee.'}
    if npos and nneg:
        low=high=mid=F(0)
        pos=[i for i,y in enumerate(outcomes) if y];neg=[i for i,y in enumerate(outcomes) if not y]
        def win(a,b):return F(1) if a>b else F(1,2) if a==b else F(0)
        for i in pos:
            for j in neg:
                low+=win(intervals[i][0],intervals[j][1]);high+=win(intervals[i][1],intervals[j][0]);mid+=win(mids[i],mids[j])
        denom=npos*nneg
        result['roc_auc']={'lower':rational(low/denom),'midpoint':rational(mid/denom),'upper':rational(high/denom),
            'scope':'Conservative pairwise bounds; simultaneous endpoint attainability is not required.'}
        # Group all tied scores before updating precision: non-interpolated AP.
        groups={}
        for p,y in zip(mids,outcomes):
            g=groups.setdefault(p,[0,0]);g[0]+=y;g[1]+=1
        tp=seen=0;ap=F(0)
        for p,(positive,total) in sorted(groups.items(),reverse=True):
            tp+=positive;seen+=total;ap+=F(positive,npos)*F(tp,seen)
        result['average_precision_midpoint']=rational(ap)
    else:
        result['discrimination_status']='Undefined two-class discrimination: only one outcome class'
    reliability=[];ece=F(0)
    for k in range(bins):
        ids=[i for i,p in enumerate(mids) if min(int(p*bins),bins-1)==k]
        row={'bin_index':k,'left':rational(F(k,bins)),'right':rational(F(k+1,bins)),
             'right_closed':k==bins-1,'cases':len(ids)}
        if ids:
            z=len(ids);observed=F(sum(outcomes[i] for i in ids),z)
            avg=[sum((intervals[i][j] for i in ids),F(0))/z for j in (0,1)]
            meanmid=sum((mids[i] for i in ids),F(0))/z
            ambiguous=sum(min(int(intervals[i][0]*bins),bins-1)!=min(int(intervals[i][1]*bins),bins-1) for i in ids)
            row.update(observed_fraction=rational(observed),mean_probability_lower=rational(avg[0]),
                       mean_probability_midpoint=rational(meanmid),mean_probability_upper=rational(avg[1]),
                       interval_crosses_bin_boundary=ambiguous)
            ece+=F(z,n)*abs(meanmid-observed)
        reliability.append(row)
    result.update(reliability=reliability,ece_midpoint=rational(ece),
        calibration_note='Bins fixed at midpoint probabilities; interval-crossing counts expose membership uncertainty. ECE depends on these bins and is descriptive, not a proper score.')
    return result


def main():
    root=Path(__file__).resolve().parents[1];source=root/'results/scoring_primary_v1';meta=json.loads((source/'summary.json').read_text())
    if meta['status']!='completed':raise RuntimeError('Completed primary scores required')
    path=source/'case_scores.jsonl';groups={};identities=set()
    for line in path.open():
        x=json.loads(line);ident=(x['origin_index'],x['horizon_days'],x['magnitude_threshold'],x['model'])
        if ident in identities:raise ValueError('Duplicate case score')
        identities.add(ident);key=ident[1:];groups.setdefault(key,[]).append(x)
    expected={(x['horizon_days'],x['magnitude_threshold'],x['model']):x for x in meta['model_summaries']}
    if set(groups)!=set(expected):raise ValueError('Model group mismatch')
    output=[]
    for key,rows in sorted(groups.items()):
        if len(rows)!=expected[key]['cases']:raise ValueError('Case accounting mismatch')
        valid=[x for x in rows if x['score_status']=='scored'];missing=len(rows)-len(valid)
        result={'horizon_days':key[0],'magnitude_threshold':key[1],'model':key[2],'missing':missing,
                'scope':'available subset only' if missing else 'full calendar'}
        if valid:result['diagnostics']=diagnostics([(decode(x['probability_lower']),decode(x['probability_upper'])) for x in valid],[x['outcome'] for x in valid])
        output.append(result)
    target=root/'results/diagnostics_primary_v1';target.mkdir(exist_ok=False)
    result={'status':'completed','groups':output,'source_scores_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_summary_sha256':hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (target/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
