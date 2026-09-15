"""Aggregate unclipped logarithmic scores from completed primary case scores.

Numerical bounds are conditional on supplied probability intervals. They are
not sampling confidence intervals or bounds on catalog/model uncertainty.
"""
from pathlib import Path
from fractions import Fraction as F
import json, hashlib
from score_intervals import decode, rational, endpoint, log_loss


def summarize(rows):
    if not rows:
        raise ValueError('Nonempty case group required')
    sums = [F(0), F(0), F(0), F(0)]
    infinite = [0, 0, 0, 0]
    valid = 0
    for row in rows:
        if row['score_status'] == 'missing_probability':
            continue
        if row['score_status'] != 'scored':
            raise ValueError('Unknown scoring status')
        lo, hi = decode(row['probability_lower']), decode(row['probability_upper'])
        y = row['outcome']
        bounds = log_loss(lo, hi, y)
        midpoint = log_loss((lo + hi) / 2, (lo + hi) / 2, y)
        # Decode outward dyadics exactly; no clipping and no float accumulation.
        for j, (value, side) in enumerate([(bounds['lower'], 'lower'),
                (bounds['upper'], 'upper'), (midpoint['lower'], 'lower'),
                (midpoint['upper'], 'upper')]):
            if value == 'infinity':
                infinite[j] += 1
            else:
                sums[j] += endpoint(value, side)
        valid += 1
    result = dict(cases=len(rows), scored_cases=valid, missing=len(rows)-valid,
                  scope='full calendar' if valid == len(rows) else 'available subset only')
    if valid:
        names = ['lower', 'upper', 'midpoint_lower', 'midpoint_upper']
        result['mean_log_loss'] = {name: 'infinity' if infinite[j] else rational(sums[j]/valid)
                                  for j, name in enumerate(names)}
        result['infinite_case_counts'] = dict(zip(names, infinite))
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'results/scoring_primary_v1'
    summary_path = source/'summary.json'
    meta = json.loads(summary_path.read_text())
    if meta['status'] != 'completed':
        raise RuntimeError('Completed primary scores required')
    path = source/'case_scores.jsonl'
    groups, identities = {}, set()
    for line in path.open():
        row = json.loads(line)
        key = (row['horizon_days'], row['magnitude_threshold'], row['model'])
        identity = (row['origin_index'],) + key
        if identity in identities:
            raise ValueError('Duplicate case score')
        identities.add(identity)
        groups.setdefault(key, []).append(row)
    expected = {(x['horizon_days'], x['magnitude_threshold'], x['model']): x
                for x in meta['model_summaries']}
    if set(groups) != set(expected):
        raise ValueError('Model groups differ from completed scores')
    output = []
    for key, rows in sorted(groups.items()):
        result = summarize(rows)
        if any(result[k] != expected[key][k] for k in ('cases', 'scored_cases', 'missing')):
            raise ValueError('Case accounting differs from completed scores')
        output.append(dict(horizon_days=key[0], magnitude_threshold=key[1], model=key[2], **result))
    target = root/'results/log_loss_primary_v1'
    target.mkdir(exist_ok=False)
    files = [path, summary_path, Path(__file__), root/'src/score_intervals.py', root/'src/validate_ball_zero.py']
    result = dict(status='completed', groups=output,
        interpretation='Descriptive mean logarithmic scores, natural logarithm. Bounds reflect numerical probability uncertainty only; no significance claim. Missing cases are not imputed and infinities are not clipped.',
        evidence={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (target/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

if __name__ == '__main__':
    main()
