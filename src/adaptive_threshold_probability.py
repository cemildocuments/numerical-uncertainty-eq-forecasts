"""Adaptive threshold-absorption enclosure; exact rational stopping decisions.

Global Flint precision belongs to the underlying solver: use separate processes,
not threads. Policy heuristics do not establish the enclosure theorem.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
import time
from validated_probability import decimal_value, positive_integer
from threshold_probability import enclose
from controlled_mean_probability import validate


def endpoint(record, side):
    d = record[side]
    return Fraction(int(d['mantissa'])) * Fraction(2)**int(d['exponent'])


def rational(x):
    return dict(numerator=str(x.numerator), denominator=str(x.denominator))


def solve(config, policy):
    """Return a valid aggregate interval even after budget exhaustion, if available.

    Required policy: initial_mark_bins, max_time_bins, max_mark_bins,
    max_evaluations, wall_seconds, absolute_probability_tolerance.
    Config time_bins supplies the initial time resolution. Marks are integrated
    below the target threshold using unconditional GR probability masses.
    Every attempted bound, including inadmissible upper-grid means, costs budget.
    """
    p = deepcopy(policy)
    for name in ('initial_mark_bins','max_time_bins','max_mark_bins','max_evaluations'):
        positive_integer(p[name], name)
    tolerance = decimal_value(p['absolute_probability_tolerance'], 'tolerance')
    wall = decimal_value(p['wall_seconds'], 'wall_seconds')
    if not 0 < tolerance < 1 or wall <= 0:
        raise ValueError('Tolerance must be in (0,1) and wall budget positive')
    c = validate(config)
    b = decimal_value(c['b'],'b'); a = decimal_value(c['alpha'],'alpha')
    k = decimal_value(c['productivity'],'productivity')
    if b <= a or k*b/(b-a) >= 1:
        raise ValueError('Reference model requires b>alpha and branching mean<1')
    if c['time_bins'] > p['max_time_bins'] or p['initial_mark_bins'] > p['max_mark_bins']:
        raise ValueError('Initial grid exceeds policy limits')
    started = time.perf_counter(); trace = []; cache = {}; seed_cache = {}
    lower = upper = None; lower_record = upper_record = None
    failure = None

    def evaluate(candidate, marks):
        nonlocal lower, upper, lower_record, upper_record, failure
        key = (candidate['time_bins'], marks)
        if key in cache:
            return cache[key]
        if (candidate['time_bins'] > p['max_time_bins'] or marks > p['max_mark_bins']
            or len(trace) >= p['max_evaluations'] or time.perf_counter()-started >= float(wall)):
            return None
        tick = time.perf_counter()
        row = dict(time_bins=candidate['time_bins'],mark_bins=marks)
        result = None
        try:
            result = enclose(candidate, marks, _seed_cache=seed_cache)
            lo = endpoint(result['lower_endpoint'], 'lower')
            hi = endpoint(result['upper_endpoint'], 'upper')
            newlo = lo if lower is None else max(lower,lo)
            newhi = hi if upper is None else min(upper,hi)
            if not 0 <= newlo <= newhi <= 1:
                raise ArithmeticError('Combined valid intervals are inconsistent')
            if lower is None or lo > lower:
                lower_record = result['lower_endpoint']
            if upper is None or hi < upper:
                upper_record = result['upper_endpoint']
            lower,upper = newlo,newhi
            row.update(status='enclosed', width=rational(hi-lo),
                       lower_endpoint=result['lower_endpoint'],upper_endpoint=result['upper_endpoint'])
        except ValueError as error:
            if 'upper non-target branching mean' in str(error):
                row.update(status='upper_grid_not_subcritical',error=str(error))
            else:
                row.update(status='numerical_failure',error=f'{type(error).__name__}: {error}')
                failure = row['error']
        except Exception as error:
            row.update(status='numerical_failure',error=f'{type(error).__name__}: {error}')
            failure = row['error']
        row['seconds'] = time.perf_counter()-tick
        trace.append(row)
        value = dict(config=deepcopy(candidate),marks=marks,result=result,row=row)
        cache[key] = value
        return value

    current = evaluate(c,p['initial_mark_bins'])
    while current is not None and failure is None:
        if lower is not None and upper-lower <= 2*tolerance:
            break
        if current['result'] is None:
            current = evaluate(current['config'],current['marks']*2)
            continue
        candidates = []
        for double_time in (True,False):
            updated = deepcopy(current['config'])
            marks = current['marks']
            if double_time: updated['time_bins'] *= 2
            else: marks *= 2
            value = evaluate(updated,marks)
            if value is not None: candidates.append(value)
            if failure is not None or (lower is not None and upper-lower <= 2*tolerance):
                break
        if not candidates:
            break
        valid = [v for v in candidates if v['result'] is not None]
        if valid:
            # Candidate choice is heuristic; all valid intervals contribute to
            # the exact intersection used by the independent stopping decision.
            current = min(valid,key=lambda v: endpoint(v['result']['upper_endpoint'],'upper')-
                          endpoint(v['result']['lower_endpoint'],'lower'))
        else:
            current = candidates[-1]
    achieved = failure is None and lower is not None and upper-lower <= 2*tolerance
    elapsed = time.perf_counter()-started
    answer = dict(status='numerical_failure' if failure else ('tolerance_achieved' if achieved else 'resource_limit'),
                  error=failure, evaluations=len(trace), wall_seconds=elapsed,
                  wall_budget_exceeded=elapsed>float(wall), trace=trace,
                  policy=p, input_config=c,
                  source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ('adaptive_threshold_probability.py','threshold_probability.py','controlled_mean_probability.py',
                                              'validated_probability.py','validate_ball_zero.py')})
    if lower is not None:
        answer.update(lower_endpoint=lower_record,upper_endpoint=upper_record,
                      zero_count_width=rational(upper-lower),
                      midpoint_error_bound=rational((upper-lower)/2),
                      exceedance_lower=rational(1-upper),exceedance_upper=rational(1-lower))
    return answer
