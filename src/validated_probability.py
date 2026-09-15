"""Validated interface to the archived synthetic zero-count kernel.

Enclosures are conditional on a fixed bounded-mark temporal branching model.
This interface does not validate the model against observations.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
from pathlib import Path
from flint import arb, ctx
from validate_continuous_marks import bound
from validate_ball_zero import encode


def decimal_value(value, name):
    # Reject binary floats: callers must save the actual decimal model input.
    if type(value) not in (str, int):
        raise ValueError(f"{name} must be a decimal string or integer")
    try:
        d = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a finite decimal") from None
    if not d.is_finite():
        raise ValueError(f"{name} must be finite")
    return Fraction(d)


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_inputs(config, mark_bins):
    c = deepcopy(config)
    keys = ('horizon', 'mu', 'c', 'p', 'm0', 'mmax', 'threshold',
            'b', 'alpha', 'productivity', 'radius_tolerance')
    q = {key: decimal_value(c[key], key) for key in keys}
    for key in ('horizon', 'c', 'b'):
        if q[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ('mu', 'alpha', 'productivity'):
        if q[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if q['p'] <= 1:
        raise ValueError("p must exceed one for the normalized lag density")
    if q['mmax'] <= q['m0']:
        raise ValueError("mmax must exceed m0")
    if not 0 < q['radius_tolerance'] < 1:
        raise ValueError("radius_tolerance must lie strictly between zero and one")
    positive_integer(c['time_bins'], 'time_bins')
    positive_integer(mark_bins, 'mark_bins')
    positive_integer(c['max_iterations'], 'max_iterations')
    positive_integer(c['precision_bits'], 'precision_bits')
    if c['precision_bits'] < 64:
        raise ValueError("precision_bits must be at least 64")
    numerator = c['threshold_bin_fraction_numerator']
    denominator = positive_integer(c['threshold_bin_fraction_denominator'], 'threshold denominator')
    if type(numerator) is not int or not 0 <= numerator <= denominator:
        raise ValueError("threshold fraction must lie in [0,1]")
    if numerator * mark_bins % denominator:
        raise ValueError("threshold does not coincide with a mark-bin boundary")
    if q['m0'] + (q['mmax']-q['m0']) * Fraction(numerator, denominator) != q['threshold']:
        raise ValueError("threshold and bin fraction are not exactly equal")
    if not isinstance(c['history'], list):
        raise ValueError("history must be a list")
    for i, event in enumerate(c['history']):
        t = decimal_value(event['time'], f"history[{i}].time")
        m = decimal_value(event['magnitude'], f"history[{i}].magnitude")
        if t >= 0:
            raise ValueError("history times must precede the forecast origin")
        if not q['m0'] <= m <= q['mmax']:
            raise ValueError("history magnitude lies outside the configured support")
    return c


def zero_count_enclosure(config, mark_bins):
    c = validate_inputs(config, mark_bins)
    previous_precision = ctx.prec
    try:
        ctx.prec = c['precision_bits']
        # Explicit guard, retained even when Python assertions are disabled.
        low = arb(c['m0']); span = arb(c['mmax']) - low
        beta = arb(c['b']) * arb(10).log()
        alpha = arb(c['alpha']) * arb(10).log()
        normalizer = 1 - (-beta * span).exp()
        mean_upper = arb(0)
        for j in range(mark_bins):
            left = span * j / mark_bins; right = span * (j+1) / mark_bins
            mass = ((-beta * left).exp() - (-beta * right).exp()) / normalizer
            mean_upper += mass * arb(c['productivity']) * (alpha * right).exp()
        if not mean_upper < 1:
            raise ValueError("upper discretized branching mean is not certified below one")
        early, _ = bound(c, mark_bins, True)
        late, _ = bound(c, mark_bins, False)
        lo = early.lower(); hi = late.upper()
        # Intersect with the exact probability domain, preserving enclosure.
        lo = max(lo, arb(0)); hi = min(hi, arb(1))
        if not lo <= hi:
            raise ArithmeticError("probability enclosure is empty")
        source = Path(__file__).parent
        return dict(parameters=c, mark_bins=mark_bins,
                    lower_endpoint=encode(lo), upper_endpoint=encode(hi),
                    width=encode(hi-lo), upper_branching_mean=encode(mean_upper),
                    source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                   for name in ('validated_probability.py', 'validate_continuous_marks.py',
                                                'validate_ball_zero.py')})
    finally:
        ctx.prec = previous_precision
