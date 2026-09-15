"""Exact catalog-time membership; serialized decimal model histories."""
import csv,calendar,hashlib,json
from datetime import datetime
from decimal import Decimal
DAY_NS=86400000000000


def timestamp_ns(text):
    value=text.rstrip('Z')
    if '.' in value:
        base,fraction=value.split('.',1)
        if not fraction.isdigit() or len(fraction)>9:raise ValueError('Unsupported timestamp fraction')
    else:base,fraction=value,''
    date=datetime.fromisoformat(base)
    if date.tzinfo is not None:raise ValueError('Use UTC Z or source naive UTC timestamps')
    return calendar.timegm(date.timetuple())*1000000000+int(fraction.ljust(9,'0') or '0')


def load_catalog(path,expected_sha256):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('Catalog hash mismatch')
    events=[];previous=None
    with path.open() as f:
        for row in csv.DictReader(f):
            t=timestamp_ns(row['time'])
            if previous is not None and t<previous:raise ValueError('Catalog times not sorted')
            previous=t;events.append(dict(time_ns=t,magnitude=row['magnitude'],id=row['id']))
    return events


def past_events(events,origin_ns):
    # Sorted input; never expose the event at or after issuance to model helpers.
    result=[]
    for event in events:
        if event['time_ns']>=origin_ns:break
        if event['time_ns']>=timestamp_ns('1971-01-01'):result.append(event)
    return result


def model_history(past,origin_ns,cutoff):
    minimum=Decimal(cutoff);history=[]
    for event in past:
        if event['time_ns']>=origin_ns:raise ValueError('Future or issuance-time event passed to model')
        if Decimal(event['magnitude'])>=minimum:
            history.append(dict(time=format((event['time_ns']-origin_ns)/DAY_NS,'.17g'),magnitude=event['magnitude']))
    return history


def count_window(events,lo,hi,threshold):
    if hi<lo:raise ValueError('Invalid window')
    threshold=Decimal(threshold)
    return sum(lo<=e['time_ns']<hi and Decimal(e['magnitude'])>=threshold for e in events)


def history_hash(history):
    return hashlib.sha256(json.dumps(history,sort_keys=True,separators=(',',':')).encode()).hexdigest()
