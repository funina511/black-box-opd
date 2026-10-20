"""Bounded compact references. Ordering is defined exclusively by source spans."""
MAX_UNITS = 8
RANGE = {'type':'object','additionalProperties':False,'required':['start','end'],
         'properties':{'start':{'type':'string'},'end':{'type':'string'}}}
# Cardinality is enforced locally: installed xgrammar does not support maxItems.
RANGES = {'type':'array','items':RANGE}


class EvidenceError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def expand(units, spans, *, allow_empty=False):
    if not isinstance(units,list) or (not units and not allow_empty):
        raise EvidenceError('event_spans_malformed')
    if len(units)>MAX_UNITS:raise EvidenceError('evidence_capacity_conflict')
    ordered=[s['id'] for s in spans]; positions={sid:i for i,sid in enumerate(ordered)}
    # Difference array bounds work by source size + number of units, including overlaps.
    delta=[0]*(len(ordered)+1)
    for unit in units:
        if not isinstance(unit,dict) or set(unit)!={'start','end'} or any(not isinstance(v,str) for v in unit.values()):
            raise EvidenceError('event_spans_malformed')
        if unit['start'] not in positions or unit['end'] not in positions:raise EvidenceError('event_span_missing')
        lo,hi=positions[unit['start']],positions[unit['end']]
        if lo>hi:raise EvidenceError('evidence_range_reversed')
        delta[lo]+=1;delta[hi+1]-=1
    active=0;result=[]
    for i,sid in enumerate(ordered):
        active+=delta[i]
        if active:result.append(sid)
    return result


def encode(ids, spans):
    """Explicit lossless migration only; never bridge uncited gaps or drop units."""
    ordered=[s['id'] for s in spans]; positions={sid:i for i,sid in enumerate(ordered)}
    if not isinstance(ids,list) or any(not isinstance(s,str) for s in ids):raise EvidenceError('event_spans_malformed')
    if any(s not in positions for s in ids):raise EvidenceError('event_span_missing')
    indices=sorted({positions[s] for s in ids}); units=[]
    for i in indices:
        if units and positions[units[-1]['end']]+1==i:units[-1]['end']=ordered[i]
        else:units.append({'start':ordered[i],'end':ordered[i]})
    if len(units)>MAX_UNITS:raise EvidenceError('evidence_capacity_conflict')
    return units


def audit(units, ids, spans):
    ratio=len(ids)/len(spans) if spans else 0
    return {'compact_ranges':units,'unit_count':len(units),'expanded_unique_spans':len(ids),
            'source_span_count':len(spans),'coverage_ratio':ratio,'broad_reference':ratio>=.8}
