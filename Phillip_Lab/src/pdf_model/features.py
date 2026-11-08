"""Observed history geometry and within-peer changes; no condition labels."""
import numpy as np
from lag_model.features import build_advanced_features as lag_features,PAYOUT_TO_IDX


def structural_features(sequence,peers,cutoff):
    seq={t:p for t,p in sequence.items() if t<cutoff}
    peers=[{t:p for t,p in s.items() if t<cutoff} for s in peers]
    ticks=sorted(seq)
    span=(ticks[-1]-ticks[0]+1) if ticks else 0
    gaps=np.diff(ticks)
    consecutive=0
    while cutoff-1-consecutive in seq:
        consecutive+=1
    result=[len(seq),span,len(seq)/max(1,span),consecutive,
            max(gaps,default=0),sum(gaps>1),cutoff-1-ticks[-1] if ticks else cutoff]
    for window in (8,40,64,128):
        result.append(sum(t>=cutoff-window for t in seq)/window)
    # Center each peer on its own older behavior instead of pooling raw payouts
    # from machines with different expected payout distributions.
    for window in (8,24,48):
        changes=[]; means=[]; supports=[]
        for s in peers:
            recent=[p for t,p in s.items() if t>=cutoff-window]
            older=[p for t,p in s.items() if cutoff-192<=t<cutoff-window]
            if not recent or not older: continue
            r=(sum(p>0 for p in recent)+1)/(len(recent)+2)
            o=(sum(p>0 for p in older)+1)/(len(older)+2)
            changes.append(r-o); means.append(r); supports.append(len(recent))
        if changes:
            result += [np.mean(changes),np.std(changes),np.mean(np.abs(changes)),
                       np.mean(np.asarray(changes)>0),np.mean(means),sum(supports)]
        else:
            result += [np.nan]*5+[0]
    return result


def build_advanced_features(query,sequence,peer_sequences,cutoff,base_forecaster=None):
    return lag_features(query,sequence,peer_sequences,cutoff,base_forecaster)+structural_features(sequence,peer_sequences,cutoff)
