"""Compare recent and older conditional win relationships, with shrinkage."""
import numpy as np
from hgb_upgrade.features import build_advanced_features as base_features,PAYOUT_TO_IDX


def category(p):
    return None if p is None else min(p,2)


def changing_relationships(sequence,cutoff):
    past={t:p for t,p in sequence.items() if t<cutoff}
    result=[]
    for lag in range(1,9):
        wanted=category(past.get(cutoff-lag))
        for recent_window in (32,64):
            recent=[t for t in past if cutoff-recent_window<=t]
            older=[t for t in past if cutoff-192<=t<cutoff-recent_window]
            def estimate(ticks):
                baseline=(sum(past[t]>0 for t in ticks)+1)/(len(ticks)+2)
                matched=[t for t in ticks if category(past.get(t-lag))==wanted] if wanted is not None else []
                conditional=(sum(past[t]>0 for t in matched)+8*baseline)/(len(matched)+8)
                # Difference from the window's marginal rate isolates association
                # from a simple overall shift in win frequency.
                return len(matched),conditional,conditional-baseline
            nr,pr,lr=estimate(recent); no,po,lo=estimate(older)
            if wanted is None:
                result += [0,0]+[np.nan]*6
            else:
                result += [nr,no,pr,po,lr,lo,lr-lo,lr*lo]
    return result


def build_advanced_features(query,sequence,peer_sequences,cutoff,base_forecaster=None):
    # Filter at this boundary too, including for historical training examples.
    past={t:p for t,p in sequence.items() if t<cutoff}
    peers=[{t:p for t,p in s.items() if t<cutoff} for s in peer_sequences]
    return base_features(query,past,peers,cutoff,base_forecaster)+changing_relationships(past,cutoff)
