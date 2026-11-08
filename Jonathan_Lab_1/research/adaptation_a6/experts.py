"""Cutoff-safe Bayesian temporal expert bank, including payout-size contexts."""
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.special import betaln,softmax

CONFIGS=[(encoding,half,strength,temp) for encoding in (2,3) for half in (0,24,64) for strength in (1,4,16) for temp in (1.,2.)]
LAGS=[(i,) for i in range(1,9)]+list(combinations(range(1,9),2))


def replace_win(P,rates,weight):
    win=(1-weight)*(1-P[:,0])+weight*rates
    return np.column_stack((1-win,P[:,1:]/P[:,1:].sum(1,keepdims=True)*win[:,None]))


def bank(queries,history,P):
    indexed=defaultdict(dict)
    for r in history: indexed[r.episode_id,r.machine_id][r.tick]=r.payout
    rows=[]; meta=[]
    for n,(q,p) in enumerate(zip(queries,P)):
        seq=indexed[q.episode_id,q.machine_id]; cutoff=q.cutoff
        full=np.full(cutoff,-1,dtype=int)
        for t,v in seq.items():
            if 0<=t<cutoff: full[t]=min(v,2)
        ticks=np.flatnonzero(full>=0); y=(full[ticks]>0).astype(float)
        outputs=[]
        for encoding in (2,3):
            source=np.where(full<0,-1,(full>0).astype(int)) if encoding==2 else full
            templates=[]
            for lags in LAGS:
                wanted=[source[cutoff-lag] if cutoff>=lag else -1 for lag in lags]
                if min(wanted)<0: continue
                valid=ticks>=max(lags); ts=ticks[valid]; yy=y[valid]
                code=np.zeros(len(ts),int); ok=np.ones(len(ts),bool); want=0
                for lag,w in zip(lags,wanted):
                    v=source[ts-lag]; ok&=v>=0; code=encoding*code+np.maximum(v,0); want=encoding*want+w
                templates.append((ts[ok],yy[ok],code[ok],encoding**len(lags),want))
            for half in (0,24,64):
                w=np.ones(len(ticks)) if half==0 else 2.**(-(cutoff-1-ticks)/half)
                summaries=[]
                for ts,yy,code,size,want in templates:
                    ww=np.ones(len(ts)) if half==0 else 2.**(-(cutoff-1-ts)/half)
                    wins=np.bincount(code,weights=ww*yy,minlength=size); losses=np.bincount(code,weights=ww*(1-yy),minlength=size)
                    summaries.append((wins,losses,want))
                for strength in (1,4,16):
                    a=max(.01,strength*(1-p[0])); b=max(.01,strength*p[0])
                    rates=[(a+np.dot(w,y))/(a+b+w.sum())]; evidence=[0.]
                    for wins,losses,want in summaries:
                        rates.append((a+wins[want])/(a+b+wins[want]+losses[want]))
                        evidence.append(np.sum(betaln(a+wins,b+losses)-betaln(a,b))-(betaln(a+wins.sum(),b+losses.sum())-betaln(a,b)))
                    logprior=np.r_[np.log(.5),np.full(len(rates)-1,np.log(.5/max(1,len(rates)-1)))]
                    for temp in (1.,2.): outputs.append(float(np.dot(softmax(np.asarray(evidence)/temp+logprior),rates)))
        rows.append(outputs)
        # History topology, changes in rates, peer synchronization, and lag polarity.
        m=[len(ticks)/256,float(len(ticks)<=64),q.signal,*[float(q.machine_type==kind) for kind in ('steady','balanced','jackpot')]]
        for window in (16,40,96,256):
            t=ticks[ticks>=cutoff-window]; v=full[t]
            m.extend([np.mean(v>0) if len(v) else .5,np.mean(v==1) if len(v) else .2,np.mean(v>=2) if len(v) else .2])
            peers=[v for (ep,machine),s in indexed.items() if ep==q.episode_id and machine!=q.machine_id for t,v in s.items() if cutoff-window<=t<cutoff]
            m.extend([np.mean(np.asarray(peers)>0) if peers else .5,np.mean(np.asarray(peers)==1) if peers else .2])
        for lag in range(1,9):
            valid=ticks>=lag; ts=ticks[valid]; ok=full[ts-lag]>=0; ts=ts[ok]
            yy=(full[ts]>0).astype(float); prev=(full[ts-lag]>0).astype(float)
            m.extend([float(np.mean(yy*prev)-np.mean(yy)*np.mean(prev)) if len(ts) else 0.,float(source[cutoff-lag]) if cutoff>=lag else -1.])
        meta.append(m)
        if n%128==0: print('Expert features',n,'/',len(queries),flush=True)
    return np.asarray(rows),np.asarray(meta)
