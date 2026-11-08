"""Machine-local categorical temporal experts selected without audit feedback."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import betaln,softmax

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from slot_forecasting.io import load_inputs,read_targets,read_rows,read_submission,write_submission,write_json
from slot_forecasting.experiments import episode_folds,subset_score
from slot_forecasting.schema import index_history

CONFIGS=[(half,strength) for half in (32,96,None) for strength in (4,16)]
CONTEXTS=[(i,) for i in range(1,9)]+list(combinations(range(1,9),2))


def priors(q,h):
    indexed=index_history(q,h); counts=defaultdict(lambda:np.ones(2))
    for query in q:
        for r in indexed.get((query.episode_id,query.machine_id),[]):
            counts[(query.machine_type,query.signal)][int(r.payout>0)]+=1
    return {key:c[1]/sum(c) for key,c in counts.items()}


def machine_features(query,rows,prior):
    cutoff=query.cutoff
    seq=np.full(cutoff,-1,dtype=int)
    for row in rows:
        if row.tick>=cutoff: raise ValueError('Future history')
        seq[row.tick]=min(row.payout,2)
    observed=np.flatnonzero(seq>=0); y=(seq[observed]>0).astype(float)
    templates=[]
    for binary in (False,True):
        source=np.where(seq<0,-1,(seq>0).astype(int)) if binary else seq
        base=2 if binary else 3
        for lags in CONTEXTS:
            wanted=[source[cutoff-lag] if cutoff>=lag else -1 for lag in lags]
            if min(wanted)<0: continue
            valid=observed>=max(lags)
            ticks=observed[valid]; targets=y[valid]
            codes=np.zeros(len(ticks),dtype=int); available=np.ones(len(ticks),dtype=bool); wanted_code=0
            for lag,value in zip(lags,wanted):
                lagged=source[ticks-lag]; available&=lagged>=0
                codes=codes*base+np.maximum(0,lagged); wanted_code=wanted_code*base+value
            templates.append((ticks[available],targets[available],codes[available],base**len(lags),wanted_code))
    features=[]
    for half,strength in CONFIGS:
        a=max(.01,prior*strength); b=max(.01,(1-prior)*strength)
        age=cutoff-1-observed; weights=np.ones(len(age)) if half is None else 2.**(-age/half)
        constant=(a+np.dot(weights,y))/(a+b+weights.sum())
        rates=[constant]; evidences=[0.]
        for ticks,targets,codes,size,wanted in templates:
            weights=np.ones(len(ticks)) if half is None else 2.**(-(cutoff-1-ticks)/half)
            wins=np.bincount(codes,weights=weights*targets,minlength=size)
            losses=np.bincount(codes,weights=weights*(1-targets),minlength=size)
            conditional=(a+wins[wanted])/(a+b+wins[wanted]+losses[wanted])
            evidence=np.sum(betaln(a+wins,b+losses)-betaln(a,b))-(betaln(a+wins.sum(),b+losses.sum())-betaln(a,b))
            rates.append(conditional); evidences.append(evidence)
        # Constant model has half the prior mass; split the remainder over experts.
        logprior=np.r_[np.log(.5),np.full(len(rates)-1,np.log(.5/max(1,len(rates)-1)))]
        for temperature in (1.,2.):
            posterior=softmax(np.asarray(evidences)/temperature+logprior)
            features.append(float(np.dot(posterior,rates)))
    return features


def feature_matrix(q,h,prior):
    indexed=index_history(q,h)
    return np.asarray([machine_features(query,indexed.get((query.episode_id,query.machine_id),[]),prior[(query.machine_type,query.signal)]) for query in q])


def compose(q,raw,features,column,weight):
    result={}
    for query,row in zip(q,features):
        p=np.asarray(raw[query.query_id]); win=weight*row[column]+(1-weight)*(1-p[0])
        positive=p[1:]/p[1:].sum()
        result[query.query_id]=tuple(np.r_[1-win,win*positive])
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='public'); p.add_argument('--out',default='Phillip_Lab/runs/v13_temporal_bayes')
    args=p.parse_args(); root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
    tq,th=load_inputs(Path(args.data)/'train'); prior=priors(tq,th)
    q,h=load_inputs(Path(args.data)/'validation'); F=feature_matrix(q,h,prior)
    raw=read_submission('Phillip_Lab/runs/v12_new_sys_confirmed/validation_predictions.csv')
    reference=read_submission(Path(args.data)/'validation/reference.csv'); targets=read_targets(Path(args.data)/'validation/targets.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(Path(args.data)/'validation/conditions.csv',('query_id','condition'))}
    tune,audit=episode_folds(targets,groups)
    def score(pred,eps): return subset_score(pred,targets,groups,reference,eps,60)
    best=score(raw,tune)['mean_loss_ratio']; choice={'weight':0.,'column':0}; winner=raw; comparison={}
    for col in range(F.shape[1]):
        for weight in (.1,.25,.5,1.):
            candidate=compose(q,raw,F,col,weight); result=score(candidate,tune)
            comparison[f'{col}_{weight}']=result
            if result['mean_loss_ratio']<best:
                best=result['mean_loss_ratio']; choice={'column':col,'weight':weight}; winner=candidate
        print('Feature',col,'best tune so far',60/(1+best),flush=True)
    np.savez(root/'features.npz',query_ids=np.asarray([x.query_id for x in q]),features=F)
    write_submission(root/'validation_predictions.csv',winner)
    report={'selection':choice,'tuning':score(winner,tune),'audit':score(winner,audit),'full':score(winner,tune|audit),
            'baseline':{fold:score(raw,eps) for fold,eps in [('tuning',tune),('audit',audit),('full',tune|audit)]},
            'comparison':comparison,'configs':CONFIGS,'note':'All priors from train history; local conditional experts use only each queried machine past. Selection on tuning only. Audit previously inspected.'}
    write_json(root/'report.json',report)
    testq,testh=load_inputs(Path(args.data)/'test'); TF=feature_matrix(testq,testh,prior)
    testraw=read_submission('Phillip_Lab/runs/v12_new_sys_confirmed/submission.csv')
    write_submission(root/'submission.csv',compose(testq,testraw,TF,choice['column'],choice['weight']))
    np.savez(root/'test_features.npz',query_ids=np.asarray([x.query_id for x in testq]),features=TF)
    print('FINAL',choice,'tune',report['tuning']['technical_points'],'audit',report['audit']['technical_points'],'full',report['full']['technical_points'])


if __name__=='__main__': main()
