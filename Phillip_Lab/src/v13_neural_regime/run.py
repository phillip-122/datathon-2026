"""Learn a neural sequence/context embedding on observed training payouts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import sys
import json
import numpy as np
from scipy.special import softmax
from slot_forecasting.io import load_inputs,read_labels,read_targets,read_rows,read_submission,write_submission,write_json
from slot_forecasting.schema import index_history,PAYOUTS
from slot_forecasting.experiments import episode_folds,subset_score

ROOT=Path('Phillip_Lab/runs/v13_neural_regime')
PAIRS=[(1,2),(1,4),(1,8),(2,4),(4,8)]


def cat(x): return None if x is None else min(x,2)


def feature(q,seq,tick,prior):
    seq={t:p for t,p in seq.items() if t<tick}
    x=[q.signal,*[float(q.machine_type==k) for k in ('steady','balanced','jackpot')],len(seq)/256,tick/256]
    for lag in range(1,17):
        v=seq.get(tick-lag)
        x += [float(v==p) for p in PAYOUTS]+[float(v is None)]
    for window in (8,16,32,64,128,256):
        values=[p for t,p in seq.items() if tick-window<=t]
        counts=np.bincount([PAYOUTS.index(p) for p in values],minlength=5)
        x += [len(values)/window]+((counts+4*prior)/(len(values)+4)).tolist()
    current=[(lag,) for lag in range(1,9)]+PAIRS
    items=list(seq.items())
    for lags in current:
        wanted=tuple(cat(seq.get(tick-l)) for l in lags)
        for window in (64,256):
            vals=[p for t,p in items if t>=tick-window and None not in wanted and tuple(cat(seq.get(t-l)) for l in lags)==wanted]
            counts=np.bincount([PAYOUTS.index(p) for p in vals],minlength=5)
            prob=(counts+8*prior)/(len(vals)+8)
            x += [min(len(vals)/64,4),float(None in wanted)]+prob.tolist()
    values=[p for t,p in items if t>=tick-64]
    counts=np.bincount([PAYOUTS.index(p) for p in values],minlength=5)
    base=(counts+20*prior)/(len(values)+20)
    x+=base.tolist()
    return x,base


def learn_priors(q,h,labels):
    indexed=index_history(q,h); counts=defaultdict(lambda:np.ones(5))
    for query in q:
        for row in indexed.get((query.episode_id,query.machine_id),[]): counts[f'{query.machine_type}|{query.signal}'][PAYOUTS.index(row.payout)]+=1
        counts[f'{query.machine_type}|{query.signal}'][PAYOUTS.index(labels[query.query_id])]+=1
    return {k:(v/v.sum()).tolist() for k,v in counts.items()}


def make_data(q,h,priors,labels=None):
    indexed=index_history(q,h); X=[]; B=[]; Y=[]
    for query in q:
        seq={r.tick:r.payout for r in indexed.get((query.episode_id,query.machine_id),[])}
        prior=np.asarray(priors[f'{query.machine_type}|{query.signal}'])
        times=[(query.cutoff,None)] if labels is None else [(t,p) for t,p in seq.items() if t>=32][::4]+[(query.cutoff,labels[query.query_id])]
        for tick,outcome in times:
            x,base=feature(query,seq,tick,prior); X.append(x); B.append(base)
            if labels is not None:Y.append(PAYOUTS.index(outcome))
    return np.asarray(X,dtype=np.float32),np.asarray(B,dtype=np.float32),np.asarray(Y,dtype=int)


def forward(X,B,params):
    h1=np.tanh(X@params[0]+params[1]); h2=np.tanh(h1@params[2]+params[3])
    p=softmax(np.log(B)+h2@params[4]+params[5],axis=1)
    return h1,h2,p


def train_network(X,B,y):
    rng=np.random.default_rng(42); d=X.shape[1]
    params=[rng.normal(0,.05,(d,64)).astype('float32'),np.zeros(64,dtype='float32'),rng.normal(0,.1,(64,32)).astype('float32'),np.zeros(32,dtype='float32'),np.zeros((32,5),dtype='float32'),np.zeros(5,dtype='float32')]
    moments=[np.zeros_like(p) for p in params]; squares=[np.zeros_like(p) for p in params]; step=0
    for epoch in range(80):
        for ids in np.array_split(rng.permutation(len(y)),max(1,len(y)//512)):
            a=X[ids]; h1,h2,p=forward(a,B[ids],params)
            p[np.arange(len(ids)),y[ids]]-=1; p/=len(ids)
            d2=(p@params[4].T)*(1-h2*h2)
            d1=(d2@params[2].T)*(1-h1*h1)
            grads=[a.T@d1+.001*params[0],d1.sum(0),h1.T@d2+.001*params[2],d2.sum(0),h2.T@p+.001*params[4],p.sum(0)]
            step+=1
            for i,g in enumerate(grads):
                moments[i]=.9*moments[i]+.1*g; squares[i]=.999*squares[i]+.001*g*g
                params[i]-=.001*(moments[i]/(1-.9**step))/(np.sqrt(squares[i]/(1-.999**step))+1e-8)
        if (epoch+1) in (20,40,80):
            print('Neural epoch',epoch+1,flush=True)
            yield epoch+1,[p.copy() for p in params]


def main(args):
    ROOT.mkdir(parents=True,exist_ok=True)
    data=Path(args.data); q,h=load_inputs(data/'train'); labels=read_labels(data/'train/labels.csv',q)
    priors=learn_priors(q,h,labels)
    cache=ROOT/'features.npz'
    if cache.exists():
        z=np.load(cache); X,B,y=z['X'],z['B'],z['y']
    else:
        print('Extracting machine-context training examples...',flush=True)
        X,B,y=make_data(q,h,priors,labels); np.savez(cache,X=X,B=B,y=y)
    mean=X.mean(0); scale=np.maximum(X.std(0),.1); X=(X-mean)/scale
    vq,vh=load_inputs(data/'validation'); VX,VB,_=make_data(vq,vh,priors); VX=(VX-mean)/scale
    targets=read_targets(data/'validation/targets.csv'); groups={r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv',('query_id','condition'))}; ref=read_submission(data/'validation/reference.csv'); tune,audit=episode_folds(targets,groups)
    base=read_submission('Phillip_Lab/compiled_versions/v12_new_sys_frozen/validation_predictions.csv')
    def score(p,eps):return subset_score(p,targets,groups,ref,eps,60)
    ids=[x.query_id for x in vq]; base_array=np.asarray([base[k] for k in ids]); report={'comparison':{},'note':'Train-only observed-outcome neural fitting. Tuning selects checkpoints/blends; previously inspected audit scored after selection.'}; best=score(base,tune)['mean_loss_ratio']; selected={'method':'base','weight':0,'epoch':0}; selected_params=None
    for epoch,params in train_network(X,B,y):
        _,embedding,p=forward(VX,VB,params)
        np.savez(ROOT/f'network_{epoch}.npz',mean=mean,scale=scale,**{f'p{i}':p for i,p in enumerate(params)})
        write_submission(ROOT/f'validation_neural_{epoch}.csv',{k:tuple(row) for k,row in zip(ids,p)})
        for w in (.25,.5,.75,1):
            mixture=w*p+(1-w)*base_array; pred={k:tuple(row) for k,row in zip(ids,mixture)}; res=score(pred,tune); report['comparison'][f'epoch{epoch}_w{w}']=res
            print(epoch,w,res['technical_points'],flush=True)
            if res['mean_loss_ratio']<best:
                best=res['mean_loss_ratio']; selected={'method':'neural_blend','weight':w,'epoch':epoch}; selected_params=params
    # A low-dimensional learned context embedding may explain remaining probability
    # residuals. Its ridge map is selected by three held-out episode folds in tuning.
    final_params=params
    _,embedding,neural=forward(VX,VB,final_params)
    Z=np.column_stack((np.ones(len(vq)),base_array,embedding))
    target_map={t.query_id:t for t in targets}; truth=np.asarray([target_map[k].probabilities for k in ids]); trainmask=np.array([target_map[k].episode_id in tune for k in ids])
    folds={}
    for c in sorted(set(groups.values())):
        eps=sorted({t.episode_id for t in targets if t.episode_id in tune and groups[t.query_id]==c},key=lambda x:hashlib.sha256(x.encode()).hexdigest()); folds.update({e:i%3 for i,e in enumerate(eps)})
    ridge_best=None
    for penalty in (10.,100.,1000.):
        oof=base_array.copy()
        for fold in range(3):
            tr=np.array([target_map[k].episode_id in tune and folds.get(target_map[k].episode_id)!=fold for k in ids]); hold=np.array([target_map[k].episode_id in tune and folds.get(target_map[k].episode_id)==fold for k in ids])
            W=np.linalg.solve(Z[tr].T@Z[tr]+penalty*np.eye(Z.shape[1]),Z[tr].T@(truth[tr]-base_array[tr]))
            values=np.maximum(base_array[hold]+Z[hold]@W,1e-7); oof[hold]=values/values.sum(1,keepdims=True)
        pred={k:tuple(row) for k,row in zip(ids,oof)}; res=score(pred,tune); report['comparison'][f'context_ridge_{penalty}']=res
        print('context ridge',penalty,res['technical_points'],flush=True)
        if res['mean_loss_ratio']<best:
            best=res['mean_loss_ratio']; selected={'method':'context_ridge','penalty':penalty,'epoch':80}; selected_params=final_params; ridge_best=penalty
    if selected['method']=='base': pred=base
    elif selected['method']=='neural_blend':
        _,_,p=forward(VX,VB,selected_params); p=selected['weight']*p+(1-selected['weight'])*base_array; pred={k:tuple(row) for k,row in zip(ids,p)}
    else:
        penalty=selected['penalty']; W=np.linalg.solve(Z[trainmask].T@Z[trainmask]+penalty*np.eye(Z.shape[1]),Z[trainmask].T@(truth[trainmask]-base_array[trainmask])); p=np.maximum(base_array+Z@W,1e-7); p/=p.sum(1,keepdims=True); pred={k:tuple(row) for k,row in zip(ids,p)}; np.save(ROOT/'ridge.npy',W)
    report.update(selection=selected,training_rows=len(y),tuning=score(pred,tune),audit=score(pred,audit),full=score(pred,tune|audit),base_full=score(base,tune|audit),base_audit=score(base,audit))
    write_submission(ROOT/'validation_predictions.csv',pred)
    write_json(ROOT/'report.json',report); write_json(ROOT/'priors.json',priors)
    tq,th=load_inputs(data/'test'); TX,TB,_=make_data(tq,th,priors); TX=(TX-mean)/scale; base_test=read_submission('Phillip_Lab/compiled_versions/v12_new_sys_frozen/submission.csv'); bt=np.asarray([base_test[t.query_id] for t in tq])
    if selected['method']=='base': out=bt
    else:
        _,emb,p=forward(TX,TB,selected_params)
        if selected['method']=='neural_blend':out=selected['weight']*p+(1-selected['weight'])*bt
        else:
            zz=np.column_stack((np.ones(len(tq)),bt,emb)); out=np.maximum(bt+zz@W,1e-7); out/=out.sum(1,keepdims=True)
    write_submission(ROOT/'submission.csv',{t.query_id:tuple(row) for t,row in zip(tq,out)})
    print('SELECTED',selected,'Full',report['full']['technical_points'],'Audit',report['audit']['technical_points'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='public'); main(p.parse_args())
