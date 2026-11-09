"""A5: episode-cross-fitted residual correction of Jonathan's frozen A4."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from scipy.special import expit
from scipy.special import betaln,softmax
from itertools import combinations


def baseline(root,data,out):
    subprocess.run([sys.executable,str((root/'base/predict.py').resolve()),'--data',str(data.resolve()),'--out',str(out.resolve())],check=True)


def features(q,h,raw):
    seq=defaultdict(dict)
    for r in h: seq[r.episode_id,r.machine_id][r.tick]=r.payout
    X=[]
    for query in q:
        p=np.asarray(raw[query.query_id]); s={t:v for t,v in seq[query.episode_id,query.machine_id].items() if t<query.cutoff}
        peers=[v for (e,m),v in seq.items() if e==query.episode_id and m!=query.machine_id]
        row=[1.,*p,*p**2,float(len(s)<=64),query.signal]
        for half in (4,12,32,96,100000):
            c=np.ones(3)*.5
            for t,v in s.items(): c[min(v,2)]+=2.**(-(query.cutoff-1-t)/half)
            row.extend(c/c.sum()); row.append(c[1]/(c[1]+c[2]))
        for window in (4,16,40):
            v=[v for ps in peers for t,v in ps.items() if query.cutoff-window<=t<query.cutoff]
            row.extend([sum(x>0 for x in v)/max(1,len(v)),sum(x==1 for x in v)/max(1,len(v))])
        for lag in range(1,9):
            wanted=s.get(query.cutoff-lag,-1)
            row.extend([float(wanted==0),float(wanted==1),float(wanted>=2)])
            c=np.array([2*p[0],2*(1-p[0])])
            for t,v in s.items():
                if wanted>=0 and s.get(t-lag,-2)==wanted: c[int(v>0)]+=2.**(-(query.cutoff-1-t)/48)
            row.append(c[1]/c.sum())
        row=np.asarray(row)
        X.append(np.r_[*[row*(query.machine_type==kind) for kind in ('steady','balanced','jackpot')]])
    return np.asarray(X)


def fit(X,Y,penalty,weights):
    mean=X.mean(0); scale=np.maximum(X.std(0),.05); Z=(X-mean)/scale
    Z=np.column_stack((np.ones(len(X)),Z)); reg=np.eye(Z.shape[1])*penalty; reg[0,0]=penalty*.01
    coef=np.linalg.solve(Z.T@(Z*weights[:,None])+reg,Z.T@(Y*weights[:,None]))
    return mean,scale,coef


def corrected(X,P,model):
    mean,scale,coef=model
    result=np.maximum(P+np.column_stack((np.ones(len(X)),(X-mean)/scale))@coef,1e-9)
    return result/result.sum(1,keepdims=True)


def structural(q,h,P,targets,half):
    truth={t.query_id:t.probabilities for t in targets}; constants={}
    for kind in ('steady','balanced','jackpot'):
        a=np.array([truth[x.query_id] for x in q if x.query_id in truth and x.machine_type==kind])
        c=(a[:,2:]/a[:,2:].sum(1,keepdims=True)).mean(0)
        share=a[:,1]/(1-a[:,0]); mid=(share.min()+share.max())/2
        constants[kind]={'c':c.tolist(),'low':float(share[share<=mid].mean()),'high':float(share[share>mid].mean())}
    return constants


def project(q,h,P,constants,half):
    grouped=defaultdict(list)
    for r in h: grouped[r.episode_id,r.machine_id].append(r)
    result=[]
    for x,p in zip(q,P):
        info=constants[x.machine_type]; lo,hi=info['low'],info['high']; logit=0.
        for r in grouped[x.episode_id,x.machine_id]:
            if r.tick>=x.cutoff or not r.payout: continue
            w=1. if half==0 else 2.**(-(x.cutoff-1-r.tick)/half)
            logit+=w*(np.log(hi/lo) if r.payout==1 else np.log((1-hi)/(1-lo)))
        share=lo+(hi-lo)*expit(logit)
        direction=np.r_[-1.,share,(1-share)*np.array(info['c'])]
        win=np.clip(np.dot(p-np.array([1.,0,0,0,0]),direction)/np.dot(direction,direction),0,1)
        result.append(np.array([1.,0,0,0,0])+win*direction)
    return np.asarray(result)


def local_rates(q,h,P,half,strength,temporal):
    grouped=defaultdict(dict)
    for r in h: grouped[r.episode_id,r.machine_id][r.tick]=r.payout
    result=[]
    for query,p in zip(q,P):
        seq=grouped[query.episode_id,query.machine_id]
        ticks=np.asarray(sorted(t for t in seq if t<query.cutoff))
        y=np.asarray([seq[t]>0 for t in ticks],dtype=float)
        prior=1-p[0]; a=max(.01,strength*prior); b=max(.01,strength*(1-prior))
        weights=np.ones(len(ticks)) if half==0 else 2.**(-(query.cutoff-1-ticks)/half)
        rates=[(a+np.dot(weights,y))/(a+b+weights.sum())]; evidence=[0.]
        if temporal:
            for lags in [(i,) for i in range(1,9)]+list(combinations(range(1,9),2)):
                wanted=[seq.get(query.cutoff-lag,-1) for lag in lags]
                if min(wanted)<0: continue
                wanted=[int(v>0) for v in wanted]; codes=[]; ys=[]; ts=[]
                for t,target in zip(ticks,y):
                    lagged=[seq.get(t-lag,-1) for lag in lags]
                    if min(lagged)<0: continue
                    code=0
                    for v in lagged: code=code*2+int(v>0)
                    codes.append(code); ys.append(target); ts.append(t)
                ww=np.ones(len(ts)) if half==0 else 2.**(-(query.cutoff-1-np.asarray(ts))/half)
                wins=np.bincount(codes,weights=ww*np.asarray(ys),minlength=2**len(lags)); losses=np.bincount(codes,weights=ww*(1-np.asarray(ys)),minlength=2**len(lags))
                code=0
                for v in wanted: code=code*2+v
                rates.append((a+wins[code])/(a+b+wins[code]+losses[code]))
                evidence.append(np.sum(betaln(a+wins,b+losses)-betaln(a,b))-(betaln(a+wins.sum(),b+losses.sum())-betaln(a,b)))
            logprior=np.r_[np.log(.5),np.full(len(rates)-1,np.log(.5/max(1,len(rates)-1)))]
            result.append(float(np.dot(softmax(np.asarray(evidence)/2+logprior),rates)))
        else: result.append(rates[0])
    return np.asarray(result)


def replace_win(P,rates,weight):
    win=(1-weight)*(1-P[:,0])+weight*rates
    return np.column_stack((1-win,P[:,1:]/P[:,1:].sum(1,keepdims=True)*win[:,None]))


def predict(root,data,out):
    from slot_forecasting.io import load_inputs,read_submission,write_submission
    with tempfile.TemporaryDirectory() as tmp:
        csv=Path(tmp)/'raw.csv'; baseline(root,data,csv)
        q,h=load_inputs(data/'test' if (data/'test').is_dir() else data)
        raw=read_submission(csv); P=np.asarray([raw[x.query_id] for x in q])
    config=json.loads((root/'config.json').read_text())
    if config['method']=='ridge':
        z=np.load(root/'residual.npz'); P=corrected(features(q,h,raw),P,(z['mean'],z['scale'],z['coef']))
    if config['method']=='projection':
        candidate=project(q,h,P,config['constants'],config['half'])
        P=(1-config['weight'])*P+config['weight']*candidate
    if config['method']=='local':
        P=replace_win(P,local_rates(q,h,P,config['half'],config['strength'],config['temporal']),config['weight'])
    write_submission(out,{x.query_id:tuple(p) for x,p in zip(q,P)})


def train(args):
    from slot_forecasting.io import load_inputs,read_targets,read_rows,read_submission,write_submission,write_json
    from slot_forecasting.experiments import episode_folds,subset_score
    from slot_forecasting.submission_contract import extract_archive,package_code
    data=Path(args.data).resolve(); root=Path(args.out).resolve(); root.mkdir(parents=True,exist_ok=False)
    meta=json.loads((data/'metadata.json').read_text()); assert meta['round']=='adaptation'
    extract_archive(Path(args.archive).read_bytes(),root/'base')
    baseline(root,data/'validation',root/'a4_validation.csv')
    q,h=load_inputs(data/'validation'); raw=read_submission(root/'a4_validation.csv'); P=np.asarray([raw[x.query_id] for x in q])
    targets=read_targets(data/'validation/targets.csv'); truth={t.query_id:t for t in targets}; Y=np.asarray([truth[x.query_id].probabilities for x in q])
    groups={r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv',('query_id','condition'))}
    ref=read_submission(data/'validation/reference.csv'); tune,audit=episode_folds(targets,groups)
    score=lambda A,eps:subset_score({x.query_id:tuple(p) for x,p in zip(q,A)},targets,groups,ref,eps,meta['max_points'])
    mask=np.array([x.episode_id in tune for x in q]); folds={}
    for c in sorted(set(groups.values())):
        eps=sorted({x.episode_id for x in q if x.episode_id in tune and groups[x.query_id]==c},key=lambda e:hashlib.sha256(e.encode()).hexdigest())
        folds.update({e:i%3 for i,e in enumerate(eps)})
    X=features(q,h,raw); baseline_score=score(P,tune); best=baseline_score['mean_loss_ratio']; choice={'method':'a4'}; winner=P; comparisons={}
    weights=np.ones(len(q)); tuning_targets=[t for t in targets if t.episode_id in tune]
    for penalty in (10.,100.,1000.,10000.):
        oof=P.copy()
        for fold in range(3):
            tr=np.array([x.episode_id in tune and folds[x.episode_id]!=fold for x in q]); ho=np.array([x.episode_id in tune and folds[x.episode_id]==fold for x in q])
            model=fit(X[tr],(Y-P)[tr],penalty,weights[tr]); oof[ho]=corrected(X[ho],P[ho],model)
        s=score(oof,tune); comparisons[f'ridge_{penalty}']=s; print('RIDGE',penalty,s['technical_points'],flush=True)
        if s['mean_loss_ratio']<best:
            best=s['mean_loss_ratio']; choice={'method':'ridge','penalty':penalty}; selected=fit(X[mask],(Y-P)[mask],penalty,weights[mask]); winner=corrected(X,P,selected)
    for half in (0,24,64):
        oof=P.copy()
        for fold in range(3):
            training=[t for t in tuning_targets if folds[t.episode_id]!=fold]; held=[i for i,x in enumerate(q) if x.episode_id in tune and folds[x.episode_id]==fold]
            constants=structural(q,h,P,training,half); oof[held]=project([q[i] for i in held],h,P[held],constants,half)
        for weight in (.5,1.):
            s=score((1-weight)*P+weight*oof,tune); comparisons[f'projection_{half}_{weight}']=s; print('PROJECTION',half,weight,s['technical_points'],flush=True)
            if s['mean_loss_ratio']<best:
                best=s['mean_loss_ratio']; constants=structural(q,h,P,tuning_targets,half); choice={'method':'projection','half':half,'weight':weight,'constants':constants}; winner=(1-weight)*P+weight*project(q,h,P,constants,half)
    for half,strength,temporal in [(0,16,False),(64,16,False),(24,16,False),(0,4,True),(64,4,True),(24,4,True)]:
        rates=local_rates(q,h,P,half,strength,temporal)
        for weight in (.25,.5,.75):
            candidate=replace_win(P,rates,weight); s=score(candidate,tune)
            comparisons[f'local_{half}_{strength}_{temporal}_{weight}']=s
            print('LOCAL',half,strength,temporal,weight,s['technical_points'],flush=True)
            if s['mean_loss_ratio']<best:
                best=s['mean_loss_ratio']; choice={'method':'local','half':half,'strength':strength,'temporal':temporal,'weight':weight}; winner=candidate
    if choice['method']=='ridge': np.savez(root/'residual.npz',mean=selected[0],scale=selected[1],coef=selected[2])
    write_json(root/'config.json',choice)
    write_submission(root/'validation_predictions.csv',{x.query_id:tuple(p) for x,p in zip(q,winner)})
    report={'selection':choice,'crossfitted_tuning':comparisons,'baseline':{name:score(P,eps) for name,eps in [('full',tune|audit),('audit',audit),('tuning',tune)]},'selected':{name:score(winner,eps) for name,eps in [('full',tune|audit),('audit',audit),('tuning_fit',tune)]},'crossfitted_selection_points':meta['max_points']/(1+best),'tuning_episodes':sorted(tune),'audit_episodes':sorted(audit),'base_archive_sha256':hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),'note':'Frozen Jonathan A4 is the baseline. Three condition-balanced episode folds select correction using tuning only; audit does not fit or select this update. A4 itself was previously tuned on this public validation. Full points include fitting episodes; hidden test score unknown.'}
    write_json(root/'report.json',report)
    predict(root,data,root/'submission.csv')
    entry=root/'entry'; entry.mkdir(); shutil.copytree(root/'base',entry/'base')
    shutil.copy2(__file__,entry/'predict.py'); shutil.copytree(Path(__import__('slot_forecasting').__file__).parent,entry/'slot_forecasting',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('config.json','residual.npz'):
        if (root/name).exists(): shutil.copy2(root/name,entry/name)
    import importlib.metadata
    (entry/'requirements.txt').write_text(''.join(f'{name}=={importlib.metadata.version(name)}\n' for name in ('numpy','scipy','scikit-learn','joblib','threadpoolctl','narwhals')))
    (entry/'README.md').write_text('# Adaptation A5 from Jonathan A4\nRun: python predict.py --data /data --out /output/submission.csv\nInstall pinned requirements before offline inference. No training or validation targets are loaded during prediction. Fixed A4 artifacts plus a cross-fitted adaptation correction; selection and limitations in report.json.\n')
    shutil.copy2(root/'report.json',entry/'report.json'); write_json(entry/'reproduce.json',{'image':'python','command':['python','predict.py','--data','/data','--out','/output/submission.csv']})
    package_code(entry,root/'submission_entry.zip')
    print('FINAL',choice,'audit',report['selected']['audit']['technical_points'],'full',report['selected']['full']['technical_points'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--train',action='store_true'); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--archive')
    a=p.parse_args()
    if a.train:
        sys.path.insert(0,str(Path(a.data).resolve().parent)); train(a)
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parent)); predict(Path(__file__).resolve().parent,Path(a.data),Path(a.out))
