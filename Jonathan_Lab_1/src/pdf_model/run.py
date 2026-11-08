from importlib.util import find_spec
"""Controlled recent-history and missing-history experiments; champion fallback."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import json
import joblib
import numpy as np
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs,read_labels,read_targets,read_rows,read_submission,write_submission,write_json
from slot_forecasting.experiments import episode_folds,subset_score
from slot_forecasting.submission_contract import package_code,extract_archive
from .features import build_advanced_features,PAYOUT_TO_IDX
from hgb_upgrade.run import predict_model as predict_champion


def sequences(queries,history):
    grouped=defaultdict(dict)
    for r in history:
        grouped[(r.episode_id,r.machine_id)][r.tick]=r.payout
    return grouped


def masked(seq,tick,rng):
    # Same sampling budget as documented short histories: 40 recent + 24 older.
    recent={t:p for t,p in seq.items() if tick-40<=t<tick}
    old=sorted(t for t in seq if t<tick-40)
    keep=rng.choice(old,min(24,len(old)),replace=False) if old else []
    return {**recent,**{int(t):seq[t] for t in keep}}


def training_rows(qs,history,labels,base,augment=False):
    grouped=sequences(qs,history); X=[]; y=[]; ages=[]; rng=np.random.default_rng(42)
    for q in qs:
        seq=grouped[(q.episode_id,q.machine_id)]
        peers=[s for (ep,m),s in grouped.items() if ep==q.episode_id and m!=q.machine_id]
        ticks=[t for t in range(30,q.cutoff,6) if t in seq]+[q.cutoff]
        for t in ticks:
            past={k:v for k,v in seq.items() if k<t}
            if len(past)<8: continue
            earlier=[{k:v for k,v in s.items() if k<t} for s in peers]
            outcome=labels[q.query_id] if t==q.cutoff else seq[t]
            X.append(build_advanced_features(q,past,earlier,t,base)); y.append(PAYOUT_TO_IDX[outcome]); ages.append(q.cutoff-t)
            if augment and len(past)>64:
                X.append(build_advanced_features(q,masked(past,t,rng),[masked(s,t,rng) for s in earlier],t,base))
                y.append(PAYOUT_TO_IDX[outcome]); ages.append(q.cutoff-t)
    return np.asarray(X,dtype=np.float32),np.asarray(y),np.asarray(ages)


def predict_model(qs,h,base,models):
    grouped=sequences(qs,h)
    X=np.asarray([build_advanced_features(q,grouped[(q.episode_id,q.machine_id)],
        [s for (ep,m),s in grouped.items() if ep==q.episode_id and m!=q.machine_id],q.cutoff,base) for q in qs],dtype=np.float32)
    logits=np.mean([m.decision_function(X) for m in models],axis=0)
    temp=np.asarray([{'steady':1.,'balanced':1.06,'jackpot':1.12}[q.machine_type] for q in qs])
    probs=softmax(logits/temp[:,None],axis=1)
    return {q.query_id:tuple(p/p.sum()) for q,p in zip(qs,probs)}


def mix(a,b,w):
    return {k:tuple(w*np.asarray(a[k])+(1-w)*np.asarray(b[k])) for k in a}


def reproduce(root,data,out):
    q,h=load_inputs(data/'test')
    base=FrequencyForecaster.load(root/'base.json')
    config=json.loads((root/'selection.json').read_text())
    champion=joblib.load(root/'champion.joblib')
    a=predict_champion(q,h,base,champion)
    selected=joblib.load(root/'selected.joblib')
    b=a if config['new_weight']==0 else predict_model(q,h,base,selected)
    write_submission(out,mix(b,a,config['new_weight']))


def train(args):
    root=Path(args.out); data=Path(args.data)
    if root.exists(): raise ValueError('Choose a fresh output folder')
    root.mkdir(parents=True)
    q,h=load_inputs(data/'train'); labels=read_labels(data/'train/labels.csv',q)
    # Champion feature priors must be identical when loading its saved trees.
    champion_entry=root/'champion-source'
    extract_archive(Path('Jonathan_Lab_1/compiled_versions/active_champion_submission/submission_entry.zip').read_bytes(),champion_entry)
    champion_path=champion_entry/'artifacts'
    base=FrequencyForecaster.load(champion_path/'base_forecaster.json')
    champion=joblib.load(champion_path/'models.joblib')
    vq,vh=load_inputs(data/'validation')
    champion_prediction=predict_champion(vq,vh,base,champion)
    targets=read_targets(data/'validation/targets.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv',('query_id','condition'))}
    ref=read_submission(data/'validation/reference.csv'); tuning,audit=episode_folds(targets,groups)
    def score(p,eps): return subset_score(p,targets,groups,ref,eps,60)
    best=score(champion_prediction,tuning)['mean_loss_ratio']; chosen={'variant':'champion','new_weight':0.}
    winner=champion; prediction=champion_prediction; comparison={}
    for name,augment,half in [('structure_peers',False,None)]:
        print(f'Building {name} training examples...',flush=True)
        X,y,ages=training_rows(q,h,labels,base,augment)
        weights=np.ones(len(y)) if half is None else 2.**(-ages/half)
        models=[]
        for leaves,l2,seed in [(40,4.,42),(60,2.,101),(80,1.5,202)]:
            print(f'Training {name}: {len(y)} rows, {leaves} leaves...',flush=True)
            m=HistGradientBoostingClassifier(max_iter=150,learning_rate=.08,max_leaf_nodes=leaves,
                min_samples_leaf=20,l2_regularization=l2,random_state=seed,early_stopping=False)
            m.fit(X,y,sample_weight=weights); models.append(m)
        p=predict_model(vq,vh,base,models)
        for weight in (.25,.5,.75,1.):
            combined=mix(p,champion_prediction,weight); result=score(combined,tuning)
            comparison[f'{name}_{weight}']=result
            print(f'{name} weight {weight}: tuning {result["technical_points"]:.3f}',flush=True)
            if result['mean_loss_ratio']<best:
                best=result['mean_loss_ratio']; chosen={'variant':name,'new_weight':weight}; winner=models; prediction=combined
        joblib.dump(models,root/f'{name}.joblib',compress=3)
    base.save(root/'base.json'); joblib.dump(champion,root/'champion.joblib',compress=3)
    joblib.dump(winner,root/'selected.joblib',compress=3); write_json(root/'selection.json',chosen)
    write_submission(root/'validation_predictions.csv',prediction)
    report={'selection':chosen,'comparison':comparison,'tuning':score(prediction,tuning),'audit':score(prediction,audit),
            'full':score(prediction,tuning|audit),'champion_full':score(champion_prediction,tuning|audit),
            'note':'Only tuning selects models. Audit previously inspected. Fixed 150 rounds; no random machine-level early stopping.'}
    write_json(root/'report.json',report)
    reproduce(root,data,root/'submission.csv'); reproduce(root,data,root/'reproduced.csv')
    a,b=read_submission(root/'submission.csv'),read_submission(root/'reproduced.csv')
    assert max(abs(x-y) for k in a for x,y in zip(a[k],b[k]))<1e-8
    entry=root/'entry'; entry.mkdir()
    for package in ('pdf_model','lag_model','hgb_upgrade','slot_forecasting'):
        dest=entry/package; dest.mkdir()
        for file in Path(find_spec(package).origin).parent.glob('*.py'): shutil.copy2(file,dest/file.name)
    for file in ('base.json','champion.joblib','selected.joblib','selection.json','report.json'): shutil.copy2(root/file,entry/file)
    (entry/'requirements.txt').write_text('numpy==2.3.4\nscipy==1.16.3\nscikit-learn==1.9.1\njoblib==1.5.2\nthreadpoolctl==3.8.0\nnarwhals==2.28.0\n',encoding='utf-8')
    (entry/'README.md').write_text('# HGB upgrade experiment\nPython 3.11+; install pinned requirements before offline run. Run python -m pdf_model.run predict --model-dir . --data /data --out /output/submission.csv.\nModel selection uses tuning episodes. Includes champion fallback and features comparing recent versus older per-machine lag relationships. See report.json.\n',encoding='utf-8')
    write_json(entry/'reproduce.json',{'image':'python','command':['python','-m','pdf_model.run','predict','--model-dir','.','--data','/data','--out','/output/submission.csv']})
    package_code(entry,root/'entry.zip')
    print('Selected:',chosen,'Full:',report['full']['technical_points'],'Audit:',report['audit']['technical_points'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['train','predict']); p.add_argument('--data',default='public'); p.add_argument('--out',required=True); p.add_argument('--model-dir',default='.')
    args=p.parse_args()
    if args.action=='train': train(args)
    else: reproduce(Path(args.model_dir),Path(args.data),Path(args.out))
