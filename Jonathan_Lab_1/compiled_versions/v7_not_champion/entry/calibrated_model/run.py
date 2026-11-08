from importlib.util import find_spec
"""Select low-capacity calibration using episode-cross-fitted tuning predictions."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from slot_forecasting.io import read_targets,read_submission,read_rows,write_json,write_submission
from slot_forecasting.experiments import episode_folds,subset_score
from slot_forecasting.submission_contract import extract_archive,package_code


def transform(P,theta):
    return softmax(np.log(np.maximum(P,1e-12))*np.exp(theta[0])+theta[1:],axis=1)


def fit(P,Q,weights,penalty):
    def objective(theta):
        logp=np.log(np.maximum(P,1e-12))*np.exp(theta[0])
        pred=softmax(logp+theta[1:],axis=1)
        residual=2*weights[:,None]*(pred-Q)
        derivative=pred*(residual-(residual*pred).sum(1,keepdims=True))
        gradient=np.r_[np.sum(derivative*logp),derivative.sum(0)]+2*penalty*theta
        return np.sum(weights[:,None]*(pred-Q)**2)+penalty*np.sum(theta**2),gradient
    result=minimize(objective,np.zeros(6),jac=True,method='L-BFGS-B',bounds=[(-.7,.7)]+[(-1,1)]*5)
    if not result.success: raise RuntimeError(result.message)
    return result.x


def predict(root,data,out):
    # Reuse exact frozen champion features, parameters and blend.
    from lag_model.run import reproduce
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        csv=Path(tmp)/'raw.csv'; reproduce(root/'base-entry',data,csv)
        raw=read_submission(csv)
    config=json.loads((root/'calibration.json').read_text())
    ids=sorted(raw); values=transform(np.asarray([raw[k] for k in ids]),np.asarray(config['theta']))
    write_submission(out,{k:tuple(p) for k,p in zip(ids,values)})


def train(args):
    root=Path(args.out); data=Path(args.data); source=Path(args.source)
    if root.exists(): raise ValueError('Choose a fresh output folder')
    root.mkdir(parents=True)
    targets=read_targets(data/'validation/targets.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv',('query_id','condition'))}
    reference=read_submission(data/'validation/reference.csv')
    raw=read_submission(source/'validation_predictions.csv',{t.query_id for t in targets})
    tuning,audit=episode_folds(targets,groups)
    def score(p,eps): return subset_score(p,targets,groups,reference,eps,60)
    T=[t for t in targets if t.episode_id in tuning]
    P=np.asarray([raw[t.query_id] for t in T]); Q=np.asarray([t.probabilities for t in T])
    def weights(rows):
        w=[]
        for t in rows:
            condition=groups[t.query_id]
            ct=[r for r in rows if groups[r.query_id]==condition]
            episodes={r.episode_id for r in ct}
            counts={e:sum(r.episode_id==e for r in ct) for e in episodes}
            baseline=sum(np.sum((np.asarray(reference[r.query_id])-r.probabilities)**2)/counts[r.episode_id] for r in ct)/len(episodes)
            w.append(1/(4*len(episodes)*counts[t.episode_id]*max(1e-9,baseline)))
        return np.asarray(w)
    folds={}
    for condition in sorted(set(groups.values())):
        eps=sorted({t.episode_id for t in T if groups[t.query_id]==condition},key=lambda e:hashlib.sha256(e.encode()).hexdigest())
        folds.update({e:i%3 for i,e in enumerate(eps)})
    crossfit={}; best=score(raw,tuning)['mean_loss_ratio']; penalty=None
    for strength in (.01,.1,1.):
        values=np.zeros_like(P)
        for fold in range(3):
            train=np.asarray([folds[t.episode_id]!=fold for t in T]); hold=~train
            theta=fit(P[train],Q[train],weights([t for t,k in zip(T,train) if k]),strength)
            values[hold]=transform(P[hold],theta)
        preds={t.query_id:tuple(p) for t,p in zip(T,values)}
        result=score(preds,tuning); crossfit[str(strength)]=result
        print(f'Penalty {strength}: cross-fitted tuning {result["technical_points"]:.6f}',flush=True)
        if result['mean_loss_ratio']<best:
            best=result['mean_loss_ratio']; penalty=strength
    theta=np.zeros(6) if penalty is None else fit(P,Q,weights(T),penalty)
    ids=sorted(raw); values=transform(np.asarray([raw[k] for k in ids]),theta)
    calibrated={k:tuple(p) for k,p in zip(ids,values)}
    report={'penalty':penalty,'crossfit':crossfit,'folds':folds,
        'base':{name:score(raw,eps) for name,eps in [('tuning',tuning),('audit',audit),('full',tuning|audit)]},
        'calibrated':{name:score(calibrated,eps) for name,eps in [('tuning',tuning),('audit',audit),('full',tuning|audit)]},
        'note':'Penalty selected using 3-fold episode-cross-fitted tuning predictions. Final calibrator fitted only on tuning true probabilities. Audit previously inspected. Final tuning score is in-sample for calibration and must not be treated as held-out.'}
    # Paired condition-stratified episode bootstrap of audit normalized losses.
    episode_losses={}
    for ep in audit:
        rows=[t for t in targets if t.episode_id==ep]
        episode_losses[ep]=[np.mean([np.sum((np.asarray(pred[t.query_id])-t.probabilities)**2) for t in rows]) for pred in (raw,calibrated,reference)]
    rng=np.random.default_rng(42); gains=[]
    for _ in range(2000):
        ratios=[[],[]]
        for condition in sorted(set(groups.values())):
            eps=sorted({t.episode_id for t in targets if t.episode_id in audit and groups[t.query_id]==condition})
            sampled=rng.choice(eps,len(eps),replace=True)
            errors=np.asarray([episode_losses[e] for e in sampled]).mean(0)
            for i in range(2): ratios[i].append(errors[i]/max(1e-9,errors[2]))
        gains.append(60/(1+np.mean(ratios[1]))-60/(1+np.mean(ratios[0])))
    report['audit_gain_bootstrap_ci95']=np.quantile(gains,[.025,.975]).tolist()
    write_json(root/'report.json',report); write_json(root/'calibration.json',{'theta':theta.tolist(),'penalty':penalty})
    write_submission(root/'validation_predictions.csv',calibrated)
    archive=source/('entry.zip' if (source/'entry.zip').exists() else 'submission_entry.zip')
    extract_archive(archive.read_bytes(),root/'base-entry')
    predict(root,data,root/'submission.csv')
    entry=root/'entry'; entry.mkdir()
    shutil.copytree(root/'base-entry',entry/'base-entry')
    for package in ('calibrated_model','lag_model','hgb_upgrade','slot_forecasting'):
        dest=entry/package; dest.mkdir()
        for p in Path(find_spec(package).origin).parent.glob('*.py'): shutil.copy2(p,dest/p.name)
    for name in ('report.json','calibration.json'): shutil.copy2(root/name,entry/name)
    shutil.copy2(root/'base-entry/requirements.txt',entry/'requirements.txt')
    (entry/'README.md').write_text('# Cross-fitted probability calibration\nPython 3.11+. Prepare pinned requirements before offline prediction. Run python -m calibrated_model.run predict --model-dir . --data /data --out /output/submission.csv.\nBase model is frozen in base-entry. A six-parameter regularized temperature/bias map is fitted to released tuning true probabilities only, after penalty selection via episode cross-fitting. No test labels. Audit has been inspected previously. See report.json, including bootstrap uncertainty.\n',encoding='utf-8')
    write_json(entry/'reproduce.json',{'image':'python','command':['python','-m','calibrated_model.run','predict','--model-dir','.','--data','/data','--out','/output/submission.csv']})
    package_code(entry,root/'entry.zip')
    for name in ('base','calibrated'): print(name,{k:round(v['technical_points'],6) for k,v in report[name].items()})
    print('Selected penalty:',penalty,'Audit gain CI:',report['audit_gain_bootstrap_ci95'])


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['train','predict']); p.add_argument('--data',default='public'); p.add_argument('--out',required=True); p.add_argument('--source',default='Phillip_Lab/compiled_versions/v2_changing_lags'); p.add_argument('--model-dir',default='.')
    args=p.parse_args()
    if args.action=='train': train(args)
    else: predict(Path(args.model_dir),Path(args.data),Path(args.out))
