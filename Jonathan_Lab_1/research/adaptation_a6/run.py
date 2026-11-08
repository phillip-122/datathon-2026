"""Select A6 additions to Jonathan A4 using tuning episodes and cross-fitting."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
from pathlib import Path
import sys,json,hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from experts import bank,CONFIGS,replace_win

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'adaptation/kit'))
from slot_forecasting.io import load_inputs,read_targets,read_submission,read_rows,write_json,write_submission
from slot_forecasting.experiments import episode_folds,subset_score


def main(a):
    root=Path(a.out); root.mkdir(parents=True,exist_ok=True); data=Path(a.data)
    q,h=load_inputs(data/'validation')
    rawfile=root/'a4_validation.csv'
    if not rawfile.exists():
        from slot_forecasting.submission_contract import extract_archive
        import subprocess
        base=root/'base'
        if not base.exists(): extract_archive((REPO/'Jonathan_Lab_1/compiled_versions/adaptation_active_submission/submission_entry.zip').read_bytes(),base)
        subprocess.run([sys.executable,str((base/'predict.py').resolve()),'--data',str((data/'validation').resolve()),'--out',str(rawfile.resolve())],check=True)
    raw=read_submission(rawfile)
    P=np.array([raw[x.query_id] for x in q]); targets=read_targets(data/'validation/targets.csv'); truth={t.query_id:t for t in targets}; Y=np.array([truth[x.query_id].probabilities for x in q]); groups={r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv',('query_id','condition'))}; ref=read_submission(data/'validation/reference.csv'); tune,audit=episode_folds(targets,groups)
    score=lambda A,eps:subset_score({x.query_id:tuple(p) for x,p in zip(q,A)},targets,groups,ref,eps,20)
    if (root/'features.npz').exists(): z=np.load(root/'features.npz'); B,M=z['B'],z['M']
    else: B,M=bank(q,h,P); np.savez(root/'features.npz',B=B,M=M,P=P)
    mask=np.array([x.episode_id in tune for x in q]); folds={}
    for c in sorted(set(groups.values())):
        eps=sorted({x.episode_id for x in q if x.episode_id in tune and groups[x.query_id]==c},key=lambda x:hashlib.sha256(x.encode()).hexdigest()); folds.update({e:i%3 for i,e in enumerate(eps)})
    F=np.array([folds.get(x.episode_id,-1) for x in q]); comparisons={}; best=0; choice=None; winner=None
    for col,config in enumerate(CONFIGS):
        for weight in (.25,.5,.75,1.):
            candidate=replace_win(P,B[:,col],weight); s=score(candidate,tune)['technical_points']; comparisons[f'expert_{col}_{weight}']=s
            if s>best: best=s; choice={'method':'expert','column':col,'config':config,'weight':weight}; winner=candidate
    print('BEST EXPERT',choice,best,flush=True)
    # Learn small smooth residual maps using only episode-cross-fitted tuning.
    X=np.column_stack((P,B,M)); residual=Y[:,0]-P[:,0]
    W=np.ones(len(q))
    for c in sorted(set(groups.values())):
        ii=np.array([groups[x.query_id]==c for x in q])&mask; denominator=np.mean([np.sum((np.asarray(ref[x.query_id])-Y[i])**2) for i,x in enumerate(q) if ii[i]]); W[[groups[x.query_id]==c for x in q]]=1/max(1e-9,denominator)
    W/=W[mask].mean()
    for family,param in [('ridge',10.),('ridge',100.),('ridge',1000.),('hgb',7),('hgb',15)]:
        oof=P.copy()
        for fold in range(3):
            tr=mask&(F!=fold); ho=mask&(F==fold)
            model=Ridge(alpha=param) if family=='ridge' else HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=param,min_samples_leaf=25,l2_regularization=10,learning_rate=.04,early_stopping=False,random_state=42)
            model.fit(X[tr],residual[tr],sample_weight=W[tr]); win=np.clip(1-P[ho,0]-model.predict(X[ho]),.001,.999); oof[ho]=replace_win(P[ho],win,1)
        for weight in (.5,1.):
            candidate=(1-weight)*P+weight*oof; s=score(candidate,tune)['technical_points']; comparisons[f'{family}_{param}_{weight}']=s; print(family,param,weight,s,flush=True)
            if s>best:
                best=s; choice={'method':family,'param':param,'weight':weight}; model=Ridge(alpha=param) if family=='ridge' else HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=param,min_samples_leaf=25,l2_regularization=10,learning_rate=.04,early_stopping=False,random_state=42); model.fit(X[mask],residual[mask],sample_weight=W[mask]); win=np.clip(1-P[:,0]-model.predict(X),.001,.999); winner=replace_win(P,win,weight)
                import joblib
                joblib.dump(model,root/'residual.joblib')
    write_submission(root/'validation_predictions.csv',{x.query_id:tuple(p) for x,p in zip(q,winner)})
    report={'selection':choice,'selection_tuning_points':best,'comparisons':comparisons,'full':score(winner,tune|audit),'audit':score(winner,audit),'tuning':score(winner,tune),'note':'Tuning-only expert/model selection. Supervised residual corrections use three episode folds. Audit already inspected in prior rounds.'}; write_json(root/'report.json',report); print('FINAL',choice,'full',report['full']['technical_points'],'audit',report['audit']['technical_points'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default=str(REPO/'adaptation/kit/public')); p.add_argument('--out',default=str(REPO/'Phillip_Lab/runs/adaptation_a6')); main(p.parse_args())
