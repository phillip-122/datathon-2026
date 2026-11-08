from importlib.util import find_spec
"""Train binary win specialists, conditional payout model, and champion fallback."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from slot_forecasting.io import load_inputs,read_labels,read_targets,read_rows,read_submission,write_submission,write_json
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.experiments import episode_folds,subset_score
from slot_forecasting.submission_contract import extract_archive,package_code
from hgb_upgrade.run import training_rows,sequences,predict_model,mix
from hgb_upgrade.features import build_advanced_features


def matrix(qs,h,base):
    grouped=sequences(qs,h)
    return np.asarray([build_advanced_features(q,grouped[(q.episode_id,q.machine_id)],
        [s for (ep,m),s in grouped.items() if ep==q.episode_id and m!=q.machine_id],q.cutoff,base) for q in qs],dtype=np.float32)


def factorize(win,amount):
    win=np.clip(np.asarray(win),1e-7,1-1e-7)
    amount=np.maximum(np.asarray(amount),1e-12)
    amount=amount/amount.sum(axis=1,keepdims=True)
    return np.column_stack((1-win,win[:,None]*amount))


def predict_win(model,X):
    return model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else np.clip(model.predict(X),0,1)


def candidate(q,X,champion,win,amount,mode):
    original=np.asarray([champion[t.query_id] for t in q])
    conditional=original[:,1:] if mode=='champion_amount' else amount.predict_proba(X)
    p=factorize(predict_win(win,X),conditional)
    return {t.query_id:tuple(row) for t,row in zip(q,p)}


def predict(root,data,out):
    q,h=load_inputs(data/'test'); base=FrequencyForecaster.load(root/'base.json')
    original=predict_model(q,h,base,joblib.load(root/'champion.joblib'))
    config=json.loads((root/'selection.json').read_text())
    if config['weight']==0:
        result=original
    else:
        X=matrix(q,h,base)
        result=mix(candidate(q,X,original,joblib.load(root/'win.joblib'),joblib.load(root/'amount.joblib'),config['mode']),original,config['weight'])
    write_submission(out,result)


def train(args):
    root=Path(args.out); data=Path(args.data)
    if root.exists(): raise ValueError('Choose a fresh output directory')
    root.mkdir(parents=True)
    extract_archive(Path('Jonathan_Lab_1/compiled_versions/active_champion_submission/submission_entry.zip').read_bytes(),root/'source')
    artifacts=root/'source/artifacts'
    base=FrequencyForecaster.load(artifacts/'base_forecaster.json'); champion=joblib.load(artifacts/'models.joblib')
    q,h=load_inputs(data/'train'); labels=read_labels(data/'train/labels.csv',q)
    print('Building cutoff-safe training examples...',flush=True)
    X,y,_=training_rows(q,h,labels,base)
    vq,vh=load_inputs(data/'validation'); VX=matrix(vq,vh,base)
    original=predict_model(vq,vh,base,champion)
    targets=read_targets(data/'validation/targets.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv',('query_id','condition'))}
    ref=read_submission(data/'validation/reference.csv'); tuning,audit=episode_folds(targets,groups)
    def score(p,eps): return subset_score(p,targets,groups,ref,eps,60)
    print(f'Training conditional payout model on {sum(y>0)} winning examples...',flush=True)
    amount=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,learning_rate=.05,min_samples_leaf=30,l2_regularization=5,early_stopping=False,random_state=42)
    amount.fit(X[y>0],y[y>0]-1)
    comparison={}; best=score(original,tuning)['mean_loss_ratio']; selection={'model':'champion','mode':'champion_amount','weight':0.}; winner=None; prediction=original
    for kind,leaves in [('classifier',15),('classifier',31),('squared_loss',15),('squared_loss',31)]:
        name=f'{kind}_{leaves}'; print(f'Training win model: {name}...',flush=True)
        cls=HistGradientBoostingClassifier if kind=='classifier' else HistGradientBoostingRegressor
        model=cls(max_iter=200,max_leaf_nodes=leaves,learning_rate=.05,min_samples_leaf=30,l2_regularization=5,early_stopping=False,random_state=42)
        model.fit(X,(y>0).astype(int))
        joblib.dump(model,root/f'{name}.joblib',compress=3)
        for mode in ('champion_amount','separate_amount'):
            p=candidate(vq,VX,original,model,amount,mode)
            write_submission(root/f'validation_{name}_{mode}.csv',p)
            for weight in (.25,.5,.75,1.):
                combined=mix(p,original,weight); result=score(combined,tuning)
                comparison[f'{name}_{mode}_{weight}']=result
                if result['mean_loss_ratio']<best:
                    best=result['mean_loss_ratio']; winner=model; prediction=combined
                    selection={'model':name,'mode':mode,'weight':weight}
            print(f'  {mode}: standalone tuning {score(p,tuning)["technical_points"]:.3f}',flush=True)
    base.save(root/'base.json'); joblib.dump(champion,root/'champion.joblib',compress=3)
    joblib.dump(winner,root/'win.joblib',compress=3); joblib.dump(amount,root/'amount.joblib',compress=3)
    write_json(root/'selection.json',selection); write_submission(root/'validation_predictions.csv',prediction)
    report={'selection':selection,'comparison':comparison,'tuning':score(prediction,tuning),'audit':score(prediction,audit),'full':score(prediction,tuning|audit),
            'champion':{fold:score(original,eps) for fold,eps in [('tuning',tuning),('audit',audit),('full',tuning|audit)]},
            'settings':{'training_rows':len(y),'rounds':200,'learning_rate':.05,'l2':5,'seed':42,'early_stopping':False},
            'note':'Selection uses tuning only; audit previously inspected. All fitted new models use train histories and sampled labels only.'}
    write_json(root/'report.json',report)
    predict(root,data,root/'submission.csv')
    entry=root/'entry'; entry.mkdir()
    for package in ('win_model','hgb_upgrade','slot_forecasting'):
        dest=entry/package; dest.mkdir()
        for p in Path(find_spec(package).origin).parent.glob('*.py'): shutil.copy2(p,dest/p.name)
    for name in ('base.json','champion.joblib','win.joblib','amount.joblib','selection.json','report.json'): shutil.copy2(root/name,entry/name)
    (entry/'requirements.txt').write_text('numpy==2.3.4\nscipy==1.16.3\nscikit-learn==1.9.1\njoblib==1.5.2\nthreadpoolctl==3.8.0\nnarwhals==2.28.0\n',encoding='utf-8')
    (entry/'README.md').write_text('# Dedicated win-probability model\nPython 3.11+. Prepare pinned requirements before offline inference. Run python -m win_model.run predict --model-dir . --data /data --out /output/submission.csv.\nBinary win models use log loss or squared error; a four-class model learns payout size given a win. Tuning selects replacing champion win probability or full decomposition, with blend weights and champion fallback. Audit was previously inspected. Frozen features are based on Jonathan V3. See report.json.\n',encoding='utf-8')
    write_json(entry/'reproduce.json',{'image':'python','command':['python','-m','win_model.run','predict','--model-dir','.','--data','/data','--out','/output/submission.csv']})
    package_code(entry,root/'entry.zip')
    print('Selected:',selection,'Full:',report['full']['technical_points'],'Audit:',report['audit']['technical_points'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['train','predict']); p.add_argument('--data',default='public'); p.add_argument('--out',required=True); p.add_argument('--model-dir',default='.')
    a=p.parse_args()
    if a.action=='train': train(a)
    else: predict(Path(a.model_dir),Path(a.data),Path(a.out))
