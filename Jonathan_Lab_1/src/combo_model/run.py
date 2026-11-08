from importlib.util import find_spec
"""Train and reproduce the House Edge ensemble using pinned NumPy/SciPy/XGBoost."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import json
import hashlib
import shutil
from pathlib import Path
import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from slot_forecasting.io import load_inputs,read_labels,read_targets,read_rows,read_submission,write_json,write_submission
from slot_forecasting.experiments import episode_folds,subset_score
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.rounds import round_metadata
from slot_forecasting.submission_contract import package_code
from .frozen_features import FEATURE_NAMES,HistoryFeatures
from .models import examples,train_shared,local_predictions,neural_predictions

SPECS = [('local_4_64',4,64,False),('local_16_256',16,256,False),
         ('hierarchical_4_64',4,64,True),('hierarchical_16_256',16,256,True)]


def components(q,h,root,linear,params):
    data = examples(q,h)
    result = {'neural':neural_predictions(data,params)}
    for name,strength,half,hier in SPECS:
        print(f'Predicting {name} ({len(q)} machines)...',flush=True)
        result[name] = local_predictions(data,linear,strength,half,hier)
    tree = xgb.Booster(params={'nthread':2})
    tree.load_model(root/'model.ubj')
    X = HistoryFeatures(q,h).matrix(q)
    p = tree.predict(xgb.DMatrix(X,feature_names=FEATURE_NAMES,nthread=2)).astype(float)
    p /= p.sum(1,keepdims=True)
    result['xgboost'] = {query.query_id:v for query,v in zip(q,p)}
    result['baseline'] = FrequencyForecaster.load(root/'baseline.json').predict(q,h)
    return result


def mix(parts,names,weights):
    return {key:tuple(sum(w*np.asarray(parts[n][key]) for n,w in zip(names,weights)))
            for key in parts[names[0]]}


def predict(root,data,out):
    saved = np.load(root/'shared.npz')
    config = json.loads((root/'ensemble.json').read_text())
    q,h = load_inputs(data/'test')
    parts = components(q,h,root,saved['linear'],[saved[f'p{i}'] for i in range(4)])
    write_submission(out,mix(parts,config['names'],config['weights']))


def train(args):
    root,data,source = Path(args.out).resolve(),Path(args.data).resolve(),Path(args.tree_source).resolve()
    if root.exists():
        raise ValueError('Choose a fresh output directory')
    if root == data or data in root.parents or root in data.parents:
        raise ValueError('Output must be separate from data')
    root.mkdir(parents=True)
    for name in ('model.ubj','baseline.json'):
        shutil.copy2(source/name,root/name)
    tq,th = load_inputs(data/'train')
    labels = read_labels(data/'train/labels.csv',tq)
    print('Training shared linear prior and neural sequence model...',flush=True)
    linear,params = train_shared(examples(tq,th,labels,stride=4),args.epochs)
    np.savez(root/'shared.npz',linear=linear,**{f'p{i}':p for i,p in enumerate(params)})
    vq,vh = load_inputs(data/'validation')
    parts = components(vq,vh,root,linear,params)
    targets = read_targets(data/'validation/targets.csv')
    groups = {r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv',('query_id','condition'))}
    reference = read_submission(data/'validation/reference.csv')
    tuning,audit = episode_folds(targets,groups)
    points = round_metadata(data)['max_points']
    names = list(parts)
    # Equal conditions, equal episodes within each condition, equal machines within episode.
    chosen = [t for t in targets if t.episode_id in tuning]
    episode_counts = {ep:sum(t.episode_id==ep for t in chosen) for ep in tuning}
    condition_episodes = {c:len({t.episode_id for t in chosen if groups[t.query_id]==c}) for c in set(groups.values())}
    denominator = {}
    for c in condition_episodes:
        ct = [t for t in chosen if groups[t.query_id]==c]
        denominator[c] = max(1e-9,sum(sum((np.asarray(reference[t.query_id])-t.probabilities)**2)/episode_counts[t.episode_id] for t in ct)/condition_episodes[c])
    scale = np.asarray([1/(4*condition_episodes[groups[t.query_id]]*episode_counts[t.episode_id]*denominator[groups[t.query_id]]) for t in chosen])
    P = np.asarray([[parts[n][t.query_id] for n in names] for t in chosen])
    truth = np.asarray([t.probabilities for t in chosen])
    def objective(w):
        error = np.einsum('nck,c->nk',P,w)-truth
        return np.sum(scale[:,None]*error**2),2*np.einsum('n,nck,nk->c',scale,P,error)
    fit = minimize(objective,np.ones(len(names))/len(names),jac=True,method='SLSQP',
                   bounds=[(0,1)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],options={'ftol':1e-12,'maxiter':500})
    if not fit.success:
        raise RuntimeError(fit.message)
    weights = np.maximum(fit.x,0); weights /= weights.sum()
    prediction = mix(parts,names,weights)
    scores = {n:subset_score(p,targets,groups,reference,tuning,points) for n,p in parts.items()}
    for n in names:
        write_submission(root/f'validation_{n}.csv',parts[n])
        print(f'{n}: tuning {scores[n]["technical_points"]:.3f}, weight {weights[names.index(n)]:.4f}',flush=True)
    config = dict(names=names,weights=weights.tolist())
    write_json(root/'ensemble.json',config)
    write_submission(root/'validation_predictions.csv',prediction)
    audit_result = subset_score(prediction,targets,groups,reference,audit,points)
    report = dict(components=scores,ensemble=config,tuning=subset_score(prediction,targets,groups,reference,tuning,points),audit=audit_result,
                  local_specs=SPECS,neural=dict(hidden_units=64,activation='tanh',epochs=args.epochs,seed=42),
                  caveat='Audit has been inspected in earlier experiments; not an untouched holdout.',
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                  input_sha256={str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('train','validation','test') for p in (data/folder).glob('*.csv')})
    write_json(root/'training-report.json',report)
    predict(root,data,root/'submission.csv')
    # Independent reload reproduces both shared neural and per-query local fitting.
    predict(root,data,root/'reproduced.csv')
    a,b = read_submission(root/'submission.csv'),read_submission(root/'reproduced.csv')
    difference = max(abs(x-y) for k in a for x,y in zip(a[k],b[k]))
    if difference > 1e-8:
        raise RuntimeError(f'Reproduction failed: {difference}')
    report['reproduction_max_difference'] = difference
    write_json(root/'training-report.json',report)
    entry = root/'entry'; entry.mkdir()
    for package in ('combo_model','slot_forecasting'):
        destination = entry/package; destination.mkdir()
        for p in Path(find_spec(package).origin).parent.glob('*.py'):
            shutil.copy2(p,destination/p.name)
    for filename in ('shared.npz','ensemble.json','model.ubj','baseline.json','training-report.json'):
        shutil.copy2(root/filename,entry/filename)
    shutil.copy2(Path(__file__).parent/'README.md',entry/'README.md')
    shutil.copy2(Path(__file__).parent.parent/'my_model/requirements.txt',entry/'requirements.txt')
    write_json(entry/'reproduce.json',dict(image='python',command=['python','-m','combo_model.run','predict','--model-dir','.','--data','/data','--out','/output/submission.csv']))
    package_code(entry,root/'entry.zip')
    print(f'Ensemble tuning: {report["tuning"]["technical_points"]:.3f}; audit: {audit_result["technical_points"]:.3f}; output: {root}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','predict'])
    parser.add_argument('--data',default='public')
    parser.add_argument('--out',required=True)
    parser.add_argument('--tree-source',default='Phillip_Lab/artifacts/neural_xgboost')
    parser.add_argument('--model-dir',default='.')
    parser.add_argument('--epochs',type=int,default=100)
    args = parser.parse_args()
    if args.action == 'train':
        train(args)
    else:
        predict(Path(args.model_dir),Path(args.data),Path(args.out))
