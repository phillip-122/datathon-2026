"""V12: frozen V10 plus cross-fitted constrained geometry/regime inference."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
import io
import zipfile
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
from slot_forecasting.io import load_inputs,read_targets,read_queries,read_rows,read_submission,write_submission,write_json
from slot_forecasting.experiments import episode_folds,subset_score
from slot_forecasting.submission_contract import extract_archive,package_code
from .system import learn_constraints,alternatives,blend


def base_predict(root,data,out):
    base=(root/'base-entry').resolve()
    subprocess.run([sys.executable,str(base/'predict.py'),'--model-dir',str(base),
                    '--data',str(data.resolve()),'--out',str(out.resolve())],cwd=base,check=True,
                   env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',LOKY_MAX_CPU_COUNT='2'))


def apply(q,h,raw,config):
    if config['method']=='v10': return raw
    candidates=alternatives(q,h,raw,config['constraints'])
    return blend(candidates[config['method']],raw,config['weight'])


def predict(root,data,out):
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        rawfile=Path(tmp)/'base.csv'; base_predict(root,data,rawfile)
        q,h=load_inputs(data/'test' if (data/'test').is_dir() else data)
        raw=read_submission(rawfile,{x.query_id for x in q})
    config=json.loads((root/'v12.json').read_text())
    write_submission(out,apply(q,h,raw,config))
    print(f'V12 offline prediction: {time.perf_counter()-start:.2f}s',flush=True)


def train(args):
    start=time.perf_counter(); root=Path(args.out); data=Path(args.data)
    if root.exists() and any(root.iterdir()): raise ValueError('Choose a fresh output folder')
    root.mkdir(parents=True,exist_ok=True)
    archive=Path(args.base_archive)
    content=archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(content)) as original:
        files=[n for n in original.namelist() if not n.endswith('/')]
        if 'README.md' not in files and all(n.startswith('entry/') for n in files):
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as normalized:
                for name in files:
                    if name[len('entry/'):]!='reproduce.json':
                        normalized.writestr(name[len('entry/'):],original.read(name))
                normalized.writestr('reproduce.json',json.dumps({'image':'python','command':['python','predict.py','--model-dir','.','--data','/data','--out','/output/submission.csv']}))
            content=buffer.getvalue()
    extract_archive(content,root/'base-entry')
    base_predict(root,data/'validation',root/'v10_validation.csv')
    q,h=load_inputs(data/'validation'); queries={x.query_id:x for x in q}
    targets=read_targets(data/'validation/targets.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv',('query_id','condition'))}
    reference=read_submission(data/'validation/reference.csv')
    raw=read_submission(root/'v10_validation.csv',{x.query_id for x in q})
    tuning,audit=episode_folds(targets,groups)
    def score(pred,eps): return subset_score(pred,targets,groups,reference,eps,60)
    T=[t for t in targets if t.episode_id in tuning]
    folds={}
    for condition in sorted(set(groups.values())):
        eps=sorted({t.episode_id for t in T if groups[t.query_id]==condition},key=lambda x:hashlib.sha256(x.encode()).hexdigest())
        folds.update({e:i%3 for i,e in enumerate(eps)})
    oof={}
    for fold in range(3):
        training=[t for t in T if folds[t.episode_id]!=fold]
        held={t.episode_id for t in T if folds[t.episode_id]==fold}
        cq=[x for x in q if x.episode_id in held]
        ch=[r for r in h if r.episode_id in held]
        parameters=learn_constraints(training,queries)
        candidates=alternatives(cq,ch,raw,parameters)
        for name,p in candidates.items(): oof.setdefault(name,{}).update(p)
    best=score(raw,tuning)['mean_loss_ratio']; choice={'method':'v10','weight':0.}; comparison={}
    for name,p in oof.items():
        for weight in (.25,.5,.75,1.):
            pred=blend(p,{k:raw[k] for k in p},weight)
            result=score(pred,tuning); comparison[f'{name}_{weight}']=result
            print(f'{name} weight {weight}: cross-fitted tuning {result["technical_points"]:.6f}',flush=True)
            if result['mean_loss_ratio']<best:
                best=result['mean_loss_ratio']; choice={'method':name,'weight':weight}
    choice['constraints']=learn_constraints(T,queries)
    write_json(root/'v12.json',choice)
    predicted=apply(q,h,raw,choice); write_submission(root/'validation_predictions.csv',predicted)
    candidates=alternatives(q,h,raw,choice['constraints'])
    # Fixed mathematical projection comparator reproduces the conservation-style
    # V11 transform separately from the new Euclidean projection.
    v11={}
    for query in q:
        p=raw[query.query_id]; c=np.asarray(choice['constraints'][query.machine_type]['high_ratio'])
        v11[query.query_id]=tuple([p[0],p[1],*((1-p[0]-p[1])*c)])
    benchmarks={name:{label:score(p,eps) for label,eps in [('tuning',tuning),('audit',audit),('full',tuning|audit)]}
                for name,p in [('v10',raw),('v11',v11),('v12_new_sys',predicted)]}
    saved_path=Path(args.saved_v10)
    saved=read_submission(saved_path,{x.query_id for x in q})
    benchmarks['v10_saved']={label:score(saved,eps) for label,eps in [('tuning',tuning),('audit',audit),('full',tuning|audit)]}
    saved_v11={}
    for query in q:
        p=saved[query.query_id]; c=np.asarray(choice['constraints'][query.machine_type]['high_ratio'])
        saved_v11[query.query_id]=tuple([p[0],p[1],*((1-p[0]-p[1])*c)])
    benchmarks['v11_saved']={label:score(saved_v11,eps) for label,eps in [('tuning',tuning),('audit',audit),('full',tuning|audit)]}
    episode_losses={}
    for ep in audit:
        rows=[t for t in targets if t.episode_id==ep]
        episode_losses[ep]=[np.mean([np.sum((np.asarray(p[t.query_id])-t.probabilities)**2) for t in rows]) for p in (saved,predicted,reference)]
    rng=np.random.default_rng(42); gains=[]
    for _ in range(2000):
        ratio=[[],[]]
        for condition in sorted(set(groups.values())):
            eps=sorted({t.episode_id for t in targets if t.episode_id in audit and groups[t.query_id]==condition})
            sample=rng.choice(eps,len(eps),replace=True)
            loss=np.asarray([episode_losses[e] for e in sample]).mean(0)
            for i in range(2): ratio[i].append(loss[i]/max(1e-9,loss[2]))
        gains.append(60/(1+np.mean(ratio[1]))-60/(1+np.mean(ratio[0])))
    report={'selection':choice,'crossfitted_tuning':comparison,'benchmarks':benchmarks,
            'audit_gain_vs_saved_v10_ci95':np.quantile(gains,[.025,.975]).tolist(),'folds':folds,
            'base_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'saved_v10_sha256':hashlib.sha256(saved_path.read_bytes()).hexdigest(),
            'base_package_vs_saved_v10_max_probability_difference':max(abs(x-y) for k in raw for x,y in zip(raw[k],saved[k])),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'note':'Constraint values estimated from tuning targets only, within folds for model selection. No audit/test targets fitted. Existing V10 and evaluation episodes have been inspected/selected previously: not an untouched test. Projection improves Euclidean error whenever truth belongs to the estimated convex region; region membership on hidden data is not guaranteed.'}
    for name,p in candidates.items(): write_submission(root/f'validation_{name}.csv',p)
    write_json(root/'report.json',report)
    predict(root,data,root/'submission.csv')
    entry=root/'entry'; entry.mkdir(); shutil.copytree(root/'base-entry',entry/'base-entry')
    for package in ('v12_new_sys','slot_forecasting'):
        dest=entry/package; dest.mkdir()
        for p in Path(find_spec(package).origin).parent.glob('*.py'): shutil.copy2(p,dest/p.name)
    for name in ('v12.json','report.json'): shutil.copy2(root/name,entry/name)
    (entry/'requirements.txt').write_text('numpy==2.3.4\nscipy==1.16.3\nscikit-learn==1.9.1\njoblib==1.5.2\nthreadpoolctl==3.8.0\nnarwhals==2.28.0\n',encoding='utf-8')
    shutil.copy2(Path(__file__).parent/'README.md',entry/'README.md')
    write_json(entry/'reproduce.json',{'image':'python','command':['python','-m','v12_new_sys.run','predict','--model-dir','.','--data','/data','--out','/output/submission.csv']})
    package_code(entry,root/'entry.zip')
    report['training_seconds']=time.perf_counter()-start; write_json(root/'report.json',report)
    print('SELECTED:',choice['method'],choice['weight'])
    for name,s in benchmarks.items(): print(name,{k:round(v['technical_points'],6) for k,v in s.items()})
    print('Audit gain vs V10 CI:',report['audit_gain_vs_saved_v10_ci95'])


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['train','predict']); p.add_argument('--data',default='public'); p.add_argument('--out',required=True); p.add_argument('--model-dir',default='.')
    p.add_argument('--base-archive',default='Jonathan_Lab_1/compiled_versions/v10_short_history_subpop_triage_solution/submission_entry.zip')
    p.add_argument('--saved-v10',default='Jonathan_Lab_1/compiled_versions/v10_short_history_subpop_triage_solution/validation_predictions.csv')
    a=p.parse_args()
    if a.action=='train': train(a)
    else: predict(Path(a.model_dir),Path(a.data),Path(a.out))
