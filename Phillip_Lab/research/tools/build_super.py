import sys, os
from pathlib import Path
_REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(_REPO/'Phillip_Lab/src'),str(_REPO)]
os.chdir(_REPO)
"""Select a two-team blend on tuning episodes and package isolated inference."""
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from slot_forecasting.io import read_submission,read_targets,read_rows,write_submission,write_json
from slot_forecasting.experiments import episode_folds,subset_score
from slot_forecasting.submission_contract import extract_archive,package_code

ROOT=Path('Phillip_Lab/runs/house-edge-output-super')


def main():
    if ROOT.exists():
        raise ValueError('Output already exists; preserve the previous experiment.')
    v=Path('public/validation')
    targets=read_targets(v/'targets.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(v/'conditions.csv',('query_id','condition'))}
    ref=read_submission(v/'reference.csv')
    tuning,audit=episode_folds(targets,groups)
    a=read_submission('Phillip_Lab/artifacts/neural_xgboost/validation_predictions.csv')
    b=read_submission('Phillip_Lab/runs/comparisons/jonathan-v3-validation.csv')
    def blend(weight,left=a,right=b):
        return {k:tuple(weight*np.asarray(left[k])+(1-weight)*np.asarray(right[k])) for k in left}
    def score(p,eps):
        return subset_score(p,targets,groups,ref,eps,60)
    fit=minimize_scalar(lambda w:score(blend(w),tuning)['mean_loss_ratio'],bounds=(0,1),method='bounded',options={'xatol':1e-9})
    candidates=[0.,1.,float(fit.x)]
    weight=min(candidates,key=lambda w:score(blend(w),tuning)['mean_loss_ratio'])
    prediction=blend(weight)
    ROOT.mkdir()
    report={'phillip_weight':weight,'jonathan_weight':1-weight,'selection':'Minimize tuning loss ratio only; audit has been inspected previously.',
            'comparison':{name:{fold:score(p,eps) for fold,eps in [('tuning',tuning),('audit',audit),('full',tuning|audit)]}
                          for name,p in [('phillip',a),('jonathan',b),('combined',prediction)]}}
    write_json(ROOT/'report.json',report)
    write_submission(ROOT/'validation_predictions.csv',prediction)
    ta=read_submission('Phillip_Lab/artifacts/neural_xgboost/submission.csv')
    tb=read_submission('Phillip_Lab/runs/comparisons/jonathan-v3-test.csv')
    write_submission(ROOT/'submission.csv',blend(weight,ta,tb))
    entry=ROOT/'entry'; entry.mkdir()
    extract_archive(Path('Phillip_Lab/artifacts/neural_xgboost/entry.zip').read_bytes(),entry/'phillip')
    extract_archive(Path('Jonathan_Lab_1/compiled_versions/active_champion_submission/submission_entry.zip').read_bytes(),entry/'jonathan')
    write_json(entry/'blend.json',{'phillip_weight':weight})
    shutil.copy2(Path(__file__).with_name('super_predict.py'),entry/'predict.py')
    (entry/'requirements.txt').write_text('numpy==2.3.4\nscipy==1.16.3\nxgboost==3.1.1\nscikit-learn==1.9.1\njoblib==1.5.2\nthreadpoolctl==3.8.0\nnarwhals==2.28.0\n',encoding='utf-8')
    (entry/'README.md').write_text('# Combined House Edge entry\n\nPython 3.11+. Prepare pinned requirements.txt before offline prediction. Run python predict.py --data /data --out /output/submission.csv.\n\nIncludes frozen Phillip neural/XGBoost ensemble and Jonathan V3 in isolated subprocesses, then mixes probabilities with a scalar weight selected on tuning episodes only. No retraining or test labels. Audit has been inspected previously. Each component keeps its original code and artifacts. Prediction is limited to two numerical threads per subprocess; components run sequentially. See report.json for scores.\n',encoding='utf-8')
    shutil.copy2(ROOT/'report.json',entry/'report.json')
    write_json(entry/'reproduce.json',{'image':'python','command':['python','predict.py','--data','/data','--out','/output/submission.csv']})
    package_code(entry,ROOT/'entry.zip')
    print('Phillip weight:',weight)
    for name,result in report['comparison'].items():
        print(name,{fold:round(s['technical_points'],6) for fold,s in result.items()})


if __name__=='__main__':
    main()
