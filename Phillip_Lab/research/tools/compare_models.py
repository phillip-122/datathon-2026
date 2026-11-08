import sys, os
from pathlib import Path
_REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(_REPO/'Phillip_Lab/src'),str(_REPO)]
os.chdir(_REPO)
"""Compare two validation CSVs on identical official episode folds."""
import argparse
from pathlib import Path
from slot_forecasting.io import read_submission,read_targets,read_rows,write_json
from slot_forecasting.experiments import episode_folds,subset_score
from slot_forecasting.rounds import round_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a',required=True)
    p.add_argument('--b',required=True)
    p.add_argument('--data',default='public')
    p.add_argument('--out',default='Phillip_Lab/runs/comparisons/comparison.json')
    args=p.parse_args()
    root=Path(args.data); v=root/'validation'
    targets=read_targets(v/'targets.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(v/'conditions.csv',('query_id','condition'))}
    reference=read_submission(v/'reference.csv')
    tuning,audit=episode_folds(targets,groups)
    report={}
    for name,path in [('a',args.a),('b',args.b)]:
        preds=read_submission(path,{t.query_id for t in targets})
        report[name]={'file':path}
        for fold,episodes in [('tuning',tuning),('audit',audit),('full',tuning|audit)]:
            report[name][fold]=subset_score(preds,targets,groups,reference,episodes,round_metadata(root)['max_points'])
        print(f"{name}: {path}")
        print('  '+', '.join(f"{fold}: {report[name][fold]['technical_points']:.6f}" for fold in ('tuning','audit','full')))
    report['note']='Same official folds and reference. Audit episodes have been inspected repeatedly.'
    write_json(args.out,report)


if __name__=='__main__':
    main()
