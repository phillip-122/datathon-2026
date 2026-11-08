"""Run team models in isolated subprocesses and mix their probabilities."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def read(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        return {r['query_id']:[float(r[c]) for c in ('p_0','p_1','p_2','p_5','p_20')] for r in csv.DictReader(f)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out',required=True); args=p.parse_args()
    root=Path(__file__).resolve().parent
    data=str(Path(args.data).resolve())
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',LOKY_MAX_CPU_COUNT='2')
    with tempfile.TemporaryDirectory() as tmp:
        a,b=Path(tmp)/'a.csv',Path(tmp)/'b.csv'
        subprocess.run([sys.executable,'-m','combo_model.run','predict','--model-dir','.', '--data',data,'--out',str(a)],cwd=root/'phillip',env=env,check=True)
        subprocess.run([sys.executable,'predict.py','--data',data,'--out',str(b)],cwd=root/'jonathan',env=env,check=True)
        left,right=read(a),read(b)
    if left.keys()!=right.keys():
        raise ValueError('Component query IDs disagree')
    weight=json.loads((root/'blend.json').read_text())['phillip_weight']
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n'); writer.writerow(['query_id','p_0','p_1','p_2','p_5','p_20'])
        for k in sorted(left):
            values=[weight*x+(1-weight)*y for x,y in zip(left[k],right[k])]
            total=sum(values)
            writer.writerow([k,*[format(v/total,'.17g') for v in values]])


if __name__=='__main__':
    main()
