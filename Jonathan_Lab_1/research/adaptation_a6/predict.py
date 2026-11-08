"""Offline A6 predictor: Jonathan A4 plus cross-fitted temporal residual model."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import joblib
import numpy as np
from experts import bank,replace_win
from slot_forecasting.io import load_inputs,read_submission,write_submission


def predict(root,data,out):
    data=data.resolve()
    split=data/'test' if (data/'test').is_dir() else data
    q,h=load_inputs(split)
    with tempfile.TemporaryDirectory() as tmp:
        rawpath=Path(tmp)/'raw.csv'
        subprocess.run([sys.executable,str((root/'base/predict.py').resolve()),'--data',str(split),'--out',str(rawpath)],check=True)
        raw=read_submission(rawpath,{x.query_id for x in q})
    P=np.asarray([raw[x.query_id] for x in q]); B,M=bank(q,h,P)
    config=json.loads((root/'config.json').read_text())
    if config['method']=='expert': rates=B[:,config['column']]
    else:
        model=joblib.load(root/'residual.joblib')
        rates=np.clip(1-P[:,0]-model.predict(np.column_stack((P,B,M))),.001,.999)
    result=replace_win(P,rates,config['weight'])
    write_submission(out,{x.query_id:tuple(p) for x,p in zip(q,result)})
    print('A6 prediction complete:',len(q),'queries',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    predict(Path(__file__).resolve().parent,Path(a.data),Path(a.out))
