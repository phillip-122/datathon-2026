"""Stable entry point for Phillip's organized model lab."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

LAB=Path(__file__).resolve().parent
REPO=LAB.parent
sys.path[:0]=[str(LAB/'src'),str(REPO)]
os.chdir(REPO)

COMMANDS={
    'train-v12':('v12_new_sys.run',['train']),
    'train-xgboost':('my_model.train',[]),
    'train-ensemble':('combo_model.run',['train']),
    'train-hgb':('hgb_upgrade.run',['train']),
    'train-lags':('lag_model.run',['train']),
    'train-structure':('pdf_model.run',['train']),
    'train-win':('win_model.run',['train']),
    'train-calibration':('calibrated_model.run',['train']),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=[*COMMANDS,'predict','test','compare'])
    args,rest=parser.parse_known_args()
    if args.command=='test':
        modules=['my_model.test_features','combo_model.test_models','hgb_upgrade.test_data',
                 'win_model.test_probability','lag_model.test_features','pdf_model.test_features',
                 'calibrated_model.test_calibration','v12_new_sys.test_system']
        sys.argv=['unittest',*modules,'-v',*rest]
        runpy.run_module('unittest',run_name='__main__')
    elif args.command=='compare':
        sys.argv=['compare_models.py',*rest]
        runpy.run_path(str(LAB/'research/tools/compare_models.py'),run_name='__main__')
    elif args.command=='predict':
        p=argparse.ArgumentParser(description='Reproduce the immutable active submission package.')
        p.add_argument('--data',default='public')
        p.add_argument('--out',default='Phillip_Lab/runs/active-reproduced.csv')
        p.add_argument('--archive',default='Phillip_Lab/compiled_versions/active_champion_submission/submission_entry.zip')
        options=p.parse_args(rest)
        from slot_forecasting.submission_contract import extract_archive
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'entry'
            config=extract_archive(Path(options.archive).read_bytes(),root)
            command=[sys.executable,*config['command'][1:]]
            # Contract uses explicit --data and --out arguments.
            command[command.index('--data')+1]=str(Path(options.data).resolve())
            command[command.index('--out')+1]=str(Path(options.out).resolve())
            subprocess.run(command,cwd=root,check=True)
        print('Reproduced:',Path(options.out).resolve())
    else:
        module,prefix=COMMANDS[args.command]
        if '--out' not in rest:
            rest+=['--out',str(Path('Phillip_Lab/runs')/(args.command+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')))]
        sys.argv=[module,*prefix,*rest]
        runpy.run_module(module,run_name='__main__')


if __name__=='__main__':
    main()
