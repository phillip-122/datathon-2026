"""Freeze the selected A6, preserving Jonathan's original A4 archive."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'adaptation/kit'))
from slot_forecasting.submission_contract import extract_archive,package_code


README='''# Adaptation A6 — Jonathan A4 plus Bayesian temporal residual correction

Upload `submission.csv` and `submission_entry.zip` together for Part 2.

Full public validation: **16.634796 / 20** (includes correction-fitting episodes).
Audit episodes: **15.637562 / 20**. Cross-fitted tuning selection: **15.050522 / 20**.
Previous A5: 12.360656 full / 12.314882 audit. Original Jonathan A4: 11.764303 full.
Public validation has been inspected repeatedly, including while developing A4.
The audit score is not an untouched estimate of test quality. Hidden test score
is unknown; the 16.63 figure must not be described as fully held out.

The predictor first runs Jonathan's frozen A4 adaptation model. A bank of 36
Bayesian local temporal experts uses binary and three-category payout contexts
at single and paired lags 1–8, half-lives 24/64/infinite, prior strengths 1/4/16
and evidence temperatures 1/2. Missing ticks remain unobserved. All inputs are
strictly earlier than each query cutoff. History summaries include changes in
local and peer payout rates and lag polarity. Other episodes are isolated.

A HistGradientBoostingRegressor learns A4's p_0 residual from those features:
150 iterations, 7 leaves, minimum leaf size 25, L2 10, learning rate .04,
early stopping disabled, random seed 42. It is fitted only on tuning episodes.
Three condition-balanced episode folds select the correction; condition-normalized
reference errors weight its training loss. Winning probabilities are clipped to
[.001,.999]; conditional positive payout proportions are retained from A4.
Audit targets never fit the new correction. No target files, condition labels,
training process, forecast lookup table, internet or API is needed at inference.

Install pinned requirements before the offline run. From the extracted ZIP:

```
python predict.py --data /absolute/path/to/adaptation/public --out /absolute/path/to/submission.csv
```

`--data` also accepts the direct folder with queries.csv and history.csv.
The ZIP has root README.md/reproduce.json and contains all fitted model weights.
No data files or secrets are packaged. Use the current adaptation kit's checker;
the old main kit's checker rejects the new 80/20 metadata.

Training and model selection from the repository root (use a fresh output folder):

```
.venv/Scripts/python.exe Jonathan_Lab_1/research/adaptation_a6/run.py --data adaptation/kit/public --out Phillip_Lab/runs/adaptation_a6_new
```

The organizer adaptation data is also at Jonathan_Lab_1/adaptation-participants/public.
Model comparisons and limitations are in report.json. Source and artifact hashes,
format checks and clean-extraction reproduction results are in MANIFEST.json
beside the upload files. Nothing is submitted automatically.
'''


def main(a):
    source=Path(a.run).resolve(); dest=Path(a.out).resolve(); dest.mkdir(exist_ok=False,parents=True)
    archive=REPO/'Jonathan_Lab_1/compiled_versions/adaptation_active_submission/submission_entry.zip'
    report=json.loads((source/'report.json').read_text())
    report.update(round='adaptation',max_points=20,base_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (dest/'report.json').write_text(json.dumps(report,indent=2))
    (dest/'config.json').write_text(json.dumps(report['selection'],indent=2))
    (dest/'README.md').write_text(README,encoding='utf-8')
    shutil.copy2(source/'validation_predictions.csv',dest/'validation_predictions.csv')
    with tempfile.TemporaryDirectory() as tmp:
        entry=Path(tmp)/'entry'; entry.mkdir()
        extract_archive(archive.read_bytes(),entry/'base')
        for name in ('experts.py','predict.py'): shutil.copy2(Path(__file__).parent/name,entry/name)
        shutil.copytree(REPO/'adaptation/kit/slot_forecasting',entry/'slot_forecasting',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(source/'residual.joblib',entry/'residual.joblib')
        for name in ('config.json','report.json','README.md'): shutil.copy2(dest/name,entry/name)
        (entry/'requirements.txt').write_text(''.join(f'{name}=={importlib.metadata.version(name)}\n' for name in ('numpy','scipy','scikit-learn','joblib','threadpoolctl','narwhals')))
        (entry/'reproduce.json').write_text(json.dumps({'image':'python','command':['python','predict.py','--data','/data','--out','/output/submission.csv']},indent=2))
        package_code(entry,dest/'submission_entry.zip')
    print('Packaged:',dest)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',default=str(REPO/'Phillip_Lab/runs/adaptation_a6')); p.add_argument('--out',default=str(REPO/'Jonathan_Lab_1/compiled_versions/adaptation_a6_submission')); main(p.parse_args())
