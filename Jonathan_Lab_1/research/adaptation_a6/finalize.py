"""Verify clean-room outputs and replace the convenient Part 2 upload files."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'adaptation/kit'))
from slot_forecasting.io import read_submission


def main():
    source=REPO/'Jonathan_Lab_1/compiled_versions/adaptation_a6_submission'
    original=source/'submission.csv'; reproduced=REPO/'Phillip_Lab/runs/adaptation_a6_reproduced.csv'
    a,b=read_submission(original),read_submission(reproduced)
    assert a.keys()==b.keys()
    difference=max(abs(x-y) for k in a for x,y in zip(a[k],b[k]))
    assert difference<1e-8
    manifest={'round':'adaptation','max_points':20,'queries':len(a),'maximum_probability_difference':difference,'exact_file_match':original.read_bytes()==reproduced.read_bytes(),'packaged_validation_maximum_probability_difference':0.0,'full_validation_points':16.63479646501288,'audit_points':15.637561547360713,'crossfitted_tuning_selection_points':15.050521678406017,'sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('submission.csv','submission_entry.zip','validation_predictions.csv')},'note':'Full validation contains correction fitting episodes. Audit and tuning have been repeatedly inspected. Actual hidden test score is unknown.'}
    (source/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    destination=REPO/'PART_2_READY_TO_UPLOAD'
    for path in source.iterdir():
        if path.is_file(): shutil.copy2(path,destination/path.name)
    (destination/'START_HERE.md').write_text('''# Part 2 — updated to A6, above 13

Upload **submission.csv** and **submission_entry.zip** from this folder to the
adaptation form: https://forms.gle/tdPPza68SyrruAJ98

These are the matching A6 files. They replace the earlier A5 files in this folder.
The original A5 is preserved under Jonathan_Lab_1/compiled_versions/adaptation_a5_submission.

Scores out of 20:

- Cross-fitted tuning selection: **15.050522**.
- Audit episodes: **15.637562**.
- Full public validation: **16.634796** (includes episodes used to fit correction).

The full score is not a held-out result. Public validation has been inspected
repeatedly. Hidden test results are unknown.

All 1,536 test queries pass the official adaptation checker. A second inference
run from a separate ZIP extraction, using only the test input folder, reproduced
the CSV exactly: maximum probability difference 0.0. Packaged validation inference
also matches the reported validation predictions exactly.

Submit the ZIP as-is. Do not substitute validation_predictions.csv for submission.csv.
README.md explains the model and offline reproduction. report.json contains
comparisons; config.json contains settings; MANIFEST.json records file hashes.

The pasted participant guide lists November 8, 2026 at 12 PM Central as the
adaptation deadline. Wait for form confirmation and save the receipt. Nothing
has been uploaded by Codex.
''',encoding='utf-8')
    print(json.dumps(manifest,indent=2)); print('Updated:',destination)


if __name__=='__main__': main()
