# Part 2 — updated to A6, above 13

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
