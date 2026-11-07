# Datathon Lab --- What to Copy from the Starter Kit

> **Important:** This is a conservative copy plan based on the challenge
> structure discussed so far. The original ZIP is not attached in this
> conversation, so this is not yet a verified file-by-file inventory.
> Re-upload the starter ZIP and this checklist can be reconciled against
> its exact paths.

## Copy into the isolated lab

Preserve the starter kit's original directory structure wherever
possible.

### Required data

-   `public/train/` --- copy the complete directory. This is the primary
    source for fitting models and constructing historical training
    examples.
-   `public/validation/` --- copy the complete directory. Use it to
    evaluate candidate systems and compare predicted probability
    distributions with the supplied validation targets.
-   `public/test/` --- copy only if the organizer-provided kit includes
    public test inputs and the rules permit their use in the lab. Treat
    it as inference-only. Never use hidden answers, future outcomes, or
    information unavailable at the prediction cutoff.

### Required starter-kit code and specifications

Copy the complete starter-kit source and documentation needed to
understand and reproduce its baseline. In particular, retain these if
present:

-   `run_starter.py` and any scripts it imports.
-   `predict.py` or the starter prediction entry point, if present.
-   All Python modules/packages imported by those scripts.
-   `requirements.txt`, `pyproject.toml`, environment files, or
    dependency/version specifications.
-   The participant guide, README, rules, data dictionary/schema,
    scoring instructions, and reproduction instructions.
-   Any small configuration files or static assets the starter scripts
    require.

Do not copy only the top-level script if it imports helper modules from
sibling folders.

## Do not treat generated outputs as source data

Do not copy previous run-output directories such as `outputs/`,
`outputs_pooled_recent/`, or similar experiment-result folders as if
they were training data.

They may be useful as **reference artifacts** if you want to compare
against a previous baseline, but they are not a substitute for
`public/train/`. If you keep them, put them under a clearly named
`reference_runs/` directory and record which command produced them.

## Suggested isolated-lab layout

``` text
house-edge-lab/
├── README.md
├── LAB_START.md
├── PROGRESS_LOG.md
├── COPY_FROM_STARTER_KIT.md
├── starter_kit/
│   ├── public/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/              # only if permitted/provided
│   ├── run_starter.py
│   ├── predict.py             # if present
│   ├── requirements.txt       # if present
│   └── challenge_docs/
├── research/
│   ├── hypotheses/
│   ├── experiments/
│   ├── agents/
│   └── notes/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── orchestration/
│   └── evaluation/
├── artifacts/
│   ├── experiment_registry/
│   ├── models/
│   └── reports/
├── runs/
│   ├── baseline/
│   └── experiments/
└── submission/
```

## Data-use boundaries

1.  Training examples may use only information available before their
    historical cutoff.
2.  A missing tick is unknown, not an observed zero payout.
3.  Cross-machine features may use only other machines' observations
    from the same episode that are available before the target cutoff.
4.  Never mix information across unrelated episodes.
5.  Validation is for model selection and diagnostics. Do not silently
    turn validation targets into training labels.
6.  Do not access or attempt to reconstruct hidden test answers.
7.  The final submission must run offline within the organizer's
    execution limits and reproduce the submitted predictions within the
    stated tolerance.

## First action after copying

Run the starter kit's documented baseline command unchanged and save its
output to a fresh directory. Do not overwrite or modify the original
starter files. Record the command, dependency versions, output files,
runtime, and validation score in `PROGRESS_LOG.md`.

## Verification still needed

Once the original ZIP is available, verify: - exact folder and file
names; - whether train, validation, and test are nested under `public/`
or another root; - which scripts are imported by the starter entry
point; - whether the kit includes a `predict.py`, requirements file, or
saved baseline; - the official output schema and exact reproduction
command.
