# LAB_START --- Start the Research Loop

## Mission

Build the strongest reproducible payout-distribution predictor for the
House Edge competition.

The lab is an interactive research environment. Agents may inspect data,
propose hypotheses, generate code, run experiments, interpret results,
and recommend the next experiment. Python scripts---not live agent
improvisation---must execute the final competition predictions.

The goal is not to preserve every experiment. The goal is to preserve
verified discoveries and turn them into a deterministic inference
pipeline.

## Core distinction: research loop vs. inference pipeline

### Research loop

``` text
Inspect data
   ↓
Form a falsifiable hypothesis
   ↓
Design a controlled experiment
   ↓
Generate/modify executable code
   ↓
Run across episode-grouped folds
   ↓
Evaluate probability quality and failure cases
   ↓
Keep / revise / reject
   ↓
Record result and choose next experiment
```

### Final inference pipeline

``` text
Organizer-provided test inputs
   ↓
Validate schema and cutoff constraints
   ↓
Build permitted features and temporary episode state
   ↓
Run frozen model(s) and combination logic
   ↓
Validate five probabilities
   ↓
Write submission.csv
```

Agents can help build and improve the pipeline during research. The
final predictor must not depend on an LLM, network access, or an agent
deciding what code to write at submission runtime unless the competition
explicitly permits that and the execution environment supports it. The
safe design is to compile successful research into ordinary Python code
before submission.

## Step 0 --- Establish a clean baseline

1.  Keep the original starter kit unchanged under `starter_kit/`.
2.  Create a separate Python environment for the lab.
3.  Install only dependencies supported by the starter kit and
    competition environment.
4.  Run the official starter command exactly as documented in the kit.
5.  Save the baseline output under a new timestamped or uniquely named
    run directory.
6.  Record the command, versions, runtime, validation metric, and output
    hashes.

Do not begin by replacing the starter model. First establish that the
supplied baseline runs and that its output format is understood.

## Step 1 --- Build the experiment runner

Create a single repeatable interface for experiments. Every run should
record:

-   experiment ID and timestamp;
-   hypothesis and expected result;
-   data split and episode folds;
-   feature set and model configuration;
-   random seeds;
-   code revision or source hash;
-   command and dependency versions;
-   runtime and resource use;
-   validation metrics;
-   decision: keep, revise, or reject;
-   next question.

Prefer configuration files over manually editing source code for every
experiment.

## Step 2 --- Use agents as research collaborators

Suggested roles (these can be separate agents or separate prompts to one
agent):

1.  **Data auditor:** verifies schemas, missingness, cutoff semantics,
    episode grouping, and leakage risks.
2.  **Population specialist:** tests machine type, signal, and their
    interactions.
3.  **Temporal specialist:** tests recency, rolling windows, lag
    features, and changes over time.
4.  **Episode specialist:** tests whether other machines' earlier
    observations improve predictions.
5.  **Modeling specialist:** compares statistical baselines,
    scikit-learn models, and XGBoost.
6.  **Evaluation critic:** challenges claims, checks fold design, looks
    for overfitting, and analyzes failure cases.
7.  **Orchestrator:** assigns bounded experiments, tracks dependencies,
    and selects the next high-value question.

Agents should produce structured experiment proposals and code
changes---not unsupported conclusions. Every claim must point to a
reproducible result.

## Step 3 --- Experimental rules

-   Split by whole episode, not random individual rows, when evaluating
    generalization to unseen episodes.
-   Fit preprocessing and model parameters only on the training portion
    of each fold.
-   Keep the official validation set separate enough to provide a
    meaningful model-selection check.
-   Use the official scoring metric and compare probability
    distributions, not only hard class accuracy.
-   Preserve the five payout classes in the required order:
    `[0, 1, 2, 5, 20]`.
-   Check that every prediction is finite, lies in `[0, 1]`, and sums to
    `1` within the required tolerance.
-   Compare every new method against the current best baseline.
-   Run ablations: remove one feature family at a time to see whether it
    actually contributes.
-   Record negative results. A failed experiment is useful if it narrows
    the search.
-   Do not repeatedly tune to the same validation episodes until the
    score becomes a proxy for memorization.

## Step 4 --- Research priorities

Work in this order unless evidence suggests a better path:

1.  Reproduce and understand the starter baseline.
2.  Audit the data-generating structure and target semantics.
3.  Establish episode-grouped cross-validation.
4.  Test population/type/signal baselines.
5.  Test machine-specific history and shrinkage toward population
    estimates.
6.  Test temporal and recency features.
7.  Test same-episode cross-machine information with strict cutoff
    controls.
8.  Compare models: smoothed frequency/statistical models, scikit-learn,
    XGBoost.
9.  Test combinations or a gating model only after individual
    specialists demonstrate value.
10. Freeze and reproduce the best pipeline.

Do not assume that a complicated model is better. Prefer the simplest
system that improves held-out episode performance reliably.

## Step 5 --- Definition of done

The lab is competition-ready when:

-   the final code runs offline from a clean environment;
-   all required modules and model artifacts are packaged;
-   inference uses only permitted inputs available at the cutoff;
-   the output has the exact required query IDs, columns, class order,
    and probability constraints;
-   the organizer's reproduction command succeeds;
-   regenerated predictions match the submitted CSV within the
    organizer's tolerance;
-   runtime and memory fit the competition limits.

## First session checklist

-   [ ] Copy the starter kit and verify the file inventory.
-   [ ] Read the official rules, data dictionary, scoring, and
    reproduction instructions.
-   [ ] Create the isolated Python environment.
-   [ ] Run the untouched starter baseline.
-   [ ] Record its score and outputs.
-   [ ] Inspect train/validation schemas and episode relationships.
-   [ ] Create the experiment registry.
-   [ ] Run one small, falsifiable experiment before building a large
    agent framework.

## Operating principle

**Agents explore. Experiments verify. The registry remembers. The final
script reproduces.**
