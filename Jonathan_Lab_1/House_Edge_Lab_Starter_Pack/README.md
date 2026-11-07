# House Edge --- Agentic Research Lab

## Project objective

Win the House Edge payout-distribution forecasting competition by
building a high-performing, reproducible prediction system.

The competition asks for a probability distribution over five payout
outcomes:

``` text
[0×, 1×, 2×, 5×, 20×]
```

For each query, the predictor must output five valid probabilities that
sum to one. The objective is to estimate the simulator's underlying
probability distribution as accurately as possible---not merely guess
the next sampled payout.

## Our thesis

We are treating model development as an observable, interactive research
environment.

Instead of asking one opaque model to learn everything, we will use
agents and executable experiments to investigate mechanisms in the data.
Agents can inspect the current evidence, choose useful analyses, invoke
tools or specialist agents, generate candidate operations, and interpret
feedback.

Successful operations become versioned, tested components of the final
predictor.

This is an **agent-assisted model-discovery process**, not a requirement
to run agents during competition inference.

## The two systems we are building

### A. Research laboratory

The research lab is allowed to be exploratory:

-   inspect datasets and schemas;
-   formulate competing hypotheses;
-   generate feature calculations and model variants;
-   run controlled experiments;
-   compare results across held-out episodes;
-   analyze where predictions fail;
-   keep, revise, or reject candidate operations;
-   preserve a trace of the experiments and decisions.

The lab should be interactive and inspectable. Each experiment must be
reproducible from code, configuration, data split, and seed.

### B. Final prediction system

The submission is deliberately much simpler operationally:

-   load the permitted input files;
-   build only features available at each query's cutoff;
-   calculate episode-local state from permitted earlier observations;
-   run the frozen model and combination logic;
-   emit the required probability distribution;
-   run offline within organizer constraints.

The final submission should not need the research agents, notebooks,
Databricks, internet access, or an external LLM.

## Mental model: environment and episode

An episode is a related group of machines sharing a timeline and
potentially shared hidden conditions. The hidden condition is not
directly available; we can only infer useful information from
observations permitted before a query's cutoff.

We do not recreate the simulator. We build an observer/predictor that
estimates behavior from the released data.

Maintain two kinds of state:

-   **Episode state:** temporary estimates inferred from one episode's
    permitted observations. Reset between episodes.
-   **Research state:** durable knowledge from experiments---features,
    models, configurations, findings, and decisions.

Never leak information between unrelated episodes.

## Architecture

``` text
                    RESEARCH TIME
┌────────────────────────────────────────────────────┐
│ Data auditor / specialist agents / orchestrator     │
│          ↓                                         │
│ Hypothesis → experiment specification → code        │
│          ↓                                         │
│ Episode-grouped experiment runner                   │
│          ↓                                         │
│ Metrics + diagnostics + failure analysis            │
│          ↓                                         │
│ Experiment registry: keep / revise / reject         │
└──────────────────────────┬─────────────────────────┘
                           │ verified components
                           ▼
                    FINAL INFERENCE
┌────────────────────────────────────────────────────┐
│ Input validation                                    │
│   → cutoff-safe feature extraction                  │
│   → episode-local state estimation                  │
│   → frozen specialist model(s)                      │
│   → combination / calibration                       │
│   → probability validation                          │
│   → submission.csv                                  │
└────────────────────────────────────────────────────┘
```

## Research hypotheses to investigate

These are hypotheses, not established facts. Each needs a controlled
experiment.

1.  **Population structure:** machine type and signal may explain
    meaningful differences in payout distributions.
2.  **Individual history:** a machine's own observed payouts may improve
    its estimate, with reliability depending on history length.
3.  **Recency:** recent observations may be more informative than older
    observations.
4.  **Temporal structure:** payout sequences or lagged outcomes may
    contain information beyond simple frequencies.
5.  **Shared episode conditions:** other machines' earlier observations
    may reveal a common hidden regime.
6.  **Regime changes:** the data-generating behavior may change over
    time.
7.  **Adaptive combination:** the best blend of population, individual,
    temporal, and episode-level evidence may depend on data quality and
    context.

## Initial modeling strategy

Start with interpretable baselines and grow only when evidence supports
complexity:

-   smoothed categorical frequencies and hierarchical shrinkage;
-   logistic regression / multinomial models;
-   random forests or Extra Trees;
-   gradient boosting;
-   XGBoost multiclass probability prediction;
-   specialist ensembles or gating only after specialists prove useful.

Use pandas and NumPy for data work, scikit-learn for
baselines/evaluation, and XGBoost for tabular nonlinear models.
Spark/Databricks are not initial dependencies; the dataset and offline
runtime constraints do not justify adding distributed infrastructure
without evidence.

## Evaluation principles

-   Hold out entire episodes when testing generalization to new
    episodes.
-   Keep all feature generation cutoff-safe.
-   Treat missing ticks as unknown, not zero.
-   For cross-machine features, use only same-episode observations
    available before the target cutoff.
-   Never combine histories across unrelated episodes.
-   Compare probability quality using the official metric.
-   Track calibration and per-class performance, especially rare
    outcomes.
-   Run ablations and retain negative results.
-   Use the official validation data carefully; do not let repeated
    tuning turn it into a memorized target.

## Repository map

``` text
house-edge-lab/
├── README.md                 # this overview and onboarding guide
├── LAB_START.md              # how to start the research loop
├── PROGRESS_LOG.md           # living progress and experiment record
├── COPY_FROM_STARTER_KIT.md  # starter-kit copy checklist
├── starter_kit/              # untouched organizer kit
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
└── submission/
```

## Onboarding: first 60--90 minutes

1.  Read the organizer's rules and reproduction requirements.
2.  Inspect `COPY_FROM_STARTER_KIT.md`; copy the original kit without
    changing its structure.
3.  Read the data dictionary and identify the exact query, history, and
    target files.
4.  Create an isolated Python environment.
5.  Run the official starter baseline unchanged.
6.  Record the baseline command, dependencies, runtime, metrics, and
    outputs in `PROGRESS_LOG.md`.
7.  Inspect the episode and machine identifiers, time/cutoff columns,
    missingness, and target format.
8.  Implement episode-grouped cross-validation.
9.  Run one simple experiment and record the result before adding more
    agents.

## Contribution protocol

Every new idea must arrive with:

-   a clear hypothesis;
-   a reason it might improve the official metric;
-   the exact data and cutoff assumptions;
-   an experiment plan and comparison baseline;
-   expected runtime;
-   a measurable keep/revise/reject criterion.

Every code change must include a reproducible command and a recorded
result. Never report an experiment as successful based only on an
agent's explanation.

## Competition constraints

The final submission must respect the organizer's current instructions.
The challenge materials discussed so far specify offline execution,
Python 3.11+, two CPU cores, 16 GiB RAM, and a 30-minute runtime limit.
Verify these against the latest official kit before final packaging.

The submission must include everything needed to reproduce predictions:
entry-point script, imported modules, dependency specification,
configuration, and any required model artifacts. The final code must not
depend on local absolute paths, network services, or files that will not
exist in the organizer's runtime.

## Success criterion

A successful lab is not the one with the most agents or experiments. It
is the one that converts good questions into reliable evidence, and
reliable evidence into a stronger predictor that can be reproduced
exactly.
