# House Edge Lab --- Progress & Learning Log

Use this as the lab's living notebook. Update it after every meaningful
experiment or architectural decision. Keep the record concise but
reproducible.

## 0. Project status

-   **Current phase:** Setup / baseline / data audit / feature research
    / model research / integration / final validation
-   **Current best model:**
-   **Best cross-validation score:**
-   **Official validation score:**
-   **Last reproducible run ID:**
-   **Known blockers:**
-   **Next highest-value action:**

## 1. Setup checklist

-   [ ] Original starter ZIP copied into `starter_kit/`
-   [ ] Starter-kit file inventory verified
-   [ ] Official rules and data dictionary read
-   [ ] Python version recorded
-   [ ] Isolated environment created
-   [ ] Dependencies installed from an explicit specification
-   [ ] Starter baseline runs unchanged
-   [ ] Baseline output schema verified
-   [ ] Baseline metrics recorded
-   [ ] Episode-grouped validation strategy implemented
-   [ ] Reproduction command tested

### Environment record

  Item                 Value
  -------------------- -------
  OS                   
  Python               
  pandas               
  NumPy                
  scikit-learn         
  XGBoost              
  Other dependencies   
  CPU / RAM            
  Install command      

## 2. Data map

Record exact filenames, columns, row counts, and semantics after
inspecting the kit. Do not guess.

  -----------------------------------------------------------------------------------
  Dataset/file   Purpose               Rows Key           Target/labels   Notes
                                            identifiers                   
  -------------- ----------- -------------- ------------- --------------- -----------
  Train                                                                   

  Validation                                                              

  Test inputs                                                             
  -----------------------------------------------------------------------------------

### Data semantics to verify

-   [ ] What uniquely identifies a query?
-   [ ] What identifies a machine?
-   [ ] What identifies an episode?
-   [ ] What defines the cutoff tick?
-   [ ] Are payout histories ordered?
-   [ ] How are missing ticks represented?
-   [ ] Which observations are available at inference?
-   [ ] Which fields are absent at test time?
-   [ ] What is the exact target meaning?
-   [ ] What is the official scoring metric?

## 3. Hypothesis backlog

Status values: `NEW`, `PLANNED`, `RUNNING`, `SUPPORTED`, `REJECTED`,
`INCONCLUSIVE`, `PARKED`.

  ---------------------------------------------------------------------------------------
  ID          Hypothesis      Why plausible      Test             Status      Evidence /
                                                                              next step
  ----------- --------------- ------------------ ---------------- ----------- -----------
  H-001       Machine type    Different machine  Type-only        NEW         
              predicts        categories may     baseline                     
              different       have distinct                                   
              payout          behavior                                        
              distributions                                                   

  H-002       Signal          Signal effect may  Type × signal    NEW         
              interacts with  vary by type       comparison                   
              machine type                                                    

  H-003       Individual      Observed payouts   Compare          NEW         
              history         reveal             population vs                
              improves        machine-specific   individual                   
              prediction      behavior           estimates                    

  H-004       Longer history  Estimates become   History-length   NEW         
              deserves more   more stable with   stratification /             
              weight          observations       shrinkage                    

  H-005       Recent          Behavior may drift Compare rolling  NEW         
              observations    or be              windows and                  
              matter more     state-dependent    decay                        

  H-006       Lag/sequence    Order may matter   Lag and sequence NEW         
              features add    beyond frequencies ablation                     
              information                                                     

  H-007       Other machines  Machines may share Same-episode,    NEW         
              reveal shared   hidden conditions  past-only                    
              episode state                      features                     

  H-008       Behavior        A single           Change-point /   NEW         
              changes by      stationary         rolling                      
              regime          distribution may   diagnostics                  
                              be inadequate                                   

  H-009       A gated         Evidence quality   Compare fixed    NEW         
              ensemble beats  varies by query    and learned                  
              one global                         mixtures                     
              model                                                           
  ---------------------------------------------------------------------------------------

## 4. Experiment registry

Create one entry per run. Never overwrite the only copy of a previous
successful run.

### Experiment template

#### EXP-\_\_\_\_ --- \[short title\]

-   **Date:**
-   **Owner / agent:**
-   **Status:** Planned / Running / Complete
-   **Hypothesis:**
-   **Question being answered:**
-   **Data split / episode folds:**
-   **Features:**
-   **Model / configuration:**
-   **Baseline being compared:**
-   **Random seed(s):**
-   **Command:**
-   **Code revision / hash:**
-   **Runtime / memory:**
-   **Official metric or CV metric:**
-   **Per-class diagnostics:**
-   **Calibration diagnostics:**
-   **Failure cases:**
-   **Leakage checks:**
-   **Result:** Keep / Revise / Reject / Inconclusive
-   **Interpretation:**
-   **Next experiment:**
-   **Artifact paths:**

------------------------------------------------------------------------

## 5. Run summary table

  ---------------------------------------------------------------------------------
  Run ID     Hypothesis / Fold       Model           Metric      Runtime Decision
             change       strategy                                       
  ---------- ------------ ---------- --------- ------------ ------------ ----------
  BASE-001   Untouched    Official   Starter                             
             starter      starter                                        
             baseline                                                    

                                                                         
  ---------------------------------------------------------------------------------

## 6. Learning history / architecture decisions

Record only durable discoveries and decisions here. Include the
experiment IDs supporting each one.

  --------------------------------------------------------------------------
  Date           Decision or    Supporting     Confidence     Consequence
                 discovery      experiments                   
  -------------- -------------- -------------- -------------- --------------
                                                              

  --------------------------------------------------------------------------

### Current architecture

``` text
Input
  → schema and cutoff validation
  → feature extraction
  → [population specialist]
  → [individual-history specialist]
  → [temporal specialist]
  → [episode-context specialist]
  → combination / calibration
  → probability validation
  → submission
```

Replace bracketed placeholders only when supported by experiments.

## 7. Agent activity log

Agents should not simply say "this feature helps." Require a run ID and
evidence.

  ---------------------------------------------------------------------------
  Date        Agent / role   Task        Output      Experiment   Accepted?
                                         produced    IDs          
  ----------- -------------- ----------- ----------- ------------ -----------
              Data auditor                                        

              Temporal                                            
              specialist                                          

              Episode                                             
              specialist                                          

              Evaluation                                          
              critic                                              

              Orchestrator                                        
  ---------------------------------------------------------------------------

## 8. Leakage and validity checklist

Before accepting any result:

-   [ ] Split is grouped by episode where appropriate.
-   [ ] No training feature uses information after its historical
    cutoff.
-   [ ] No future payout is used.
-   [ ] No observations from unrelated episodes are mixed.
-   [ ] Cross-machine features use only same-episode, earlier
    observations.
-   [ ] Missing ticks are not converted into zero payouts.
-   [ ] Preprocessing is fitted only on the training fold.
-   [ ] Validation targets are not accidentally used as training labels.
-   [ ] Feature availability matches test-time availability.
-   [ ] The comparison uses the official metric or a justified proxy.
-   [ ] Result is reproducible from the recorded command and
    configuration.

## 9. Finalization checklist

-   [ ] Select the best validated architecture.
-   [ ] Remove unused research and agent dependencies from inference.
-   [ ] Freeze model parameters and feature definitions.
-   [ ] Train final model only on data permitted by the rules.
-   [ ] Save all required model artifacts.
-   [ ] Verify class order `[0, 1, 2, 5, 20]`.
-   [ ] Verify all probabilities are finite and in `[0, 1]`.
-   [ ] Verify each row sums to one within tolerance.
-   [ ] Verify query IDs and row ordering.
-   [ ] Run from a clean environment with networking disabled.
-   [ ] Test the exact organizer reproduction command.
-   [ ] Compare regenerated predictions with submitted CSV.
-   [ ] Check runtime, memory, and package contents.
-   [ ] Archive source, config, dependencies, hashes, and final metrics.

## 10. Session handoff

At the end of each work session, fill this in:

**What changed?**

**What did we learn?**

**Which experiment IDs support it?**

**What remains uncertain?**

**What should the next session do first?**

**Exact command to resume:**
