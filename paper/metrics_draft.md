# Experiment 1 Metrics — Draft
## Paper: "Is Context the Bottleneck? Causal Decomposition of Failure in Long-Horizon LLM Agents"

---

## Annotation Agreement (Cohen's Kappa)

### Failure Attribution Agreement
Computed on 9 overlapping trajectories annotated by both Adithya (Annotator 1) and Muhammad (Annotator 2).

**Kappa = 0.3415** (fair agreement)

| Trajectory | Adithya | Muhammad | Agreement |
|------------|---------|----------|-----------|
| task_322 | capability-caused | context-caused | DISAGREE |
| task_332 | capability-caused | context-caused | DISAGREE |
| task_401 | capability-caused | capability-caused | AGREE |
| task_415 | capability-caused | capability-caused | AGREE |
| task_574 | capability-caused | capability-caused | AGREE |
| task_603 | context-caused | context-caused | AGREE |
| task_604 | context-caused | capability-caused | DISAGREE |
| task_606 | context-caused | context-caused | AGREE |
| task_607 | context-caused | context-caused | AGREE |

Agreed: 6/9 (66.7%)
Disagreed: 3/9 (33.3%)

### Disagreement Analysis
Tasks 322 and 332 are Shopping login-loop trajectories where the agent 
repeatedly clicked My Account without authenticating. Adithya classified 
these as capability-caused (agent failed to adapt despite visible 
alternatives). Muhammad classified them as context-caused (agent lacked 
credentials, task structurally unsolvable). Muhammad marked t*=0 for 
both, indicating failure from step 1.

Task 604 disagreement: Adithya saw rate limit error as context-caused. 
Muhammad saw agent choosing wrong forum as the primary failure 
(capability-caused) with rate limit as secondary.

### t* Agreement
| Trajectory | Adithya t* | Muhammad t* | Difference |
|------------|-----------|------------|------------|
| task_322 | 2 | 0 | 2 |
| task_332 | 2 | 0 | 2 |
| task_401 | 4 | 3 | 1 |
| task_415 | 9 | 8 | 1 |
| task_574 | 3 | 4 | 1 |
| task_603 | 6 | 0 | 6 |
| task_604 | 6 | 2 | 4 |
| task_606 | 9 | 0 | 9 |
| task_607 | 6 | 0 | 6 |

Mean t* difference: 3.4 steps
Excluding t*=0 cases: 1.25 steps

---

## Alpha Decomposition
[TBD — pending Rocky's Experiment 1 results]

| Metric | Value | 95% CI |
|--------|-------|--------|
| alpha_context | TBD | TBD |
| alpha_env | TBD | TBD |
| alpha_capability | TBD | TBD |

---

## Ablation Results
[TBD — pending Rocky's Experiment 1 pipeline]

| Field removed | Delta_f | 
|---------------|---------|
| g | TBD |
| P_t | TBD |
| R_t | TBD |
| e_t | TBD |
| C | TBD |
| F_t | TBD |
| K_t | TBD |

---

## Gamma(L) Results
[TBD — pending Experiment 3 data from Muhammad and Adithya]

| Length bin | Raw acc | Oracle acc | Gamma(L) | N tasks |
|------------|---------|------------|----------|---------|
| <=10 | TBD | TBD | TBD | TBD |
| 11-20 | TBD | TBD | TBD | TBD |
| 21-35 | TBD | TBD | TBD | TBD |
| 36-50 | TBD | TBD | TBD | TBD |
| 51-75 | TBD | TBD | TBD | TBD |
| >75 | TBD | TBD | TBD | TBD |
