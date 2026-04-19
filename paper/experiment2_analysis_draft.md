# Experiment 2 Analysis — Pre-registration of Interpretations

## Purpose
This document pre-registers how we will interpret Gamma(L) results
before seeing the numbers, to avoid post-hoc rationalization.

---

## Possible Outcomes and Interpretations

### Outcome A: Gamma(L) near zero for short bins, large for long bins
- Interpretation: Confirms H2. Context is the bottleneck specifically
  for long trajectories. Oracle advantage is driven by context
  compression and failure attribution in later steps.
- Expected shape: monotonically increasing, possibly with inflection
  at 11-20 step bin where trajectory state accumulates non-trivially.
- Cite: Agent's Marathon (2025) shows accuracy approaches zero beyond
  120 steps — our Gamma(L) curve shows the complementary view of where
  oracle intervention recovers that lost accuracy.

### Outcome B: Gamma(L) uniformly large across all bins
- Interpretation: Oracle advantage is driven by environmental access
  (alpha_env) rather than context compression per se. Even on short
  tasks, the agent's observations are unreliable enough that ground-
  truth DOM access provides substantial benefit.
- Implication: The bottleneck is observation fidelity, not context
  length. Partially undermines H2 but strengthens alpha_env finding
  from Experiment 1.
- Cross-check: Verify with Experiment 1 alpha_env values. If alpha_env
  is large, Outcome B is expected and consistent.

### Outcome C: Gamma(L) large only for middle bins (11-40), drops for 41-80
- Interpretation: For very long trajectories, even oracle context
  injection is insufficient — capability limitations dominate.
  The agent cannot recover even with perfect context.
- Implication: There is a length threshold beyond which alpha_capability
  becomes the primary failure mode regardless of context quality.
- Action: Report as a limitation. Distinguish from Outcome A by checking
  raw accuracy in the 41-80 bin — if near zero, this is a floor effect.

### Outcome D: Gamma(L) non-monotonic or noisy
- Interpretation: Likely insufficient data per bin. Check N per bin.
  If any bin has fewer than 20 tasks, results are unreliable.
- Action: Merge bins or report with wider confidence intervals.
  Flag as preliminary.

---

## Category-Level Analysis

For each of the 4 WebArena categories (Shopping, Reddit, GitLab, CMS):
- Expect GitLab to show highest Gamma(L) in long bins — form-heavy
  tasks with many intermediate steps where context accumulates fastest
- Expect Shopping to show lowest Gamma(L) — tasks often complete in
  fewer steps with less state to track
- If category patterns diverge strongly from aggregate, report
  category breakdown prominently

---

## Statistical Tests

For each bin: two-proportion z-test comparing raw vs. oracle accuracy.
Report p-values. Apply Bonferroni correction for 4 bins (threshold p < 0.0125).
Report effect size (Cohen's h) alongside p-values.

---

## Connection to Other Experiments

- If Gamma(L) grows with L: consistent with delta_synth(L) also growing
  (Experiment 3) — both measure context degradation at scale
- If Gamma(L) is flat: check delta_synth(L) — if also flat, context is
  not the bottleneck at any length; if delta_synth grows, env access
  is the driver not context length
- Ablation cross-check: if F_t ablation shows largest Delta_f, then
  the mechanism driving Gamma(L) is failure attribution, not goal tracking

---

## Key Numbers to Track When Results Arrive

| Metric | Expected | Actual | Interpretation |
|--------|----------|--------|----------------|
| Gamma(L<=10) | ~0.05 | TBD | Baseline — should be small |
| Gamma(11-20) | ~0.10 | TBD | Early context effects |
| Gamma(21-40) | ~0.20 | TBD | Main effect bin |
| Gamma(41-80) | ~0.30 | TBD | Full context degradation |
| N per bin | >=20 | TBD | Reliability check |
| GitLab Gamma | Highest | TBD | Category validation |
| Shopping Gamma | Lowest | TBD | Category validation |
