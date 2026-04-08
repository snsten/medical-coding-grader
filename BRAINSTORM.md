# Medical Coding Auditor — Improvement Brainstorm

## Current State Assessment

### Cases (ground_truth_cases.json) — Clinically Correct
- **easy_demographic**: O80 on male patient. Real CMS rule — O-codes (O00–O9A) are obstetrics chapter, female only. Correct.
- **medium_ncci_conflict**: 93306+93307 echo bundling. Real NCCI PTP edit — complete TTE subsumes limited TTE. Correct.
- **hard_specificity_untraceable**: S52.501A wrong 7th char (initial→subsequent) + Z99.999 invalid code. Both clinically accurate per ICD-10-CM 7th char guidelines and Z99 category range. Correct.

All three cases follow real CMS/AMA coding rules and would be recognized by actual medical coders.

### Current Reward Function — Issues
1. **Score normalization is inflated**: `MAX_REWARDS = max_steps * 0.3 + 1.0` assumes every step earns +0.3 (the max), but most steps earn +0.1 (queries). This makes normalized scores artificially low.
2. **No step efficiency incentive**: An agent solving in 3 steps gets the same reward structure as one solving in 10. No incentive for brevity.
3. **No auto-termination at max_steps**: Agent can exceed step budget without consequence.
4. **False positives unpunished in grader**: Per-step -0.2 exists but grader only counts true positives. An agent that flags everything gets full grader credit.
5. **Flat query reward**: +0.1 regardless of whether the code has an error — no information-gain signal.
6. **Wrong error_type gives 0.0 reward**: Should be slightly negative to discourage guessing.

---

## Improvements to Implement Now (Minimal Code Changes)

### A. Reward Shaping Overhaul
1. **Step efficiency bonus**: On submit, add `efficiency_bonus = 0.1 * (max_steps - steps_used) / max_steps`. Rewards faster solves.
2. **False positive penalty in grader**: Deduct `0.15 * num_false_positives` from grader score. Discourages flag-everything strategy.
3. **Wrong error_type → slight negative**: Change from 0.0 to -0.1 to penalize incorrect classification.
4. **Auto-termination**: When step_count reaches max_steps, auto-submit the draft report. Adds a -0.1 "timeout penalty" to signal the agent should have submitted earlier.
5. **Better score normalization in inference.py**: Use `score = grader_score` directly from the terminal observation (it's already [0,1]) instead of summing intermediate rewards with an inflated denominator.

### B. Justification Quality Signal
Add keyword-matching bonus: if `flag_error` justification contains key terms from the expected error description (e.g., "female", "obstetrics", "subsequent encounter", "untraceable"), add +0.05 bonus. This rewards reasoning quality, not just correct classification.

---

## Implemented: C. More Tasks / Scenarios

### medium_excludes1 (new)
- **Patient**: 58F, pulmonary clinic, COPD exacerbation
- **Error**: J44.1 (COPD) + J45.20 (asthma) Excludes1 conflict
- **Trap**: Z87.891 (personal history) is valid — should NOT be flagged
- **Why it's good**: COPD/asthma overlap is one of the most common real-world Excludes1 violations. Clinical note explicitly documents the asthma diagnosis was superseded by COPD — agent must read and reason about it.

### expert_multi_error (new)
- **Patient**: 65M, cardiology + endocrinology, complex multi-system encounter
- **Errors**: E10.9/E11.9 Excludes1 (Type 1 + Type 2 DM) + 93306/93308 NCCI PTP bundling
- **Traps**: 99214 (office visit) and M79.621 (arm pain) are both valid
- **Why it's good**: 6 proposed codes, 2 distinct errors, 2 false-positive traps. Tests comprehensive auditing — agent must not stop after finding the first error. Clinical note has explicit evidence (C-peptide, autoantibody panel) ruling out Type 1 DM.

### Task coverage now: 5 tasks, 4 difficulty levels
| Task | Difficulty | Error Types Tested |
|---|---|---|
| easy_demographic | easy | demographic_mismatch |
| medium_ncci_conflict | medium | ncci_edit |
| medium_excludes1 | medium | excludes1_conflict |
| hard_specificity_untraceable | hard | specificity_error, untraceable_code |
| expert_multi_error | expert | excludes1_conflict, ncci_edit (combined) |

---

## Future Improvements (Not Implementing Now)

### D. Randomized Case Generation
Generate cases programmatically from a larger code pool rather than 5 fixed scenarios. Would improve eval robustness but increases complexity significantly.

### E. Partial Observability Mode
Hide parts of the clinical note until the agent queries specific codes. Would model real-world information gathering more faithfully.

### F. Multi-Turn Conversation History
Track the full reasoning chain and reward coherent investigation strategies (query → analyze → flag → verify → submit) over random exploration.

---

## Priority Order
1. **A** (reward shaping) — DONE. Efficiency bonus, FP penalty, auto-termination, justification quality.
2. **B** (justification quality) — DONE. Key-term matching bonus.
3. **C** (more tasks) — DONE. 2 new tasks (medium_excludes1, expert_multi_error) → 5 total.
4. **D–F** — Aspirational, skip for hackathon
