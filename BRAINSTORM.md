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

## Implemented: D. Randomized Case Generation

Added `data/case_generator.py` — procedural case generator that composes error templates from the existing guideline data.

- **Seeded RNG**: deterministic from episode_id, so grading is reproducible
- **Difficulty targeting**: `task_id="random_easy"`, `random_medium"`, `random_hard"`, `random_expert"`, or plain `"random"`
- **Error templates**: demographic_mismatch, excludes1 (COPD/asthma and DM), NCCI (echo and E/M), specificity, untraceable
- **Filler codes**: valid codes added as false-positive traps
- **Clinical note assembly**: section templates composed per error type + generic exam/plan

Difficulty → error_count mapping:
| Difficulty | Errors | Filler Codes | Max Steps |
|---|---|---|---|
| easy | 1 | 1-2 | 10 |
| medium | 1 | 1-2 | 15 |
| hard | 2 | 1-2 | 20 |
| expert | 3 | 1-3 | 25 |

---

## Implemented: E. Partial Observability Mode

Activated via `reset(partial_observability=True)`.

- Clinical note is parsed into sections (REASON FOR VISIT, HPI, PMH, EXAM, ASSESSMENT, etc.)
- Initially only the REASON FOR VISIT section is visible
- Each `query_guideline` call reveals the next section
- Agent must actively investigate to gain information — models real chart review
- Hidden section count shown: `[N more section(s) not yet revealed]`

---

## Implemented: F. Investigation Strategy Rewards

Tracks the agent's action sequence to reward coherent investigation strategies.

**Per-step signals:**
- **+0.05 coherence bonus**: flag_error on a code that was previously queried via query_guideline (or check_ncci_edits for NCCI errors)
- **-0.05 no-investigation penalty**: flag_error on a code without prior investigation

**Terminal signal:**
- **+0.05 thorough review bonus**: awarded at submit if all proposed codes were investigated (queried or NCCI-checked)

This incentivizes the pattern: query all codes → analyze results → flag errors → submit, rather than random guessing.

---

## Priority Order (A–F: DONE)
1. **A** (reward shaping) — DONE
2. **B** (justification quality) — DONE
3. **C** (more tasks) — DONE. 5 fixed tasks.
4. **D** (random cases) — DONE. Procedural generation with 7 error templates.
5. **E** (partial observability) — DONE. Progressive note reveal.
6. **F** (investigation strategy) — DONE. Coherence bonuses + thorough review reward.

---
---

# Phase 2: Research-Driven Improvements

Based on analysis of `docs/OpenEnv for Medical Coding LLMs.md` — a comprehensive blueprint
referencing CLH (Code Like Humans) framework, HERON hierarchical rewards, Med-PRM,
HiCu curriculum learning, and RLAIF synthetic data generation.

## Gap Analysis: Current Implementation vs. Research Recommendations

| Research Recommendation | Our Current State | Gap Severity |
|---|---|---|
| CLH 4-pillar action space (extract → index → validate → reconcile) | Only validate + reconcile (query, check, flag, submit) | Medium |
| `ask_clarifying_question` action for ambiguous notes | No disambiguation mechanic | High — creativity differentiator |
| `extract_evidence` action (span highlighting) | No evidence extraction step | Medium |
| Hierarchical reward (HERON) — ICD tree distance for near-misses | Binary match/no-match grading | High — novel reward design |
| Working memory in observation (`current_working_memory`) | Only `codes_queried` + `pairs_checked` lists | Low |
| Curriculum learning (HiCu) — adaptive difficulty | Random generator picks difficulty, no adaptation | Medium |
| Asymmetric / frequency-weighted rewards for rare codes | Flat reward regardless of code rarity | Medium |
| RLAIF synthetic permutations (demographics, typos, symptom order) | Static templates in case_generator.py | Low |
| Agentic evaluation metrics (trajectory stats, tool failure rate) | Only grader_score scalar | Medium |
| ICD-11 post-coordination (stem + extension cluster coding) | ICD-10-CM only | Low (out of scope for hackathon) |

---

## G. Add `ask_clarifying_question` Action (HIGH PRIORITY)

**What**: New action_type that lets the agent pause auditing to ask a simulated physician
for missing information. The environment responds with a deterministic answer from a
pre-defined clarification pool attached to each task.

**Why**: The research document calls this "the ultimate differentiator between a rudimentary
automated text processor and a true AI co-pilot." Judges will notice this as a novel mechanic
that no toy environment has. It directly maps to real-world medical coding workflow where
coders halt and issue physician queries when documentation is insufficient.

**Implementation plan**:
1. Add `"ask_clarifying_question"` to `action_type` Literal in `models.py`
2. Add optional `question: str` field to `MedicalCodingAction`
3. Add `clarification_pool` to select tasks in `ground_truth_cases.json` — each entry has:
   - `trigger_keywords`: list of strings the agent's question should match
   - `response`: the physician's clarifying answer
   - `reveals_error`: bool — whether the answer contains info needed to identify an error
4. In `environment.py`, `_handle_ask_clarification`:
   - Match agent's question against `trigger_keywords` using fuzzy keyword overlap
   - Return the matched response as `tool_result`
   - Reward: +0.15 if the question is relevant (matches a trigger), -0.1 if irrelevant
   - Add `clarifications_asked: List[str]` to observation
5. Create 1–2 new tasks with intentionally ambiguous notes that require clarification:
   - e.g., fracture note missing laterality → agent must ask "which side?"
   - e.g., diabetes note without lab confirmation → agent must ask for HbA1c/C-peptide

**Reward signals**:
- +0.15: relevant clarification question (matches trigger keywords)
- -0.10: irrelevant or repetitive clarification question
- +0.10: terminal bonus if agent asked necessary clarifications before flagging

**Files to modify**: `models.py`, `server/environment.py`, `data/ground_truth_cases.json`

---

## H. Add `extract_evidence` Action (MEDIUM PRIORITY)

**What**: New action_type that lets the agent highlight specific text spans from the
clinical note as evidence supporting a coding decision. The environment validates whether
the extracted span is clinically relevant.

**Why**: Maps to CLH Pillar 1 (Evidence Extraction). Forces the agent to ground its
reasoning in specific documentation rather than making unsupported claims. The research
doc emphasizes this prevents hallucination and improves long-tail code accuracy.

**Implementation plan**:
1. Add `"extract_evidence"` to `action_type` Literal in `models.py`
2. Add optional `evidence_text: str` field to `MedicalCodingAction`
3. Add `evidence_spans` to each expected_error in ground_truth — key phrases from the
   clinical note that constitute valid evidence for that error
4. In `environment.py`, `_handle_extract_evidence`:
   - Check if `evidence_text` appears as substring in the clinical note (prevent hallucination)
   - Check if it overlaps with any `evidence_spans` for expected errors
   - Track extracted evidence in `_extracted_evidence: List[Dict]`
   - Reward: +0.05 for relevant evidence, -0.05 for irrelevant or hallucinated text
5. Add `extracted_evidence: List[str]` to `MedicalCodingObservation`
6. Bonus at flag_error: +0.05 if the agent previously extracted evidence for that code

**Reward signals**:
- +0.05: extracted span matches an evidence_span for an expected error
- -0.05: extracted span doesn't appear in clinical note or is irrelevant
- +0.05: flag_error bonus when evidence was previously extracted for that code

**Files to modify**: `models.py`, `server/environment.py`, `data/ground_truth_cases.json`

---

## I. Hierarchical Reward — HERON-style ICD Tree Distance (HIGH PRIORITY)

**What**: When the agent flags a code with the wrong error_type, compute partial credit
based on how "close" the error classification is in a conceptual hierarchy, rather than
binary 0/wrong. Also, apply this to the grader for near-miss code identification.

**Why**: The HERON framework shows that predicting a sibling code is "less wrong" than
predicting a code from a different chapter. Our current grader gives 0.5x partial credit
for correct-code-wrong-type, but all wrong types are equally penalized. Some misclassifications
are closer than others (e.g., `excludes1_conflict` vs `ncci_edit` are both "conflict" types,
while `demographic_mismatch` vs `untraceable_code` are very different).

**Implementation plan**:
1. Define an error_type distance matrix in `environment.py`:
   ```
   Distances (0=same, 1=close, 2=far):
   excludes1_conflict ↔ ncci_edit: 1 (both are "code conflict" types)
   specificity_error ↔ untraceable_code: 1 (both are "code validity" issues)
   demographic_mismatch ↔ anything else: 2 (unique category)
   ```
2. Update `_grade_audit` partial credit: instead of flat 0.5x for wrong type, use
   `0.6 - 0.1 * distance` (so close=0.5, far=0.4)
3. Update `_score_flag` per-step reward: instead of flat -0.1 for wrong type, use
   `-0.05 * distance` (close miss = -0.05, far miss = -0.10)

**Files to modify**: `server/environment.py` only

---

## J. Structured Working Memory in Observation (LOW PRIORITY)

**What**: Add a `working_memory` field to the observation that tracks the agent's
accumulated findings in a structured format — not just which codes were queried, but
what was learned from each query.

**Why**: The research doc recommends `current_working_memory` as a persistent list of
extracted entities and partial findings. This helps agents with limited context windows
maintain state across long episodes without re-reading the full clinical note.

**Implementation plan**:
1. Add `working_memory: List[Dict[str, str]]` to `MedicalCodingObservation`
2. Each entry has: `{"code": str, "finding": str, "step": int}`
3. Populate on each query_guideline with a summary of key findings:
   - Gender restriction detected → `{"code": "O80", "finding": "female_only", "step": 1}`
   - Excludes1 found → `{"code": "J44.1", "finding": "excludes1:J45", "step": 2}`
   - Code invalid → `{"code": "Z99.999", "finding": "untraceable", "step": 3}`
4. Populate on check_ncci_edits with conflict result

**Files to modify**: `models.py`, `server/environment.py`

---

## K. Adaptive Curriculum Learning — HiCu-style (MEDIUM PRIORITY)

**What**: When using `task_id="random"`, the environment tracks the agent's rolling
performance and automatically escalates difficulty as the agent improves.

**Why**: HiCu algorithm research shows that starting with easy cases and progressively
increasing difficulty stabilizes policy gradients and improves convergence. Currently
our random generator picks difficulty uniformly.

**Implementation plan**:
1. Add class-level `_performance_history: Dict[str, List[float]]` to track scores per difficulty
2. In `reset()` when `task_id="random"` (no difficulty suffix):
   - If avg score for current difficulty > 0.8 over last 5 episodes, promote to next level
   - If avg score < 0.3, demote to easier level
   - Start at "easy" by default
3. Add `_curriculum_level: str` to track current level
4. Expose current curriculum level in observation (maybe via tool_result on reset)

**Files to modify**: `server/environment.py`

---

## L. Frequency-Weighted Asymmetric Rewards (MEDIUM PRIORITY)

**What**: Scale reward magnitude based on how common/rare the error type is in real-world
coding. Successfully identifying a rare specificity error yields higher reward than catching
an obvious demographic mismatch.

**Why**: Research doc cites Asymmetric Loss (ASL) — rare code identification should be
disproportionately rewarded to prevent the agent from only learning common patterns.

**Implementation plan**:
1. Define rarity multipliers per error_type:
   ```
   demographic_mismatch: 1.0x (common, easy to catch)
   ncci_edit: 1.2x (moderate rarity)
   excludes1_conflict: 1.2x (moderate rarity)
   specificity_error: 1.5x (hard, subtle)
   untraceable_code: 1.3x (moderate)
   ```
2. Apply multiplier to both per-step flag_error rewards and grader credit
3. This makes the hard task worth more than the easy task even with the same base scores

**Files to modify**: `server/environment.py`

---

## M. Agentic Evaluation Metrics Export (MEDIUM PRIORITY)

**What**: Beyond the scalar grader_score, expose detailed trajectory statistics in the
terminal observation and/or via the `state` property.

**Why**: Research doc recommends: trajectory length, tool invocation failure rates,
context hygiene, hierarchical semantic distance. These metrics allow researchers to
visualize how the agent's reasoning evolves. Judges will notice richer evaluation output.

**Implementation plan**:
1. Add `episode_metrics: Dict[str, Any]` to `MedicalCodingObservation` (populated only on done=True)
2. Track and expose:
   - `trajectory_length`: total steps taken
   - `tool_failure_rate`: fraction of steps that returned errors
   - `investigation_coverage`: fraction of proposed codes that were queried
   - `flag_precision`: true_positives / (true_positives + false_positives)
   - `flag_recall`: true_positives / total_expected_errors
   - `avg_reward_per_step`: mean of all step rewards
   - `investigation_before_flag_rate`: fraction of flags preceded by investigation
   - `clarification_count`: number of clarification questions asked (if feature G implemented)
3. Print these in inference.py alongside the score summary

**Files to modify**: `models.py`, `server/environment.py`, `inference.py`

---

## Phase 2 Priority Order

| Priority | Item | Impact | Effort | Hackathon Value |
|---|---|---|---|---|
| 1 | **G** (clarification action) | High | Medium | Creativity/novelty differentiator |
| 2 | **I** (hierarchical reward) | High | Low | Reward design quality |
| 3 | **M** (evaluation metrics) | Medium | Low | Professional polish |
| 4 | **H** (evidence extraction) | Medium | Medium | CLH framework alignment |
| 5 | **L** (frequency-weighted rewards) | Medium | Low | Reward sophistication |
| 6 | **K** (curriculum learning) | Medium | Medium | Training signal quality |
| 7 | **J** (working memory) | Low | Low | Observation completeness |

**Recommendation**: Implement G + I + M first — they collectively address the three
highest-weighted judging criteria (real-world utility 30%, task/grader quality 25%,
environment design 20%) and the creativity bonus (10%).
