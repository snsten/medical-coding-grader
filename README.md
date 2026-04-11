---
title: Medical Coding Auditor
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# Medical Coding Auditor — OpenEnv Environment

A real-world reinforcement learning environment that simulates a **hospital pre-bill compliance review**. The AI agent acts as a **Medical Coding Auditor**, reviewing proposed ICD-10-CM and CPT billing codes against clinical notes and patient demographics to identify sequencing violations, demographic mismatches, and mutually exclusive code conflicts.

## Motivation

Medical coding errors cost the US healthcare system billions of dollars annually in claim denials, penalties, and compliance audits. A skilled auditor must cross-reference ICD-10-CM guidelines, CMS NCCI (National Correct Coding Initiative) edit tables, and patient demographics — a task ideal for evaluating multi-step reasoning in language model agents.

This environment tests:
- **Demographic awareness** — recognizing codes inapplicable to a patient's sex or age
- **Guideline reasoning** — interpreting official ICD-10-CM Excludes1/2 notes
- **Regulatory knowledge** — applying CMS NCCI PTP (Procedure-to-Procedure) bundling rules
- **7th character specificity** — validating encounter-type extensions for injury codes
- **Evidence grounding** — extracting exact clinical note text to support coding decisions
- **Disambiguation** — asking physician clarification questions when documentation is ambiguous
- **Hallucination resistance** — penalizing references to codes outside the proposed set

---

## Action Space

The agent selects one action per step from the following tools:

| `action_type` | Required Fields | Description |
|---|---|---|
| `query_guideline` | `code` | Look up official ICD-10-CM or CPT coding guidelines, Excludes1/2 notes, and gender restrictions for a specific code. |
| `check_ncci_edits` | `code1`, `code2` | Check the CMS NCCI PTP edit table to determine if two CPT codes have a bundling conflict (mutually exclusive). |
| `flag_error` | `code`, `error_type`, `justification` | Record a confirmed coding error in the draft audit report. |
| `ask_clarifying_question` | `question` | Ask the simulated physician for missing clinical information when the note is ambiguous. Returns a deterministic physician response. |
| `extract_evidence` | `evidence_text` | Highlight an exact text span from the clinical note as documentary evidence for a coding decision. Must be a verbatim substring of the note. |
| `submit_audit` | *(none)* | End the episode and submit the draft report for deterministic grading. |

### `error_type` values for `flag_error`:

| Value | Description |
|---|---|
| `demographic_mismatch` | Code is inapplicable to the patient's sex or age (e.g., maternity code for a male patient). |
| `excludes1_conflict` | Two mutually exclusive ICD-10-CM diagnosis codes are billed together (Excludes1 rule violation). |
| `ncci_edit` | Two CPT codes have a CMS NCCI PTP bundling conflict; the component service is billed separately from its comprehensive code. |
| `specificity_error` | Wrong 7th character extension (e.g., "initial encounter" used for a follow-up visit). |
| `untraceable_code` | The code does not exist in any official ICD-10-CM or CPT code set. |

---

## Observation Space

Each step returns a `MedicalCodingObservation` with the following fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Current task identifier. |
| `difficulty` | `str` | `easy`, `medium`, `hard`, or `expert`. |
| `patient_demographics` | `dict` | Age, sex, MRN, insurance carrier. |
| `clinical_note` | `str` | Unstructured clinical note documenting the encounter. |
| `proposed_codes` | `dict[code → {description, code_type}]` | The billing codes to audit. |
| `draft_report` | `list[{code, error_type, justification, step}]` | Errors flagged so far. |
| `tool_result` | `str` | Text output of the last action (guideline text, NCCI result, physician response, etc.). |
| `step_count` | `int` | Steps taken this episode. |
| `codes_queried` | `list[str]` | Codes already queried via `query_guideline` (loop detection). |
| `pairs_checked` | `list[str]` | NCCI pairs already checked (formatted as `code1\|code2`). |
| `clarifications_asked` | `list[str]` | Clarifying questions asked via `ask_clarifying_question` this episode. |
| `extracted_evidence` | `list[str]` | Text spans extracted from the clinical note via `extract_evidence` this episode. |
| `last_action_error` | `str \| null` | Error message if the last action was invalid. |
| `grader_score` | `float \| null` | Final grader score `[0.0, 1.0]` (set only after `submit_audit`). |
| `episode_metrics` | `dict \| null` | Detailed trajectory stats (populated on `done=True`): trajectory length, tool failure rate, investigation coverage, flag precision/recall, avg reward per step, investigation-before-flag rate, clarification count. |
| `reward` | `float` | Reward for the last action. |
| `done` | `bool` | Whether the episode has ended. |

---

## Tasks

Five tasks spanning four difficulty levels:

### Task 1 — Easy: Demographic Mismatch (`easy_demographic`)

**Scenario:** A 34-year-old **male** patient's annual physical exam chart contains a maternity-specific ICD-10 code (`O80 — Encounter for full-term uncomplicated delivery`) alongside valid codes `E11.9` and `Z00.00`.

**Objective:** Query the guideline for `O80`, recognize the female-only restriction, and flag the demographic mismatch before submitting the audit.

**Expected error:** `O80` → `demographic_mismatch`

**Max steps:** 10

---

### Task 2 — Medium: NCCI PTP Bundling Conflict (`medium_ncci_conflict`)

**Scenario:** A 67-year-old female's cardiology billing includes `93306` (complete transthoracic echocardiography) and `93307` (limited TTE), which are subject to a CMS NCCI Procedure-to-Procedure edit — a classic unbundling violation. Also includes a valid `99213` office visit.

**Objective:** Use `check_ncci_edits` to identify the bundling conflict, then flag the violation.

**Expected error:** `93306` → `ncci_edit`

**Max steps:** 15

---

### Task 3 — Medium: Excludes1 Conflict (`medium_excludes1`)

**Scenario:** A 58-year-old female presents with an acute COPD exacerbation. The proposed codes list both `J44.1` (COPD with exacerbation) and `J45.20` (mild intermittent asthma), which have an Excludes1 (mutually exclusive) relationship. A valid `Z87.891` history code is present as a false-positive trap.

**Objective:** Query guidelines to identify the Excludes1 restriction, read the clinical note to confirm the asthma diagnosis was superseded, and flag the conflict.

**Expected error:** `J44.1` → `excludes1_conflict`

**Max steps:** 15

---

### Task 4 — Hard: 7th Character Specificity + Untraceable Code (`hard_specificity_untraceable`)

**Scenario:** A 28-year-old male presents for a **follow-up visit** (6 weeks post-fracture). The proposed codes include:
- `S52.501A` — uses 7th character `A` (initial encounter) for what the clinical note explicitly documents as a subsequent/follow-up encounter (should be `S52.501D`).
- `Z99.999` — a code that does not exist in any official ICD-10-CM code set.

**Objective:** Read the clinical note carefully, verify `Z99.999` is untraceable, and flag both errors.

**Expected errors:** `S52.501A` → `specificity_error` AND `Z99.999` → `untraceable_code`

**Max steps:** 20

---

### Task 5 — Expert: Multi-Error Complex Encounter (`expert_multi_error`)

**Scenario:** A 65-year-old male has a comprehensive cardiology + endocrinology visit. The proposed 6 codes contain two distinct errors — an Excludes1 conflict between Type 1 and Type 2 diabetes (`E10.9` + `E11.9`) and an NCCI PTP bundling violation between echocardiography codes (`93306` + `93308`). `99214` and `M79.621` are valid and must NOT be flagged.

**Objective:** Identify both errors without false-positiving the valid codes.

**Expected errors:** `E10.9` → `excludes1_conflict` AND `93306` → `ncci_edit`

**Max steps:** 25

---

## Reward Function

The environment provides **dense, multi-signal rewards** throughout each episode:

| Action | Condition | Reward |
|---|---|---|
| `query_guideline` | Valid code, first query | `+0.10` |
| `query_guideline` | Code NOT in proposed set (hallucination) | `−0.50` |
| `query_guideline` | Code already queried (loop) | `−0.10` |
| `check_ncci_edits` | Valid pair, first check — no edit found | `+0.05` |
| `check_ncci_edits` | Valid pair, first check — edit found | `+0.10` |
| `check_ncci_edits` | Any code NOT in proposed set | `−0.50` |
| `ask_clarifying_question` | Relevant question (matches physician pool) | `+0.15` |
| `ask_clarifying_question` | Irrelevant or repeated question | `−0.10` |
| `extract_evidence` | Span is in note AND matches expected evidence | `+0.05` |
| `extract_evidence` | Span is in note but irrelevant | `±0.00` |
| `extract_evidence` | Span NOT in clinical note (hallucination) | `−0.05` |
| `flag_error` | Correct code + correct `error_type` (conflict type) | `+0.30 × rarity` |
| `flag_error` | Correct code + correct `error_type` (other) | `+0.20 × rarity` |
| `flag_error` | Correct code + prior evidence extracted | additional `+0.05` |
| `flag_error` | Correct code + justification contains key terms | additional `+0.05` |
| `flag_error` | Correct code + investigated before flagging | additional `+0.05` |
| `flag_error` | Correct code + no prior investigation | `−0.05` |
| `flag_error` | Correct code + wrong error_type (close miss) | `−0.05` |
| `flag_error` | Correct code + wrong error_type (far miss) | `−0.10` |
| `flag_error` | False positive (code has no expected error) | `−0.20` |
| `flag_error` | Code NOT in proposed set | `−0.50` |
| `submit_audit` | Terminal: grader score − FP penalty + efficiency bonus | `[0.0 – 1.0]` |

**Rarity multipliers** (Asymmetric Loss — rare errors rewarded more):
`demographic_mismatch=1.0×`, `ncci_edit/excludes1_conflict=1.2×`, `untraceable_code=1.3×`, `specificity_error=1.5×`

**HERON-style hierarchical wrong-type penalty:** close misclassifications (e.g. `excludes1_conflict` ↔ `ncci_edit`) penalized less (`−0.05`) than cross-category misclassifications (`−0.10`).

### Grader (terminal)

The grader runs deterministically at `submit_audit`:

- **Full credit (1.0 × rarity):** correct code + correct `error_type`
- **HERON partial credit (0.4–0.55 × rarity):** correct code + wrong `error_type` (scaled by conceptual distance)
- **No credit (0.0):** code not flagged at all
- **Score = `sum(credits) / sum(rarity weights)`** → range `[0.0, 1.0]`
- **Efficiency bonus:** `+0.1 × (max_steps − steps_used) / max_steps`
- **False positive penalty:** `−0.15 × false_positive_count`
- **Thorough review bonus:** `+0.05` if all proposed codes were investigated before submit

---

## Setup & Usage

### Prerequisites

- Python ≥ 3.10
- Docker (for containerized deployment)
- `uv` or `pip` for dependency management

### Local Development

```bash
cd medical_coding_env

# Install dependencies
pip install "openenv-core[core]>=0.2.2" openai

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860
# or with uv:
uv run server

# Validate the environment
openenv validate .
```

### Docker

```bash
cd medical_coding_env

# Build (using root Dockerfile — recommended)
docker build -t medical-coding-env:latest .

# Run
docker run -p 7860:7860 medical-coding-env:latest
```

### Running Inference

```bash
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

### API Usage

```python
import httpx

# Reset to a specific task
resp = httpx.post("http://localhost:7860/reset", json={"task_id": "easy_demographic"})
obs = resp.json()["observation"]
print(obs["clinical_note"])

# Execute an action
resp = httpx.post("http://localhost:7860/step", json={
    "action": {
        "action_type": "query_guideline",
        "code": "O80"
    }
})
print(resp.json()["observation"]["tool_result"])
```

---

## Baseline Scores

Evaluated with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router at temperature 0.2:

| Task | Difficulty | Expected Errors | Score |
|---|---|---|---|
| `easy_demographic` | Easy | 1 | ~0.75 |
| `medium_ncci_conflict` | Medium | 1 | ~0.65 |
| `medium_excludes1` | Medium | 1 | ~0.65 |
| `hard_specificity_untraceable` | Hard | 2 | ~0.55 |
| `expert_multi_error` | Expert | 2 | ~0.40 |
| **Average** | — | — | **~0.60** |

*(Scores are reproducible given the deterministic grader and fixed random seed.)*

---

## Project Structure

```
medical_coding_env/
├── openenv.yaml               # OpenEnv spec manifest
├── pyproject.toml             # Package config + uv/pip dependencies
├── uv.lock                    # Reproducible dependency lockfile
├── Dockerfile                 # Root Dockerfile (build context = project root)
├── .dockerignore              # Excludes .git/, __pycache__, docs/ from image
├── models.py                  # Action + Observation Pydantic models
├── client.py                  # EnvClient (typed HTTP/WS client)
├── inference.py               # Baseline inference script
├── README.md                  # This file
├── data/
│   ├── ground_truth_cases.json  # ICD-10/CPT guidelines + 5 task scenarios
│   └── case_generator.py        # Procedural random case generator (seeded)
└── server/
    ├── __init__.py
    ├── environment.py         # MedicalCodingEnvironment (reset/step/state)
    ├── app.py                 # FastAPI app (create_app wrapper)
    ├── requirements.txt       # Server runtime deps (pip fallback)
    └── Dockerfile             # Multi-stage Dockerfile (openenv-base pattern)
```

---

## License

Open source under the MIT License. The ICD-10-CM codes and NCCI edit examples used in this environment are based on publicly available CMS government data.
