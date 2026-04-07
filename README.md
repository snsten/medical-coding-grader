---
title: Medical Coding Auditor
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
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
- **Hallucination resistance** — penalizing references to codes outside the proposed set

---

## Action Space

The agent selects one action per step from the following four tools:

| `action_type` | Required Fields | Description |
|---|---|---|
| `query_guideline` | `code` | Look up official ICD-10-CM or CPT coding guidelines, Excludes1/2 notes, and gender restrictions for a specific code. |
| `check_ncci_edits` | `code1`, `code2` | Check the CMS NCCI PTP edit table to determine if two CPT codes have a bundling conflict (mutually exclusive). |
| `flag_error` | `code`, `error_type`, `justification` | Record a confirmed coding error in the draft audit report. |
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
| `difficulty` | `str` | `easy`, `medium`, or `hard`. |
| `patient_demographics` | `dict` | Age, sex, MRN, insurance carrier. |
| `clinical_note` | `str` | Unstructured clinical note documenting the encounter. |
| `proposed_codes` | `dict[code → {description, code_type}]` | The billing codes to audit. |
| `draft_report` | `list[{code, error_type, justification, step}]` | Errors flagged so far. |
| `tool_result` | `str` | Text output of the last action (guideline text, NCCI result, etc.). |
| `step_count` | `int` | Steps taken this episode. |
| `codes_queried` | `list[str]` | Codes already queried (for loop detection). |
| `pairs_checked` | `list[str]` | NCCI pairs already checked. |
| `last_action_error` | `str \| null` | Error message if the last action was invalid. |
| `grader_score` | `float \| null` | Final grader score (set only after `submit_audit`). |
| `reward` | `float` | Reward for the last action. |
| `done` | `bool` | Whether the episode has ended. |

---

## Tasks

### Task 1 — Easy: Demographic Mismatch (`easy_demographic`)

**Scenario:** A 34-year-old **male** patient's annual physical exam chart contains a maternity-specific ICD-10 code (`O80 — Encounter for full-term uncomplicated delivery`) alongside valid codes `E11.9` and `Z00.00`.

**Objective:** Query the guideline for `O80`, recognize the female-only restriction, and flag the demographic mismatch before submitting the audit.

**Expected error:** `O80` → `demographic_mismatch`

**Max steps:** 10 | **Baseline score:** ~0.75

---

### Task 2 — Medium: NCCI PTP Bundling Conflict (`medium_ncci_conflict`)

**Scenario:** A 67-year-old female's cardiology billing includes `93306` (complete transthoracic echocardiography) and `93307` (limited TTE), which are subject to a CMS NCCI Procedure-to-Procedure edit — a classic unbundling violation. Also includes a valid `99213` office visit.

**Objective:** Use `check_ncci_edits` to identify the bundling conflict between `93306` and `93307`, then flag the NCCI edit violation.

**Expected error:** `93306` + `93307` → `ncci_edit`

**Max steps:** 15 | **Baseline score:** ~0.65

---

### Task 3 — Hard: 7th Character Specificity + Untraceable Code (`hard_specificity_untraceable`)

**Scenario:** A 28-year-old male presents for a **follow-up visit** (6 weeks post-fracture). The proposed codes include:
- `S52.501A` — uses 7th character `A` (initial encounter) for what the clinical note explicitly documents as a subsequent/follow-up encounter (should be `S52.501D`).
- `Z99.999` — a code that does not exist in any official ICD-10-CM code set.

**Objective:** Read the clinical note carefully to identify the incorrect 7th character, verify `Z99.999` is untraceable via `query_guideline`, and flag both errors.

**Expected errors:** `S52.501A` → `specificity_error` AND `Z99.999` → `untraceable_code`

**Max steps:** 20 | **Baseline score:** ~0.55

---

## Reward Function

The environment provides **dense rewards** throughout each episode:

| Action | Condition | Reward |
|---|---|---|
| `query_guideline` | Code is in the proposed set (first time) | `+0.10` |
| `query_guideline` | Code NOT in proposed set (hallucination) | `−0.50` |
| `query_guideline` | Code already queried this episode (loop) | `−0.10` |
| `check_ncci_edits` | Both codes in proposed set (first time) | `+0.05 / +0.10` |
| `check_ncci_edits` | Any code NOT in proposed set | `−0.50` |
| `flag_error` | Correct code + `ncci_edit` or `excludes1_conflict` | `+0.30` |
| `flag_error` | Correct code + other error types | `+0.20` |
| `flag_error` | Correct code + wrong error_type | `0.00` |
| `flag_error` | False positive (code has no expected error) | `−0.20` |
| `flag_error` | Code NOT in proposed set | `−0.50` |
| `submit_audit` | Terminal reward = grader score | `[0.0 – 1.0]` |

### Grader

The grader runs deterministically at `submit_audit`:

- **Full credit (1.0×):** correct code + correct `error_type`
- **Partial credit (0.5×):** correct code + wrong `error_type`
- **No credit (0.0×):** code not flagged at all
- **Score = `sum(credits) / total_expected_errors`** → range `[0.0, 1.0]`

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
python -m server.app
# or with uv:
# uv run server

# Validate the environment
openenv validate .
```

### Docker

```bash
cd medical_coding_env

# Build
docker build -f server/Dockerfile -t medical-coding-env:latest .

# Run
docker run -p 7860:7860 medical-coding-env:latest
```

### Running Inference

```bash
# Set environment variables
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:7860"   # where the server is running

# Run baseline
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
result = resp.json()
print(result["observation"]["tool_result"])
```

---

## Baseline Scores

Evaluated with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Difficulty | Score |
|---|---|---|
| `easy_demographic` | Easy | ~0.75 |
| `medium_ncci_conflict` | Medium | ~0.65 |
| `hard_specificity_untraceable` | Hard | ~0.55 |
| **Average** | — | **~0.65** |

*(Scores are reproducible given the deterministic grader and fixed temperature 0.2)*

---

## Project Structure

```
medical_coding_env/
├── openenv.yaml               # OpenEnv spec manifest
├── pyproject.toml             # Package config + uv/pip dependencies
├── uv.lock                    # Reproducible dependency lockfile
├── models.py                  # Action + Observation Pydantic models
├── client.py                  # EnvClient (typed HTTP/WS client)
├── inference.py               # Baseline inference script
├── README.md                  # This file
├── data/
│   └── ground_truth_cases.json  # ICD-10/CPT guidelines + 3 task scenarios
└── server/
    ├── __init__.py
    ├── environment.py         # MedicalCodingEnvironment (reset/step/state)
    ├── app.py                 # FastAPI app (create_app wrapper)
    └── Dockerfile             # Container build spec
```

---

## License

Open source under the MIT License. The ICD-10-CM codes and NCCI edit examples used in this environment are based on publicly available CMS government data.
