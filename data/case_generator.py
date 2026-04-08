"""
Randomized Case Generator for the Medical Coding Auditor Environment.

Generates procedurally-created task scenarios by composing error templates
from the existing guideline data. Each generated case is deterministic given
the same seed (derived from episode_id), so grading remains reproducible.

Supports difficulty targeting: random_easy, random_medium, random_hard, random_expert,
or plain "random" for a random difficulty.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Patient templates
# ---------------------------------------------------------------------------

_FIRST_NAMES_M = ["James", "Robert", "Michael", "David", "William", "Thomas", "Richard", "Charles"]
_FIRST_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan", "Karen", "Nancy"]
_LAST_INITIALS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_INSURANCES = [
    "Blue Cross PPO", "Medicare Part B", "Aetna HMO",
    "United Healthcare PPO", "Cigna EPO", "Humana Gold Plus",
    "Anthem Blue Shield", "Tricare Standard",
]


def _make_patient(rng: random.Random, sex: Optional[str] = None) -> Dict[str, Any]:
    """Generate a random patient with demographics."""
    if sex is None:
        sex = rng.choice(["male", "female"])
    age = rng.randint(22, 78)
    first = rng.choice(_FIRST_NAMES_M if sex == "male" else _FIRST_NAMES_F)
    last_init = rng.choice(_LAST_INITIALS)
    return {
        "age": age,
        "sex": sex,
        "mrn": f"MRN-GEN-{rng.randint(1000, 9999)}",
        "insurance": rng.choice(_INSURANCES),
        "_name": f"{first} {last_init}.",
    }


# ---------------------------------------------------------------------------
# Error templates — each returns (proposed_codes_fragment, expected_error, note_snippet)
# ---------------------------------------------------------------------------

def _error_demographic_mismatch(
    rng: random.Random, patient: Dict[str, Any], data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any], str]:
    """O80 on a male patient."""
    patient["sex"] = "male"  # force male for this error
    codes = {
        "O80": {"description": "Encounter for full-term uncomplicated delivery", "code_type": "ICD-10-CM"},
    }
    error = {
        "code": "O80",
        "error_type": "demographic_mismatch",
        "description": "O80 is a maternity code applicable only to female patients. Cannot be assigned to a male patient.",
        "key_terms": ["female", "male", "maternity", "obstetric", "delivery", "sex", "gender", "demographic"],
    }
    title = "Mr." if patient["sex"] == "male" else "Ms."
    note = (
        f"HISTORY OF PRESENT ILLNESS: {title} {patient['_name']} is a "
        f"{patient['age']}-year-old {patient['sex']} presenting for a routine wellness visit. "
        f"No obstetric or gynecological history is documented."
    )
    return codes, error, note


def _error_excludes1_copd_asthma(
    rng: random.Random, patient: Dict[str, Any], data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any], str]:
    """J44.1 + J45.20 Excludes1 conflict."""
    codes = {
        "J44.1": {"description": "COPD with acute exacerbation", "code_type": "ICD-10-CM"},
        "J45.20": {"description": "Mild intermittent asthma, uncomplicated", "code_type": "ICD-10-CM"},
    }
    error = {
        "code": "J44.1",
        "error_type": "excludes1_conflict",
        "conflicting_code": "J45.20",
        "description": "J44.1 (COPD) has Excludes1 for J45 (asthma). Mutually exclusive — cannot code together.",
        "key_terms": ["Excludes1", "mutually exclusive", "asthma", "COPD", "J45", "J44", "cannot be coded together"],
    }
    note = (
        f"HISTORY OF PRESENT ILLNESS: Patient has a 15-year history of COPD (GOLD stage II). "
        f"Presenting with acute exacerbation — increased dyspnea and sputum. "
        f"Prior asthma label from primary care was superseded by COPD diagnosis after PFTs "
        f"confirmed irreversible obstruction. Patient does NOT carry a concurrent asthma diagnosis.\n\n"
        f"SPIROMETRY: FEV1 48% predicted, post-bronchodilator improvement <10% (consistent with COPD)."
    )
    return codes, error, note


def _error_excludes1_diabetes(
    rng: random.Random, patient: Dict[str, Any], data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any], str]:
    """E10.9 + E11.9 Excludes1 conflict."""
    codes = {
        "E10.9": {"description": "Type 1 diabetes mellitus without complications", "code_type": "ICD-10-CM"},
        "E11.9": {"description": "Type 2 diabetes mellitus without complications", "code_type": "ICD-10-CM"},
    }
    error = {
        "code": "E10.9",
        "error_type": "excludes1_conflict",
        "conflicting_code": "E11.9",
        "description": "E10.9 (Type 1 DM) has Excludes1 for E11 (Type 2 DM). Mutually exclusive.",
        "key_terms": ["Excludes1", "mutually exclusive", "Type 1", "Type 2", "E11", "E10", "diabetes"],
    }
    note = (
        f"ENDOCRINOLOGY ASSESSMENT: Patient has Type 2 diabetes mellitus, managed with metformin. "
        f"C-peptide is normal (2.3 ng/mL), GAD65 autoantibodies are negative, "
        f"ruling out autoimmune Type 1 diabetes. Diagnosis is definitively Type 2 only."
    )
    return codes, error, note


def _error_ncci_echo(
    rng: random.Random, patient: Dict[str, Any], data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any], str]:
    """93306 + 93307 or 93306 + 93308 NCCI bundling."""
    limited = rng.choice(["93307", "93308"])
    limited_desc = (
        "Echocardiography, transthoracic, limited study without Doppler"
        if limited == "93307"
        else "Echocardiography, transthoracic, follow-up or limited study"
    )
    codes = {
        "93306": {"description": "Echocardiography, transthoracic, complete with Doppler", "code_type": "CPT"},
        limited: {"description": limited_desc, "code_type": "CPT"},
    }
    error = {
        "code": "93306",
        "error_type": "ncci_edit",
        "conflicting_code": limited,
        "description": f"93306 (complete TTE) and {limited} (limited TTE) are NCCI PTP edit — unbundling violation.",
        "key_terms": ["bundling", "unbundling", "NCCI", "PTP", "complete", "limited", limited, "comprehensive"],
    }
    note = (
        f"SERVICES RENDERED: A complete transthoracic echocardiogram was performed with 2D imaging, "
        f"M-mode, spectral Doppler, and color flow Doppler. LVEF estimated at {rng.randint(40, 60)}%. "
        f"Additionally, a limited echocardiographic study was documented for focused wall motion assessment."
    )
    return codes, error, note


def _error_ncci_em(
    rng: random.Random, patient: Dict[str, Any], data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any], str]:
    """99213 + 99214 E/M unbundling."""
    codes = {
        "99213": {"description": "Office visit, low-moderate complexity", "code_type": "CPT"},
        "99214": {"description": "Office visit, moderate-high complexity", "code_type": "CPT"},
    }
    error = {
        "code": "99213",
        "error_type": "ncci_edit",
        "conflicting_code": "99214",
        "description": "99213 and 99214 are different E/M levels for the same service — NCCI edit, only one should be billed.",
        "key_terms": ["bundling", "NCCI", "E/M", "99213", "99214", "same service", "unbundling"],
    }
    note = (
        f"SERVICES RENDERED: An office visit was performed. Medical decision making was "
        f"moderate to high complexity given the patient's multiple chronic conditions."
    )
    return codes, error, note


def _error_specificity_fracture(
    rng: random.Random, patient: Dict[str, Any], data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any], str]:
    """S52.501A wrong 7th character (should be D for follow-up)."""
    codes = {
        "S52.501A": {
            "description": "Fracture of lower end of right radius, initial encounter",
            "code_type": "ICD-10-CM",
        },
    }
    error = {
        "code": "S52.501A",
        "error_type": "specificity_error",
        "description": "7th character 'A' (initial) is wrong — this is a follow-up visit. Should be S52.501D (subsequent).",
        "key_terms": ["7th character", "subsequent", "initial", "follow-up", "healing", "S52.501D", "phase of care"],
    }
    weeks = rng.randint(4, 12)
    note = (
        f"HISTORY OF PRESENT ILLNESS: Patient presents for follow-up {weeks} weeks after closed "
        f"fracture of distal right radius. Cast was removed previously. X-ray shows routine healing "
        f"with appropriate callus formation.\n\n"
        f"Note: This is a SUBSEQUENT ENCOUNTER for a healing fracture — active treatment concluded {weeks} weeks ago."
    )
    return codes, error, note


def _error_untraceable_code(
    rng: random.Random, patient: Dict[str, Any], data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any], str]:
    """A fabricated ICD-10 code that doesn't exist."""
    fake_codes = ["Z99.999", "E99.99", "J99.999", "M99.999", "R99.999"]
    fake = rng.choice(fake_codes)
    codes = {
        fake: {"description": "Unspecified condition (placeholder)", "code_type": "ICD-10-CM"},
    }
    error = {
        "code": fake,
        "error_type": "untraceable_code",
        "description": f"{fake} does not exist in the official ICD-10-CM tabular list. Untraceable code.",
        "key_terms": ["invalid", "untraceable", "not exist", "not found", "official", "billable"],
    }
    note = ""  # untraceable codes don't need clinical note context
    return codes, error, note


# ---------------------------------------------------------------------------
# Filler codes (valid, no errors — false positive traps)
# ---------------------------------------------------------------------------

_FILLER_CODES: List[Dict[str, Any]] = [
    {"code": "Z00.00", "description": "Encounter for general adult medical examination", "code_type": "ICD-10-CM"},
    {"code": "Z87.891", "description": "Personal history of other specified conditions", "code_type": "ICD-10-CM"},
    {"code": "M79.621", "description": "Pain in right upper arm", "code_type": "ICD-10-CM"},
    {"code": "99213", "description": "Office visit, low-moderate complexity", "code_type": "CPT"},
    {"code": "99214", "description": "Office visit, moderate-high complexity", "code_type": "CPT"},
]


# ---------------------------------------------------------------------------
# Error pools per difficulty
# ---------------------------------------------------------------------------

_ERROR_POOL_EASY = [_error_demographic_mismatch]
_ERROR_POOL_MEDIUM = [_error_excludes1_copd_asthma, _error_excludes1_diabetes, _error_ncci_echo, _error_ncci_em]
_ERROR_POOL_HARD = [_error_specificity_fracture, _error_untraceable_code]

_DIFFICULTY_CONFIG = {
    "easy":   {"error_count": 1, "pools": [_ERROR_POOL_EASY], "filler_count": (1, 2), "max_steps": 10},
    "medium": {"error_count": 1, "pools": [_ERROR_POOL_MEDIUM], "filler_count": (1, 2), "max_steps": 15},
    "hard":   {"error_count": 2, "pools": [_ERROR_POOL_MEDIUM, _ERROR_POOL_HARD], "filler_count": (1, 2), "max_steps": 20},
    "expert": {"error_count": 3, "pools": [_ERROR_POOL_EASY, _ERROR_POOL_MEDIUM, _ERROR_POOL_HARD], "filler_count": (1, 3), "max_steps": 25},
}


# ---------------------------------------------------------------------------
# Clinical note assembly
# ---------------------------------------------------------------------------

def _assemble_clinical_note(
    patient: Dict[str, Any],
    error_snippets: List[str],
    rng: random.Random,
) -> str:
    """Compose a synthetic clinical note from patient info and error-specific snippets."""
    title = "Mr." if patient["sex"] == "male" else "Ms."
    age = patient["age"]
    sex = patient["sex"]

    parts = [
        f"REASON FOR VISIT: Comprehensive evaluation and management.",
        "",
    ]

    # Add error-specific clinical content
    for snippet in error_snippets:
        if snippet:
            parts.append(snippet)
            parts.append("")

    # Generic physical exam
    bp_sys = rng.randint(110, 150)
    bp_dia = rng.randint(70, 95)
    hr = rng.randint(62, 98)
    parts.extend([
        f"PHYSICAL EXAM: Vitals: BP {bp_sys}/{bp_dia}, HR {hr}. "
        f"General: {title} {patient['_name']} is a {age}-year-old {sex} "
        f"in no acute distress. Exam otherwise unremarkable for areas not addressed above.",
        "",
        f"ASSESSMENT AND PLAN: See individual assessments above. "
        f"Return for follow-up as indicated.",
    ])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_random_task(
    data: Dict[str, Any],
    seed: str,
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a random task scenario.

    Args:
        data: The loaded ground_truth_cases.json data (for guideline reference).
        seed: A string seed for deterministic generation (typically episode_id).
        difficulty: One of 'easy', 'medium', 'hard', 'expert', or None (random).

    Returns:
        A task dict in the same format as the fixed tasks in ground_truth_cases.json.
    """
    # Deterministic RNG from seed
    seed_int = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed_int)

    # Pick difficulty
    if difficulty not in _DIFFICULTY_CONFIG:
        difficulty = rng.choice(list(_DIFFICULTY_CONFIG.keys()))
    config = _DIFFICULTY_CONFIG[difficulty]

    # Generate patient
    patient = _make_patient(rng)

    # Select error generators — pick one from each pool, up to error_count
    error_generators = []
    pools = list(config["pools"])
    for _ in range(config["error_count"]):
        if not pools:
            break
        pool = pools.pop(0)
        gen = rng.choice(pool)
        error_generators.append(gen)

    # Generate errors
    proposed_codes: Dict[str, Dict[str, str]] = {}
    expected_errors: List[Dict[str, Any]] = []
    note_snippets: List[str] = []
    used_codes: set = set()

    for gen_fn in error_generators:
        codes_frag, error, snippet = gen_fn(rng, patient, data)
        # Avoid code collisions
        if any(c in used_codes for c in codes_frag):
            continue
        proposed_codes.update(codes_frag)
        used_codes.update(codes_frag.keys())
        expected_errors.append(error)
        note_snippets.append(snippet)

    # Add filler codes (false-positive traps)
    filler_min, filler_max = config["filler_count"]
    n_fillers = rng.randint(filler_min, filler_max)
    available_fillers = [f for f in _FILLER_CODES if f["code"] not in used_codes]
    rng.shuffle(available_fillers)
    for filler in available_fillers[:n_fillers]:
        proposed_codes[filler["code"]] = {
            "description": filler["description"],
            "code_type": filler["code_type"],
        }

    # Assemble clinical note
    clinical_note = _assemble_clinical_note(patient, note_snippets, rng)

    # Build hints
    hints = [
        "This is a procedurally generated case — investigate all proposed codes systematically.",
        f"Expect approximately {len(expected_errors)} error(s) in this case.",
        "Not every code has an error — avoid false positives.",
    ]

    # Remove _name from patient before returning (internal only)
    patient_clean = {k: v for k, v in patient.items() if not k.startswith("_")}

    return {
        "task_id": f"random_{difficulty}_{seed[:8]}",
        "difficulty": difficulty,
        "description": f"Procedurally generated {difficulty} case with {len(expected_errors)} error(s).",
        "scenario": f"A {patient['age']}-year-old {patient['sex']} patient encounter with {len(proposed_codes)} proposed codes.",
        "patient": patient_clean,
        "clinical_note": clinical_note,
        "proposed_codes": proposed_codes,
        "expected_errors": expected_errors,
        "max_steps": config["max_steps"],
        "hints": hints,
    }
