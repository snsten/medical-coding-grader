"""
Medical Coding Auditor Environment — Core Logic.

The agent plays the role of a Medical Coding Auditor performing a pre-bill review.
It validates a proposed set of ICD-10 and CPT codes against a clinical note and
patient demographics without assigning new codes.

Reward shaping follows the problem specification:
  +0.10  QueryGuideline on a code present in the proposed set
  +0.30  FlagError that correctly identifies an NCCI edit or Excludes1 conflict
  +0.20  FlagError that correctly identifies a demographic mismatch,
         specificity error, or untraceable code
  -0.10  Repeated QueryGuideline / CheckNCCIEdits on an already-queried code/pair
  -0.20  FlagError with incorrect error_type (false positive classification)
  -0.50  Any action that references a code NOT in the proposed set (hallucination)
  +final  grader_score added as terminal reward on SubmitAudit
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MedicalCodingAction, MedicalCodingObservation
except ImportError:
    try:
        from models import MedicalCodingAction, MedicalCodingObservation
    except ImportError:
        from medical_coding_env.models import MedicalCodingAction, MedicalCodingObservation

# ---------------------------------------------------------------------------
# Locate the ground truth data file
# ---------------------------------------------------------------------------
_DATA_FILE = Path(__file__).parent.parent / "data" / "ground_truth_cases.json"


def _load_data() -> Dict[str, Any]:
    with open(_DATA_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Deterministic Grader
# ---------------------------------------------------------------------------

def _grade_audit(
    draft_report: List[Dict[str, Any]],
    expected_errors: List[Dict[str, Any]],
) -> float:
    """
    Compare the agent's submitted audit report against the ground truth.

    Scoring logic:
    - For each expected error, check if the agent correctly flagged it.
    - A flag is "correct" if both the code AND error_type match.
    - Partial credit (0.5x) if the correct code is flagged but with a wrong error_type.
    - Over-flagging (false positives) is not penalized in the final score
      (the per-step -0.20 reward already penalizes false positives during the episode).
    - Final score = sum(credit for each expected error) / len(expected_errors)

    Returns a float in [0.0, 1.0].
    """
    if not expected_errors:
        return 1.0

    flagged_by_code = {}
    for entry in draft_report:
        code = entry.get("code", "")
        if code not in flagged_by_code:
            flagged_by_code[code] = []
        flagged_by_code[code].append(entry.get("error_type", ""))

    total_credit = 0.0
    for expected in expected_errors:
        exp_code = expected["code"]
        exp_error_type = expected["error_type"]

        if exp_code not in flagged_by_code:
            # Code not flagged at all — no credit
            continue

        agent_error_types = flagged_by_code[exp_code]

        if exp_error_type in agent_error_types:
            # Full credit: correct code + correct error_type
            total_credit += 1.0
        else:
            # Partial credit: correct code, wrong error_type classification
            total_credit += 0.5

    return round(total_credit / len(expected_errors), 4)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MedicalCodingEnvironment(Environment):
    """
    Medical Coding Auditor OpenEnv Environment.

    The agent reviews a case containing patient demographics, a clinical note,
    and a proposed set of ICD-10-CM / CPT billing codes. It uses four tools:
      - query_guideline: fetch coding rules for a specific code
      - check_ncci_edits: check for CMS NCCI PTP bundling conflicts
      - flag_error: add an entry to the draft audit report
      - submit_audit: end the episode and trigger the grader

    Task selection: pass task_id as a keyword argument to reset().
    Available task_ids: easy_demographic, medium_ncci_conflict, hard_specificity_untraceable

    The environment supports concurrent WebSocket sessions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Default task cycling order for sequential inference runs
    _TASK_ORDER = [
        "easy_demographic",
        "medium_ncci_conflict",
        "hard_specificity_untraceable",
    ]

    def __init__(self) -> None:
        self._data = _load_data()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[Dict[str, Any]] = None
        self._proposed_codes: Dict[str, Dict[str, str]] = {}
        self._draft_report: List[Dict[str, Any]] = []
        self._codes_queried: List[str] = []
        self._pairs_checked: List[str] = []
        self._done: bool = False
        self._grader_score: Optional[float] = None
        self._max_steps: int = 20

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, **kwargs: Any) -> MedicalCodingObservation:  # type: ignore[override]
        """
        Reset the environment for a new episode.

        Args:
            task_id: Which task to load. If None, cycles through tasks in order
                     based on the episode count. Valid values:
                     'easy_demographic', 'medium_ncci_conflict',
                     'hard_specificity_untraceable'

        Returns:
            Initial MedicalCodingObservation with the full case context.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._draft_report = []
        self._codes_queried = []
        self._pairs_checked = []
        self._done = False
        self._grader_score = None

        # Select task
        tasks = {t["task_id"]: t for t in self._data["tasks"]}
        if task_id and task_id in tasks:
            self._task = tasks[task_id]
        else:
            # Default: cycle through tasks based on episode count hash
            idx = hash(self._state.episode_id) % len(self._TASK_ORDER)
            self._task = tasks[self._TASK_ORDER[idx]]

        self._proposed_codes = self._task["proposed_codes"]
        self._max_steps = self._task["max_steps"]

        return self._build_observation(
            tool_result=(
                f"Audit session started. Task: {self._task['task_id']} "
                f"({self._task['difficulty']} difficulty). "
                f"Review the patient demographics, clinical note, and proposed codes. "
                f"Use query_guideline, check_ncci_edits, flag_error, then submit_audit."
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: MedicalCodingAction) -> MedicalCodingObservation:  # type: ignore[override]
        """
        Execute one auditor action and return the resulting observation.

        Dense reward shaping is applied per the specification.
        """
        if self._done:
            return self._build_observation(
                tool_result="Episode already ended. Call reset() to start a new audit.",
                reward=0.0,
                done=True,
                error="Episode is done. Call reset().",
            )

        self._state.step_count += 1

        # Route to the appropriate handler
        handler = {
            "query_guideline": self._handle_query_guideline,
            "check_ncci_edits": self._handle_check_ncci_edits,
            "flag_error": self._handle_flag_error,
            "submit_audit": self._handle_submit_audit,
        }.get(action.action_type)

        if handler is None:
            return self._build_observation(
                tool_result=f"Unknown action_type: {action.action_type}",
                reward=-0.2,
                done=False,
                error=f"Unknown action_type: {action.action_type}",
            )

        return handler(action)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_query_guideline(self, action: MedicalCodingAction) -> MedicalCodingObservation:
        """Look up coding guidelines for a specific code."""
        code = action.code
        if not code:
            return self._build_observation(
                tool_result="query_guideline requires a 'code' field.",
                reward=-0.2,
                done=False,
                error="Missing required field: code",
            )

        # Hallucination check — code must be in the proposed set
        if code not in self._proposed_codes:
            return self._build_observation(
                tool_result=(
                    f"Code '{code}' is not in the proposed code set for this encounter. "
                    f"Proposed codes are: {list(self._proposed_codes.keys())}. "
                    f"Only query codes that appear in the proposed billing set."
                ),
                reward=-0.5,
                done=False,
                error=f"Hallucinated code: '{code}' is not in the proposed code set.",
            )

        # Repeated query penalty
        if code in self._codes_queried:
            return self._build_observation(
                tool_result=f"You have already queried guidelines for '{code}' this session. Check your draft report.",
                reward=-0.1,
                done=False,
                error=f"Repeated query: '{code}' already queried.",
            )

        # Look up guideline from data
        guideline = self._lookup_guideline(code)
        self._codes_queried.append(code)

        return self._build_observation(
            tool_result=guideline,
            reward=0.1,
            done=False,
        )

    def _handle_check_ncci_edits(self, action: MedicalCodingAction) -> MedicalCodingObservation:
        """Check CMS NCCI PTP edit table for two codes."""
        code1, code2 = action.code1, action.code2
        if not code1 or not code2:
            return self._build_observation(
                tool_result="check_ncci_edits requires both 'code1' and 'code2' fields.",
                reward=-0.2,
                done=False,
                error="Missing required field(s): code1 and/or code2",
            )

        # Hallucination check — both codes must be in the proposed set
        hallucinated = [c for c in (code1, code2) if c not in self._proposed_codes]
        if hallucinated:
            return self._build_observation(
                tool_result=(
                    f"Code(s) {hallucinated} are not in the proposed code set. "
                    f"Only check NCCI edits for codes in the proposed billing set: "
                    f"{list(self._proposed_codes.keys())}."
                ),
                reward=-0.5,
                done=False,
                error=f"Hallucinated code(s): {hallucinated}",
            )

        # Repeated check penalty
        pair_key = "|".join(sorted([code1, code2]))
        if pair_key in self._pairs_checked:
            return self._build_observation(
                tool_result=f"You have already checked the NCCI edit between '{code1}' and '{code2}'.",
                reward=-0.1,
                done=False,
                error=f"Repeated NCCI check: {pair_key}",
            )

        # Look up NCCI edit
        ncci_result, found_edit = self._lookup_ncci_edit(code1, code2)
        self._pairs_checked.append(pair_key)

        # Small positive reward for checking a valid pair (no hallucination)
        reward = 0.05 if not found_edit else 0.1

        return self._build_observation(
            tool_result=ncci_result,
            reward=reward,
            done=False,
        )

    def _handle_flag_error(self, action: MedicalCodingAction) -> MedicalCodingObservation:
        """Record a coding error in the draft audit report."""
        code = action.code
        error_type = action.error_type
        justification = action.justification

        if not code or not error_type or not justification:
            missing = [f for f, v in [("code", code), ("error_type", error_type), ("justification", justification)] if not v]
            return self._build_observation(
                tool_result=f"flag_error requires: code, error_type, justification. Missing: {missing}",
                reward=-0.2,
                done=False,
                error=f"Missing required field(s): {missing}",
            )

        # Hallucination check — code must be in the proposed set
        if code not in self._proposed_codes:
            return self._build_observation(
                tool_result=(
                    f"Cannot flag '{code}' — it is not in the proposed code set. "
                    f"Only flag codes that appear in the proposed billing set: "
                    f"{list(self._proposed_codes.keys())}."
                ),
                reward=-0.5,
                done=False,
                error=f"Hallucinated code: '{code}' is not in the proposed code set.",
            )

        # Repeated flag check
        already_flagged = any(
            e["code"] == code and e["error_type"] == error_type
            for e in self._draft_report
        )
        if already_flagged:
            return self._build_observation(
                tool_result=f"Code '{code}' has already been flagged as '{error_type}'. Avoid duplicate flags.",
                reward=-0.1,
                done=False,
                error=f"Duplicate flag: '{code}' / '{error_type}'",
            )

        # Evaluate correctness against ground truth
        expected_errors = {e["code"]: e["error_type"] for e in self._task["expected_errors"]}
        reward, feedback = self._score_flag(code, error_type, expected_errors)

        # Record in draft report
        self._draft_report.append({
            "code": code,
            "error_type": error_type,
            "justification": justification,
            "step": self._state.step_count,
        })

        return self._build_observation(
            tool_result=f"Error flagged for '{code}' as '{error_type}'. {feedback}",
            reward=reward,
            done=False,
        )

    def _handle_submit_audit(self, action: MedicalCodingAction) -> MedicalCodingObservation:
        """End the episode and grade the submitted audit report."""
        expected_errors = self._task["expected_errors"]
        grader_score = _grade_audit(self._draft_report, expected_errors)
        self._grader_score = grader_score
        self._done = True

        errors_found = len(self._draft_report)
        errors_expected = len(expected_errors)

        summary_lines = [
            f"AUDIT SUBMITTED. Grader score: {grader_score:.2f}",
            f"Errors flagged: {errors_found} | Errors expected: {errors_expected}",
        ]

        for exp in expected_errors:
            flagged = any(
                e["code"] == exp["code"] and e["error_type"] == exp["error_type"]
                for e in self._draft_report
            )
            status = "FOUND" if flagged else "MISSED"
            summary_lines.append(f"  [{status}] {exp['code']} ({exp['error_type']})")

        false_positives = [
            e for e in self._draft_report
            if not any(exp["code"] == e["code"] and exp["error_type"] == e["error_type"] for exp in expected_errors)
        ]
        if false_positives:
            summary_lines.append(f"  False positives: {[e['code'] for e in false_positives]}")

        return self._build_observation(
            tool_result="\n".join(summary_lines),
            reward=grader_score,  # terminal reward = grader score
            done=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lookup_guideline(self, code: str) -> str:
        """Return official guideline text for a given code."""
        icd10 = self._data.get("icd10_guidelines", {})
        cpt = self._data.get("cpt_guidelines", {})

        if code in icd10:
            g = icd10[code]
            parts = [
                f"ICD-10-CM Guideline for {code}: {g['description']}",
                f"Official Note: {g['note']}",
            ]
            if g.get("gender_restriction"):
                parts.append(f"Gender Restriction: APPLICABLE TO {g['gender_restriction'].upper()} PATIENTS ONLY.")
            if g.get("seventh_char_guidance"):
                parts.append(f"7th Character Guidance: {g['seventh_char_guidance']}")
            if g.get("is_valid") is False:
                parts.append("VALIDATION STATUS: THIS CODE IS INVALID AND UNTRACEABLE.")
            if g.get("excludes1"):
                parts.append(f"Excludes1 (mutually exclusive — NEVER code together): {g['excludes1']}")
            return "\n".join(parts)

        if code in cpt:
            g = cpt[code]
            parts = [
                f"CPT Guideline for {code}: {g['description']}",
                f"Official Note: {g['note']}",
            ]
            if g.get("ncci_conflicts"):
                parts.append(f"NCCI Conflicts (cannot bill together): {g['ncci_conflicts']}")
            return "\n".join(parts)

        return (
            f"CODE NOT FOUND: '{code}' could not be located in the ICD-10-CM or CPT "
            f"guideline database. This code is untraceable. "
            f"Flag it using flag_error with error_type='untraceable_code'."
        )

    def _lookup_ncci_edit(self, code1: str, code2: str) -> tuple[str, bool]:
        """Check NCCI PTP edit table for a pair of codes. Returns (result_text, found_edit)."""
        ncci_edits = self._data.get("ncci_edits", [])

        for edit in ncci_edits:
            c1, c2 = edit["column1"], edit["column2"]
            if {code1, code2} == {c1, c2}:
                return (
                    f"NCCI PTP EDIT FOUND between {code1} and {code2}:\n"
                    f"Conflict Type: {edit['conflict_type']}\n"
                    f"Modifier Allowed: {edit.get('modifier_allowed', 'unknown')}\n"
                    f"Details: {edit['note']}",
                    True,
                )

        return (
            f"No NCCI PTP edit found between '{code1}' and '{code2}'. "
            f"These codes may be billed together on the same claim without a bundling conflict.",
            False,
        )

    def _score_flag(
        self,
        code: str,
        error_type: str,
        expected_errors: Dict[str, str],
    ) -> tuple[float, str]:
        """
        Assign per-step reward for a flag_error action.

        Returns (reward, feedback_message).
        """
        if code not in expected_errors:
            # False positive — this code has no expected error
            return -0.2, (
                f"Warning: No coding error is expected for '{code}' in this case. "
                f"This may be a false positive. Review the guidelines again."
            )

        correct_error_type = expected_errors[code]
        if error_type == correct_error_type:
            # Correct code AND correct error_type
            if error_type in ("ncci_edit", "excludes1_conflict"):
                return +0.3, (
                    f"Correct! '{code}' has a '{error_type}' error. "
                    f"Full reward for NCCI/Excludes1 conflict identification."
                )
            else:
                return +0.2, (
                    f"Correct! '{code}' has a '{error_type}' error. "
                    f"Reward for accurate error classification."
                )
        else:
            # Correct code but wrong error_type classification
            return 0.0, (
                f"Code '{code}' does have an error, but the error_type '{error_type}' "
                f"does not match the expected classification. "
                f"Re-examine the coding guidelines for this code."
            )

    def _build_observation(
        self,
        tool_result: str,
        reward: float,
        done: bool,
        error: Optional[str] = None,
    ) -> MedicalCodingObservation:
        """Construct a full observation from current environment state."""
        if self._task is None:
            # Before reset() is called
            return MedicalCodingObservation(
                task_id="",
                difficulty="",
                patient_demographics={},
                clinical_note="",
                proposed_codes={},
                draft_report=[],
                tool_result=tool_result,
                step_count=self._state.step_count,
                codes_queried=self._codes_queried,
                pairs_checked=self._pairs_checked,
                last_action_error=error,
                grader_score=self._grader_score,
                reward=reward,
                done=done,
            )

        return MedicalCodingObservation(
            task_id=self._task["task_id"],
            difficulty=self._task["difficulty"],
            patient_demographics=self._task["patient"],
            clinical_note=self._task["clinical_note"],
            proposed_codes=self._proposed_codes,
            draft_report=list(self._draft_report),
            tool_result=tool_result,
            step_count=self._state.step_count,
            codes_queried=list(self._codes_queried),
            pairs_checked=list(self._pairs_checked),
            last_action_error=error,
            grader_score=self._grader_score,
            reward=reward,
            done=done,
        )
