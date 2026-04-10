"""
Medical Coding Auditor Environment — Core Logic.

The agent plays the role of a Medical Coding Auditor performing a pre-bill review.
It validates a proposed set of ICD-10 and CPT codes against a clinical note and
patient demographics without assigning new codes.

Reward shaping (dense, multi-signal):
  +0.10  QueryGuideline on a code present in the proposed set
  +0.30  FlagError that correctly identifies an NCCI edit or Excludes1 conflict
  +0.20  FlagError that correctly identifies a demographic mismatch,
         specificity error, or untraceable code
  +0.05  Justification quality bonus (key terms from expected error description)
  +0.05  Investigation coherence bonus (queried code's guideline before flagging it)
  -0.05  Flag without investigation penalty (flagging a code without querying first)
  -0.10  Repeated QueryGuideline / CheckNCCIEdits on an already-queried code/pair
  -0.10  FlagError with correct code but wrong error_type classification
  -0.20  FlagError on a code with no expected error (false positive)
  -0.50  Any action that references a code NOT in the proposed set (hallucination)
  +final Terminal reward on SubmitAudit:
         grader_score - 0.15 * false_positives + 0.1 * (max_steps - steps) / max_steps
         + 0.05 thorough_review bonus (queried all proposed codes before submitting)
  +0.15  Relevant clarifying question (matches trigger keywords in clarification pool)
  -0.10  Irrelevant or repeated clarifying question
  +0.10  Terminal bonus for asking necessary clarifications before flagging
  Auto-termination: episode ends with auto-submit and -0.1 penalty at max_steps

Hierarchical reward (HERON-style):
  Wrong error_type penalty scaled by conceptual distance between error types.
  Close misclassifications (e.g. excludes1↔ncci_edit) penalized less than
  distant ones (e.g. demographic_mismatch↔untraceable_code).

Partial observability mode:
  When enabled via reset(partial_observability=True), the clinical note is
  progressively revealed as the agent queries codes. Initially only the
  REASON FOR VISIT section is visible. Each query_guideline reveals the
  next section. This models real-world information gathering.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
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
# Feature I: HERON-style error_type distance matrix
# ---------------------------------------------------------------------------
# Distance 0 = same type, 1 = close (same category), 2 = far (different category)
# Categories: "conflict" (excludes1, ncci_edit), "validity" (specificity, untraceable),
#             "demographic" (demographic_mismatch)

_ERROR_TYPE_DISTANCE: Dict[tuple[str, str], int] = {}
_ERROR_TYPES = [
    "demographic_mismatch", "excludes1_conflict", "ncci_edit",
    "specificity_error", "untraceable_code",
]
_ERROR_CATEGORIES = {
    "excludes1_conflict": "conflict",
    "ncci_edit": "conflict",
    "specificity_error": "validity",
    "untraceable_code": "validity",
    "demographic_mismatch": "demographic",
}

for _a in _ERROR_TYPES:
    for _b in _ERROR_TYPES:
        if _a == _b:
            _ERROR_TYPE_DISTANCE[(_a, _b)] = 0
        elif _ERROR_CATEGORIES[_a] == _ERROR_CATEGORIES[_b]:
            _ERROR_TYPE_DISTANCE[(_a, _b)] = 1
        else:
            _ERROR_TYPE_DISTANCE[(_a, _b)] = 2


# ---------------------------------------------------------------------------
# Partial Observability — Clinical Note Section Parser
# ---------------------------------------------------------------------------

# Common clinical note section headers
_SECTION_PATTERN = re.compile(
    r"^(REASON FOR VISIT|HISTORY OF PRESENT ILLNESS|PAST MEDICAL HISTORY|"
    r"SOCIAL HISTORY|REVIEW OF SYSTEMS|MEDICATIONS|PHYSICAL EXAM|"
    r"EXAMINATION|IMAGING|SPIROMETRY[^:]*|SERVICES RENDERED|"
    r"ENDOCRINOLOGY ASSESSMENT|CARDIOLOGY ASSESSMENT|RIGHT ARM|"
    r"ASSESSMENT AND PLAN|ASSESSMENT|PLAN)\s*:",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_note_sections(clinical_note: str) -> List[tuple[str, str]]:
    """
    Parse a clinical note into (header, body) sections.
    Returns at least one section. Unparseable notes return as a single section.
    """
    matches = list(_SECTION_PATTERN.finditer(clinical_note))
    if not matches:
        return [("CLINICAL NOTE", clinical_note)]

    sections: List[tuple[str, str]] = []
    for i, match in enumerate(matches):
        header = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(clinical_note)
        body = clinical_note[start:end].strip()
        sections.append((header, body))

    return sections


def _build_visible_note(
    sections: List[tuple[str, str]],
    revealed_count: int,
) -> str:
    """Build the currently visible clinical note from revealed sections."""
    if not sections:
        return ""
    visible = sections[:revealed_count]
    hidden_count = len(sections) - revealed_count
    parts = [f"{header}: {body}" for header, body in visible]
    if hidden_count > 0:
        parts.append(f"\n[{hidden_count} more section(s) not yet revealed — query codes to investigate further]")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Deterministic Grader
# ---------------------------------------------------------------------------

def _grade_audit(
    draft_report: List[Dict[str, Any]],
    expected_errors: List[Dict[str, Any]],
) -> tuple[float, int]:
    """
    Compare the agent's submitted audit report against the ground truth.

    Scoring logic:
    - For each expected error, check if the agent correctly flagged it.
    - A flag is "correct" if both the code AND error_type match → 1.0 credit.
    - Partial credit (0.5x) if the correct code is flagged but with a wrong error_type.
    - False positives (flags on codes with no expected error) are counted and penalized
      in the terminal reward calculation.
    - Final base score = sum(credit for each expected error) / len(expected_errors)

    Returns (base_score, false_positive_count).
    """
    if not expected_errors:
        return 1.0, 0

    expected_codes = {e["code"] for e in expected_errors}

    flagged_by_code: Dict[str, List[str]] = {}
    false_positives = 0
    for entry in draft_report:
        code = entry.get("code", "")
        if code not in flagged_by_code:
            flagged_by_code[code] = []
        flagged_by_code[code].append(entry.get("error_type", ""))
        if code not in expected_codes:
            false_positives += 1

    total_credit = 0.0
    for expected in expected_errors:
        exp_code = expected["code"]
        exp_error_type = expected["error_type"]

        if exp_code not in flagged_by_code:
            continue

        agent_error_types = flagged_by_code[exp_code]

        if exp_error_type in agent_error_types:
            total_credit += 1.0
        else:
            total_credit += 0.5

    base_score = round(total_credit / len(expected_errors), 4)
    return base_score, false_positives


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
        "medium_excludes1",
        "hard_specificity_untraceable",
        "expert_multi_error",
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
        # Feature F: investigation strategy tracking
        self._action_history: List[tuple[str, Optional[str]]] = []
        # Feature E: partial observability
        self._partial_observability: bool = False
        self._note_sections: List[tuple[str, str]] = []  # (header, body) pairs
        self._revealed_section_count: int = 0
        # Feature G: physician clarifications
        self._clarifications_asked: List[str] = []
        # Feature M: episode metrics (populated at episode end)
        self._episode_metrics: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, partial_observability: Optional[bool] = None, **kwargs: Any) -> MedicalCodingObservation:  # type: ignore[override]
        """
        Reset the environment for a new episode.

        Args:
            task_id: Which task to load. If None, cycles through tasks in order.
                     Use 'random' for a procedurally generated case.
            partial_observability: If True, the clinical note is progressively
                     revealed as the agent queries codes. Default: False.

        Returns:
            Initial MedicalCodingObservation with the case context.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._draft_report = []
        self._codes_queried = []
        self._pairs_checked = []
        self._done = False
        self._grader_score = None
        self._action_history = []

        # Select task
        tasks = {t["task_id"]: t for t in self._data["tasks"]}
        if task_id and task_id.startswith("random"):
            from data.case_generator import generate_random_task
            difficulty = task_id.split("_", 1)[1] if "_" in task_id else None
            self._task = generate_random_task(
                self._data, self._state.episode_id, difficulty=difficulty,
            )
        elif task_id and task_id in tasks:
            self._task = tasks[task_id]
        else:
            idx = hash(self._state.episode_id) % len(self._TASK_ORDER)
            self._task = tasks[self._TASK_ORDER[idx]]

        self._proposed_codes = self._task["proposed_codes"]
        self._max_steps = self._task["max_steps"]

        # Partial observability setup
        self._partial_observability = bool(partial_observability)
        if self._partial_observability:
            self._note_sections = _parse_note_sections(self._task["clinical_note"])
            self._revealed_section_count = 1  # start with just REASON FOR VISIT
        else:
            self._note_sections = []
            self._revealed_section_count = 0

        po_note = " [PARTIAL OBSERVABILITY: clinical note will be revealed as you investigate.]" if self._partial_observability else ""

        return self._build_observation(
            tool_result=(
                f"Audit session started. Task: {self._task['task_id']} "
                f"({self._task['difficulty']} difficulty). "
                f"Review the patient demographics, clinical note, and proposed codes. "
                f"Use query_guideline, check_ncci_edits, flag_error, then submit_audit."
                f"{po_note}"
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: MedicalCodingAction) -> MedicalCodingObservation:  # type: ignore[override]
        """
        Execute one auditor action and return the resulting observation.

        Dense reward shaping is applied per the specification.
        Auto-terminates at max_steps with a timeout penalty.
        """
        if self._done:
            return self._build_observation(
                tool_result="Episode already ended. Call reset() to start a new audit.",
                reward=0.0,
                done=True,
                error="Episode is done. Call reset().",
            )

        self._state.step_count += 1

        # Auto-termination: if this is the last allowed step, force submit
        if self._state.step_count >= self._max_steps and action.action_type != "submit_audit":
            return self._handle_auto_termination()

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
            self._action_history.append(("query_guideline", None))
            return self._build_observation(
                tool_result="query_guideline requires a 'code' field.",
                reward=-0.2,
                done=False,
                error="Missing required field: code",
            )

        self._action_history.append(("query_guideline", code))

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

        # Partial observability: reveal next section
        if self._partial_observability and self._revealed_section_count < len(self._note_sections):
            self._revealed_section_count += 1

        return self._build_observation(
            tool_result=guideline,
            reward=0.1,
            done=False,
        )

    def _handle_check_ncci_edits(self, action: MedicalCodingAction) -> MedicalCodingObservation:
        """Check CMS NCCI PTP edit table for two codes."""
        code1, code2 = action.code1, action.code2
        self._action_history.append(("check_ncci_edits", f"{code1}|{code2}"))
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

        self._action_history.append(("flag_error", code))

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
        reward, feedback = self._score_flag(code, error_type, justification, expected_errors)

        # Feature F: investigation coherence bonus/penalty
        investigated = self._was_investigated(code, error_type)
        if investigated:
            reward += 0.05
            feedback += " (Investigation coherence bonus: +0.05)"
        else:
            reward -= 0.05
            feedback += " (No prior investigation of this code: -0.05)"

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
        return self._finalize_episode(timeout=False)

    def _handle_auto_termination(self) -> MedicalCodingObservation:
        """Auto-submit when max_steps is reached. Applies a timeout penalty."""
        return self._finalize_episode(timeout=True)

    def _finalize_episode(self, timeout: bool) -> MedicalCodingObservation:
        """
        Grade the audit and compute the terminal reward.

        Terminal reward = base_grader_score
                        - 0.15 * false_positive_count
                        + efficiency_bonus (0.1 * remaining_steps / max_steps)
                        - timeout_penalty (0.1 if auto-terminated)

        Clamped to [0.0, 1.0].
        """
        expected_errors = self._task["expected_errors"]
        base_score, fp_count = _grade_audit(self._draft_report, expected_errors)

        # Efficiency bonus: reward finishing early
        steps_used = self._state.step_count
        efficiency_bonus = 0.1 * max(self._max_steps - steps_used, 0) / self._max_steps

        # False positive penalty
        fp_penalty = 0.15 * fp_count

        # Timeout penalty
        timeout_penalty = 0.1 if timeout else 0.0

        # Thorough review bonus
        review_bonus = self._thorough_review_bonus() if not timeout else 0.0

        terminal_reward = base_score - fp_penalty + efficiency_bonus - timeout_penalty + review_bonus
        terminal_reward = round(max(min(terminal_reward, 1.0), 0.0), 4)

        self._grader_score = terminal_reward
        self._done = True

        errors_found = len(self._draft_report)
        errors_expected = len(expected_errors)

        summary_lines = [
            f"AUDIT {'AUTO-SUBMITTED (max steps reached)' if timeout else 'SUBMITTED'}.",
            f"Base grader score: {base_score:.2f} | Efficiency bonus: +{efficiency_bonus:.2f}"
            + (f" | Thorough review bonus: +{review_bonus:.2f}" if review_bonus > 0 else "")
            + (f" | Timeout penalty: -{timeout_penalty:.2f}" if timeout else "")
            + (f" | False positive penalty: -{fp_penalty:.2f}" if fp_count > 0 else ""),
            f"Final score: {terminal_reward:.2f}",
            f"Errors flagged: {errors_found} | Errors expected: {errors_expected}",
        ]

        for exp in expected_errors:
            flagged = any(
                e["code"] == exp["code"] and e["error_type"] == exp["error_type"]
                for e in self._draft_report
            )
            status = "FOUND" if flagged else "MISSED"
            summary_lines.append(f"  [{status}] {exp['code']} ({exp['error_type']})")

        if fp_count > 0:
            expected_codes = {e["code"] for e in expected_errors}
            fp_entries = [e["code"] for e in self._draft_report if e["code"] not in expected_codes]
            summary_lines.append(f"  False positives: {fp_entries}")

        return self._build_observation(
            tool_result="\n".join(summary_lines),
            reward=terminal_reward,
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

    def _was_investigated(self, code: str, error_type: str) -> bool:
        """
        Check whether the agent investigated this code before flagging it.

        Returns True if:
        - For any error_type: agent called query_guideline on the code prior to this flag
        - For ncci_edit: agent also called check_ncci_edits with this code in the pair
        """
        queried = code in self._codes_queried
        if error_type == "ncci_edit":
            # Also check if the agent did an NCCI check involving this code
            ncci_checked = any(code in pair for pair in self._pairs_checked)
            return queried or ncci_checked
        return queried

    def _thorough_review_bonus(self) -> float:
        """
        Bonus for querying all proposed codes before submitting.
        Returns 0.05 if all codes were queried, 0.0 otherwise.
        """
        all_proposed = set(self._proposed_codes.keys())
        all_queried = set(self._codes_queried)
        # Consider NCCI-checked codes as "investigated" too
        for pair in self._pairs_checked:
            all_queried.update(pair.split("|"))
        if all_proposed <= all_queried:
            return 0.05
        return 0.0

    def _justification_quality_bonus(self, code: str, justification: str) -> float:
        """
        Award a small bonus if the justification contains key terms from the
        expected error description. Rewards reasoning quality, not just classification.

        Returns bonus in [0.0, 0.05].
        """
        expected_errors = self._task.get("expected_errors", [])
        for exp in expected_errors:
            if exp["code"] != code:
                continue
            key_terms = exp.get("key_terms", [])
            if not key_terms:
                return 0.0
            justification_lower = justification.lower()
            hits = sum(1 for t in key_terms if t.lower() in justification_lower)
            # Require at least 2 key terms for the bonus
            if hits >= 2:
                return 0.05
        return 0.0

    def _score_flag(
        self,
        code: str,
        error_type: str,
        justification: str,
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
            justification_bonus = self._justification_quality_bonus(code, justification)
            if error_type in ("ncci_edit", "excludes1_conflict"):
                base = 0.3
                return base + justification_bonus, (
                    f"Correct! '{code}' has a '{error_type}' error. "
                    f"Full reward for NCCI/Excludes1 conflict identification."
                    + (f" Justification quality bonus: +{justification_bonus:.2f}" if justification_bonus > 0 else "")
                )
            else:
                base = 0.2
                return base + justification_bonus, (
                    f"Correct! '{code}' has a '{error_type}' error. "
                    f"Reward for accurate error classification."
                    + (f" Justification quality bonus: +{justification_bonus:.2f}" if justification_bonus > 0 else "")
                )
        else:
            # Correct code but wrong error_type classification — small negative
            return -0.1, (
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

        # Partial observability: show only revealed sections
        if self._partial_observability and self._note_sections:
            visible_note = _build_visible_note(self._note_sections, self._revealed_section_count)
        else:
            visible_note = self._task["clinical_note"]

        return MedicalCodingObservation(
            task_id=self._task["task_id"],
            difficulty=self._task["difficulty"],
            patient_demographics=self._task["patient"],
            clinical_note=visible_note,
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
