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
# Feature L: Frequency-weighted asymmetric reward multipliers
# ---------------------------------------------------------------------------
# Rarity reflects how often each error type occurs in real-world billing audits.
# Rare, subtle errors are disproportionately rewarded (ASL-inspired).
_ERROR_RARITY: Dict[str, float] = {
    "demographic_mismatch": 1.0,   # common, visually obvious
    "ncci_edit": 1.2,              # moderate — requires NCCI lookup
    "excludes1_conflict": 1.2,     # moderate — requires guideline knowledge
    "untraceable_code": 1.3,       # less common — requires code validation
    "specificity_error": 1.5,      # rare, subtle — 7th character nuance
}

# ---------------------------------------------------------------------------
# Feature K: Adaptive curriculum learning — class-level state
# ---------------------------------------------------------------------------
_CURRICULUM_LEVELS = ["easy", "medium", "hard", "expert"]
_CURRICULUM_PROMOTE_THRESHOLD = 0.8   # avg score to move up
_CURRICULUM_DEMOTE_THRESHOLD = 0.3    # avg score to move down
_CURRICULUM_WINDOW = 5                # episodes to average over

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

        # Feature L: rarity multiplier applied to grader credit
        rarity = _ERROR_RARITY.get(exp_error_type, 1.0)
        if exp_error_type in agent_error_types:
            total_credit += 1.0 * rarity
        else:
            # HERON: partial credit scaled by conceptual distance of best agent guess
            best_distance = min(
                _ERROR_TYPE_DISTANCE.get((atype, exp_error_type), 2)
                for atype in agent_error_types
            )
            # distance=1 (close miss) → 0.55 credit, distance=2 (far miss) → 0.4 credit
            partial = max(0.4, 0.7 - 0.15 * best_distance)
            total_credit += partial * rarity

    # Normalize by total possible credit (sum of rarity weights, not raw count)
    max_credit = sum(_ERROR_RARITY.get(e["error_type"], 1.0) for e in expected_errors)
    base_score = round(total_credit / max(max_credit, 1.0), 4)
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

    # Feature K: class-level curriculum state (shared across episodes)
    _performance_history: Dict[str, List[float]] = {
        level: [] for level in _CURRICULUM_LEVELS
    }
    _curriculum_level: str = "easy"

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
        # Feature H: evidence extraction
        self._extracted_evidence: List[str] = []        # all extracted spans
        self._evidence_covered_codes: Set[str] = set()  # codes with relevant evidence
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
        self._clarifications_asked = []
        self._extracted_evidence = []
        self._evidence_covered_codes = set()
        self._episode_metrics = None

        # Select task
        tasks = {t["task_id"]: t for t in self._data["tasks"]}
        if task_id and task_id.startswith("random"):
            from data.case_generator import generate_random_task
            if "_" in task_id:
                # Explicit difficulty suffix: random_easy, random_medium, etc.
                difficulty = task_id.split("_", 1)[1]
            else:
                # Feature K: no suffix → use adaptive curriculum level
                difficulty = self._update_curriculum()
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
            "ask_clarifying_question": self._handle_ask_clarification,
            "extract_evidence": self._handle_extract_evidence,
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

    def _handle_ask_clarification(self, action: MedicalCodingAction) -> MedicalCodingObservation:
        """
        Simulate asking the physician a clarifying question.

        The environment matches the question against keyword triggers in the task's
        clarification_pool. A relevant question yields a physician response and a
        +0.15 reward. Irrelevant or repeated questions are penalized.
        """
        question = action.question
        self._action_history.append(("ask_clarifying_question", question))

        if not question:
            return self._build_observation(
                tool_result="ask_clarifying_question requires a 'question' field.",
                reward=-0.2,
                done=False,
                error="Missing required field: question",
            )

        # Repeated question penalty
        if question in self._clarifications_asked:
            return self._build_observation(
                tool_result="You have already asked this question. Avoid repeating clarifications.",
                reward=-0.1,
                done=False,
                error="Repeated clarifying question.",
            )

        self._clarifications_asked.append(question)

        # Match against clarification_pool (keyword-based)
        pool = self._task.get("clarification_pool", [])
        question_lower = question.lower()

        for entry in pool:
            triggers = entry.get("triggers", [])
            if any(t.lower() in question_lower for t in triggers):
                response = entry["response"]
                return self._build_observation(
                    tool_result=(
                        f"PHYSICIAN RESPONSE: {response}"
                    ),
                    reward=0.15,
                    done=False,
                )

        # No matching clarification — irrelevant question
        return self._build_observation(
            tool_result=(
                "PHYSICIAN RESPONSE: I don't have additional information on that. "
                "Please proceed with the available documentation."
            ),
            reward=-0.1,
            done=False,
            error="Irrelevant clarifying question (no matching trigger in task pool).",
        )

    def _handle_extract_evidence(self, action: MedicalCodingAction) -> MedicalCodingObservation:
        """
        Feature H: Extract a text span from the clinical note as evidence.

        Validates the span exists verbatim in the note (hallucination guard).
        Checks if it overlaps with expected evidence spans — rewards grounded reasoning.
        Tracks which codes have documented evidence for the flag_error bonus.
        """
        span = action.evidence_text
        self._action_history.append(("extract_evidence", span))

        if not span:
            return self._build_observation(
                tool_result="extract_evidence requires an 'evidence_text' field.",
                reward=-0.2,
                done=False,
                error="Missing required field: evidence_text",
            )

        clinical_note = self._task["clinical_note"]

        # Hallucination guard: span must appear verbatim in the note
        if span not in clinical_note:
            return self._build_observation(
                tool_result=(
                    f"Evidence text not found verbatim in the clinical note. "
                    f"Only extract exact substrings from the note."
                ),
                reward=-0.05,
                done=False,
                error="Hallucinated evidence: span not found in clinical note.",
            )

        # Repeated extraction penalty
        if span in self._extracted_evidence:
            return self._build_observation(
                tool_result="This evidence span has already been extracted.",
                reward=-0.05,
                done=False,
                error="Repeated evidence extraction.",
            )

        self._extracted_evidence.append(span)
        span_lower = span.lower()

        # Check against expected error evidence spans
        matched_codes = []
        for exp in self._task.get("expected_errors", []):
            for evidence_span in exp.get("evidence_spans", []):
                if evidence_span.lower() in span_lower or span_lower in evidence_span.lower():
                    matched_codes.append(exp["code"])
                    self._evidence_covered_codes.add(exp["code"])
                    break

        if matched_codes:
            return self._build_observation(
                tool_result=(
                    f"Evidence extracted: \"{span[:120]}\"\n"
                    f"This span is relevant to coding decision(s) for: {matched_codes}."
                ),
                reward=0.05,
                done=False,
            )

        # Valid span but not linked to any expected error
        return self._build_observation(
            tool_result=(
                f"Evidence extracted: \"{span[:120]}\"\n"
                f"This span does not appear to be directly relevant to the expected coding errors. "
                f"Continue investigating."
            ),
            reward=0.0,
            done=False,
        )

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

        # Feature G: clarification terminal bonus
        # +0.10 if agent asked at least one relevant clarification before flagging
        clarification_bonus = 0.0
        pool = self._task.get("clarification_pool", [])
        if pool and self._clarifications_asked:
            # Check if any asked question matched a pool trigger
            matched = False
            for entry in pool:
                triggers = entry.get("triggers", [])
                for q in self._clarifications_asked:
                    if any(t.lower() in q.lower() for t in triggers):
                        matched = True
                        break
                if matched:
                    break
            if matched:
                clarification_bonus = 0.10

        terminal_reward = (
            base_score
            - fp_penalty
            + efficiency_bonus
            - timeout_penalty
            + review_bonus
            + clarification_bonus
        )
        terminal_reward = round(max(min(terminal_reward, 1.0), 0.0), 4)

        self._grader_score = terminal_reward
        self._done = True

        # Feature K: record score in curriculum history
        difficulty = self._task.get("difficulty", "easy")
        if difficulty in MedicalCodingEnvironment._performance_history:
            MedicalCodingEnvironment._performance_history[difficulty].append(terminal_reward)

        # Feature M: compute episode metrics
        self._episode_metrics = self._compute_episode_metrics(base_score, fp_count)

        errors_found = len(self._draft_report)
        errors_expected = len(expected_errors)

        summary_lines = [
            f"AUDIT {'AUTO-SUBMITTED (max steps reached)' if timeout else 'SUBMITTED'}.",
            f"Base grader score: {base_score:.2f} | Efficiency bonus: +{efficiency_bonus:.2f}"
            + (f" | Thorough review bonus: +{review_bonus:.2f}" if review_bonus > 0 else "")
            + (f" | Clarification bonus: +{clarification_bonus:.2f}" if clarification_bonus > 0 else "")
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

    def _update_curriculum(self) -> str:
        """
        Feature K: Adaptive curriculum learning (HiCu-style).

        Reads the class-level performance history to decide what difficulty to use
        for the next random episode. Promotes when avg score > 0.8 over the last
        N episodes; demotes when avg < 0.3. Returns the chosen difficulty string.
        """
        level = MedicalCodingEnvironment._curriculum_level
        history = MedicalCodingEnvironment._performance_history.get(level, [])

        if len(history) >= _CURRICULUM_WINDOW:
            avg = sum(history[-_CURRICULUM_WINDOW:]) / _CURRICULUM_WINDOW
            idx = _CURRICULUM_LEVELS.index(level)
            if avg >= _CURRICULUM_PROMOTE_THRESHOLD and idx < len(_CURRICULUM_LEVELS) - 1:
                MedicalCodingEnvironment._curriculum_level = _CURRICULUM_LEVELS[idx + 1]
            elif avg <= _CURRICULUM_DEMOTE_THRESHOLD and idx > 0:
                MedicalCodingEnvironment._curriculum_level = _CURRICULUM_LEVELS[idx - 1]

        return MedicalCodingEnvironment._curriculum_level

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

    def _compute_episode_metrics(self, base_score: float, fp_count: int) -> Dict[str, Any]:
        """
        Feature M: Compute trajectory evaluation metrics at episode end.

        Returns a dict with metrics useful for RL training diagnostics:
        - trajectory_length: total steps taken
        - tool_failure_rate: fraction of steps that returned an error
        - investigation_coverage: fraction of proposed codes investigated
        - flag_precision: fraction of flags that are true positives
        - flag_recall: fraction of expected errors that were flagged correctly
        - avg_reward_per_step: mean per-step reward (excluding terminal)
        - investigation_before_flag_rate: fraction of flags preceded by investigation
        - clarification_count: number of clarifying questions asked
        """
        steps = self._state.step_count
        expected_errors = self._task.get("expected_errors", [])
        expected_by_code = {e["code"]: e["error_type"] for e in expected_errors}

        # Tool failure rate: actions with an error response
        error_actions = sum(
            1 for atype, _ in self._action_history
            if atype in ("query_guideline", "check_ncci_edits", "flag_error", "ask_clarifying_question")
        )
        # Count repeated/hallucinated actions as failures by checking action_history length
        failure_steps = sum(
            1 for atype, val in self._action_history
            if val is not None and atype == "query_guideline" and (
                val not in self._proposed_codes or self._action_history.count((atype, val)) > 1
            )
        )
        tool_failure_rate = round(failure_steps / max(steps, 1), 4)

        # Investigation coverage
        investigated: Set[str] = set(self._codes_queried)
        for pair in self._pairs_checked:
            investigated.update(pair.split("|"))
        proposed_set = set(self._proposed_codes.keys())
        investigation_coverage = round(
            len(investigated & proposed_set) / max(len(proposed_set), 1), 4
        )

        # Flag precision / recall
        true_positives = sum(
            1 for e in self._draft_report
            if e["code"] in expected_by_code and e["error_type"] == expected_by_code[e["code"]]
        )
        flag_precision = round(true_positives / max(len(self._draft_report), 1), 4)
        flag_recall = round(true_positives / max(len(expected_errors), 1), 4)

        # Investigation-before-flag rate
        flag_actions = [
            val for atype, val in self._action_history if atype == "flag_error" and val
        ]
        investigated_before_flag = sum(
            1 for code in flag_actions if code in investigated
        )
        investigation_before_flag_rate = round(
            investigated_before_flag / max(len(flag_actions), 1), 4
        )

        return {
            "trajectory_length": steps,
            "tool_failure_rate": tool_failure_rate,
            "investigation_coverage": investigation_coverage,
            "flag_precision": flag_precision,
            "flag_recall": flag_recall,
            "base_grader_score": round(base_score, 4),
            "false_positive_count": fp_count,
            "investigation_before_flag_rate": investigation_before_flag_rate,
            "clarification_count": len(self._clarifications_asked),
        }

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
            # Feature H: evidence extraction bonus
            evidence_bonus = 0.05 if code in self._evidence_covered_codes else 0.0
            # Feature L: rarity multiplier scales the base reward
            rarity = _ERROR_RARITY.get(error_type, 1.0)
            if error_type in ("ncci_edit", "excludes1_conflict"):
                base = round(0.3 * rarity, 3)
                return base + justification_bonus + evidence_bonus, (
                    f"Correct! '{code}' has a '{error_type}' error. "
                    f"Full reward for NCCI/Excludes1 conflict identification (rarity×{rarity})."
                    + (f" Justification bonus: +{justification_bonus:.2f}" if justification_bonus > 0 else "")
                    + (f" Evidence bonus: +{evidence_bonus:.2f}" if evidence_bonus > 0 else "")
                )
            else:
                base = round(0.2 * rarity, 3)
                return base + justification_bonus + evidence_bonus, (
                    f"Correct! '{code}' has a '{error_type}' error. "
                    f"Reward for accurate error classification (rarity×{rarity})."
                    + (f" Justification bonus: +{justification_bonus:.2f}" if justification_bonus > 0 else "")
                    + (f" Evidence bonus: +{evidence_bonus:.2f}" if evidence_bonus > 0 else "")
                )
        else:
            # Correct code but wrong error_type — HERON-scaled penalty by conceptual distance
            distance = _ERROR_TYPE_DISTANCE.get((error_type, correct_error_type), 2)
            penalty = -0.05 * distance  # -0.05 for close miss, -0.10 for far miss
            return penalty, (
                f"Code '{code}' does have an error, but '{error_type}' does not match "
                f"the expected classification (distance={distance}). "
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
                clarifications_asked=self._clarifications_asked,
                extracted_evidence=self._extracted_evidence,
                last_action_error=error,
                grader_score=self._grader_score,
                episode_metrics=self._episode_metrics,
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
            clarifications_asked=list(self._clarifications_asked),
            extracted_evidence=list(self._extracted_evidence),
            last_action_error=error,
            grader_score=self._grader_score,
            episode_metrics=self._episode_metrics,
            reward=reward,
            done=done,
        )
