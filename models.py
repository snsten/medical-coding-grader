"""
Data models for the Medical Coding Auditor Environment.

The agent acts as a Medical Coding Auditor reviewing proposed ICD-10 and CPT codes
against clinical notes and patient demographics. It uses structured tools to query
coding guidelines, check NCCI edits, flag errors, and submit a final audit report.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import ConfigDict, Field


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class MedicalCodingAction(Action):
    """
    A single action the auditor agent can take.

    The agent selects one action_type per step and populates the relevant fields:

    - query_guideline: Look up official ICD-10-CM or CPT coding guidelines for a code.
        Required: code (str)
    - check_ncci_edits: Check CMS NCCI PTP edit table for a pair of procedure codes.
        Required: code1 (str), code2 (str)
    - flag_error: Record a coding error in the draft audit report.
        Required: code (str), error_type (str), justification (str)
        error_type must be one of: demographic_mismatch, excludes1_conflict,
        ncci_edit, specificity_error, untraceable_code
    - ask_clarifying_question: Ask a simulated physician for missing clinical information.
        Required: question (str)
        Use when the clinical note is ambiguous or lacks information needed to
        determine the correct code. The environment returns the physician's response.
    - extract_evidence: Highlight a specific text span from the clinical note as evidence
        supporting a coding decision. Grounds reasoning in documentation.
        Required: evidence_text (str) — must be an exact substring of the clinical note.
    - submit_audit: End the episode and submit the draft report for grading.
        No additional fields required.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    action_type: Literal[
        "query_guideline",
        "check_ncci_edits",
        "flag_error",
        "ask_clarifying_question",
        "extract_evidence",
        "submit_audit",
    ] = Field(
        ...,
        description=(
            "Type of auditor action to execute. One of: "
            "'query_guideline' (look up a code's official guidelines), "
            "'check_ncci_edits' (check if two CPT codes have a PTP bundling conflict), "
            "'flag_error' (record a coding error in the draft audit report), "
            "'ask_clarifying_question' (ask physician for missing clinical information), "
            "'submit_audit' (submit the completed audit report for grading)."
        ),
    )

    # -- query_guideline & flag_error fields --
    code: Optional[str] = Field(
        default=None,
        description=(
            "ICD-10-CM or CPT code to query or flag. "
            "Required for action_type='query_guideline' and 'flag_error'."
        ),
    )

    # -- check_ncci_edits fields --
    code1: Optional[str] = Field(
        default=None,
        description=(
            "First CPT code to check in the NCCI edit table. "
            "Required for action_type='check_ncci_edits'."
        ),
    )
    code2: Optional[str] = Field(
        default=None,
        description=(
            "Second CPT code to check in the NCCI edit table. "
            "Required for action_type='check_ncci_edits'."
        ),
    )

    # -- flag_error fields --
    error_type: Optional[
        Literal[
            "demographic_mismatch",
            "excludes1_conflict",
            "ncci_edit",
            "specificity_error",
            "untraceable_code",
        ]
    ] = Field(
        default=None,
        description=(
            "Classification of the coding error. "
            "Required for action_type='flag_error'. "
            "Options: 'demographic_mismatch' (code inapplicable to patient sex/age), "
            "'excludes1_conflict' (two mutually exclusive diagnosis codes billed together), "
            "'ncci_edit' (NCCI PTP bundling violation between procedure codes), "
            "'specificity_error' (wrong 7th character or insufficient code specificity), "
            "'untraceable_code' (code does not exist in any official code set)."
        ),
    )
    justification: Optional[str] = Field(
        default=None,
        description=(
            "Clinical and coding rationale for the flagged error. "
            "Required for action_type='flag_error'. "
            "Should cite the specific guideline rule or NCCI edit that is violated."
        ),
    )

    # -- ask_clarifying_question fields --
    question: Optional[str] = Field(
        default=None,
        description=(
            "Question to ask the simulated physician for clarification. "
            "Required for action_type='ask_clarifying_question'. "
            "Use when documentation is ambiguous or missing information needed for coding."
        ),
    )

    # -- extract_evidence fields --
    evidence_text: Optional[str] = Field(
        default=None,
        description=(
            "Exact text span from the clinical note to extract as evidence. "
            "Required for action_type='extract_evidence'. "
            "Must be an exact substring of the clinical note. "
            "Use to ground a coding decision in specific documentation."
        ),
    )


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------



class MedicalCodingObservation(Observation):
    """
    Observation returned by the environment after each step.

    Contains the full case context (demographics, clinical note, proposed codes),
    the agent's accumulating draft report, the result of the last tool call,
    and the current reward/done signals.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # -- Case context (constant across episode) --
    task_id: str = Field(
        default="",
        description="Identifier of the current task (easy_demographic / medium_ncci_conflict / hard_specificity_untraceable).",
    )
    difficulty: str = Field(
        default="",
        description="Difficulty level: easy, medium, or hard.",
    )
    patient_demographics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Patient demographics: age (int), sex ('male'/'female'), "
            "mrn (str), insurance (str)."
        ),
    )
    clinical_note: str = Field(
        default="",
        description="Unstructured clinical note describing the encounter and diagnoses.",
    )
    proposed_codes: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description=(
            "Dictionary of proposed billing codes to validate. "
            "Keys are code strings (e.g. 'E11.9', '93306'). "
            "Values are dicts with 'description' (str) and 'code_type' ('ICD-10-CM' or 'CPT')."
        ),
    )

    # -- Dynamic state --
    draft_report: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Accumulating list of errors flagged so far. Each entry has: "
            "code (str), error_type (str), justification (str), step (int)."
        ),
    )
    tool_result: str = Field(
        default="",
        description=(
            "Textual result of the last action executed: "
            "guideline text for query_guideline, "
            "NCCI edit details for check_ncci_edits, "
            "confirmation for flag_error, "
            "grader score summary for submit_audit."
        ),
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken in the current episode.",
    )
    codes_queried: List[str] = Field(
        default_factory=list,
        description="List of codes already queried via query_guideline this episode (for loop detection).",
    )
    pairs_checked: List[str] = Field(
        default_factory=list,
        description="List of code pairs already checked via check_ncci_edits (formatted as 'code1|code2').",
    )
    clarifications_asked: List[str] = Field(
        default_factory=list,
        description="List of clarifying questions asked via ask_clarifying_question this episode.",
    )
    extracted_evidence: List[str] = Field(
        default_factory=list,
        description=(
            "List of text spans extracted from the clinical note via extract_evidence this episode. "
            "Each entry is an exact substring of the clinical note."
        ),
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message if the last action was invalid (e.g. missing required field, hallucinated code), else null.",
    )
    grader_score: Optional[float] = Field(
        default=None,
        description="Final grader score [0.0, 1.0] set when submit_audit is called, else null.",
    )
    episode_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Detailed trajectory evaluation metrics, populated only when done=True. "
            "Includes: trajectory_length, tool_failure_rate, investigation_coverage, "
            "flag_precision, flag_recall, avg_reward_per_step, "
            "investigation_before_flag_rate, clarification_count."
        ),
    )


__all__ = ["MedicalCodingAction", "MedicalCodingObservation"]
