"""
Tests for reward signal correctness.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.environment import MedicalCodingEnvironment
from models import MedicalCodingAction


def test_correct_flag_positive_reward():
    """Correctly flagging an expected error should yield positive reward."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    # Query first (investigation bonus)
    env.step(MedicalCodingAction(action_type="query_guideline", code="O80"))

    # Flag correctly
    obs = env.step(MedicalCodingAction(
        action_type="flag_error",
        code="O80",
        error_type="demographic_mismatch",
        justification="O80 is a maternity code, patient is male.",
    ))
    assert obs.reward > 0


def test_false_positive_negative_reward():
    """Flagging a code that has no expected error should yield -0.20 reward."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    obs = env.step(MedicalCodingAction(
        action_type="flag_error",
        code="E11.9",
        error_type="demographic_mismatch",
        justification="Test false positive.",
    ))
    assert obs.reward < 0


def test_wrong_error_type_penalized():
    """Flagging the right code with wrong error_type should penalize (not reward)."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    obs = env.step(MedicalCodingAction(
        action_type="flag_error",
        code="O80",
        error_type="untraceable_code",  # wrong type
        justification="Testing wrong type.",
    ))
    assert obs.reward < 0  # should be penalized


def test_ncci_check_positive_reward():
    """Checking NCCI edits for a conflicting pair should yield positive reward."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="medium_ncci_conflict")

    obs = env.step(MedicalCodingAction(
        action_type="check_ncci_edits",
        code1="93306",
        code2="93307",
    ))
    assert obs.reward > 0  # conflict found = +0.10


def test_repeated_query_penalized():
    """Querying the same code twice should be penalized on the second call."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    # First query: positive
    obs1 = env.step(MedicalCodingAction(action_type="query_guideline", code="O80"))
    assert obs1.reward > 0

    # Second query: negative (loop detection)
    obs2 = env.step(MedicalCodingAction(action_type="query_guideline", code="O80"))
    assert obs2.reward < 0


def test_grader_score_range():
    """Grader score should be in [0.0, 1.0] for all tasks."""
    env = MedicalCodingEnvironment()
    tasks = [
        "easy_demographic",
        "medium_ncci_conflict",
        "medium_excludes1",
        "hard_specificity_untraceable",
        "expert_multi_error",
    ]
    for task_id in tasks:
        env.reset(task_id=task_id)
        obs = env.step(MedicalCodingAction(action_type="submit_audit"))
        assert obs.grader_score is not None
        assert 0.0 <= obs.grader_score <= 1.0, \
            f"{task_id}: grader_score {obs.grader_score} out of range"


def test_efficiency_bonus_for_fast_solve():
    """Solving quickly should yield higher score than solving slowly."""
    # Fast solve: 3 steps
    env_fast = MedicalCodingEnvironment()
    env_fast.reset(task_id="easy_demographic")
    env_fast.step(MedicalCodingAction(action_type="query_guideline", code="O80"))
    env_fast.step(MedicalCodingAction(
        action_type="flag_error",
        code="O80",
        error_type="demographic_mismatch",
        justification="Maternity code for male patient.",
    ))
    obs_fast = env_fast.step(MedicalCodingAction(action_type="submit_audit"))

    # Slow solve: 7 steps (query every code first)
    env_slow = MedicalCodingEnvironment()
    env_slow.reset(task_id="easy_demographic")
    env_slow.step(MedicalCodingAction(action_type="query_guideline", code="O80"))
    env_slow.step(MedicalCodingAction(action_type="query_guideline", code="E11.9"))
    env_slow.step(MedicalCodingAction(action_type="query_guideline", code="Z00.00"))
    env_slow.step(MedicalCodingAction(
        action_type="check_ncci_edits",
        code1="O80",
        code2="E11.9",
    ))
    env_slow.step(MedicalCodingAction(
        action_type="flag_error",
        code="O80",
        error_type="demographic_mismatch",
        justification="Maternity code for male patient.",
    ))
    # Extra step: ask a question
    env_slow.step(MedicalCodingAction(
        action_type="ask_clarifying_question",
        question="Is the patient male?",
    ))
    obs_slow = env_slow.step(MedicalCodingAction(action_type="submit_audit"))

    # Fast solve should have higher grader_score due to efficiency bonus
    assert obs_fast.grader_score >= obs_slow.grader_score


def test_extract_evidence_in_note_no_penalty():
    """Extracting evidence that exists in the clinical note should not penalize."""
    env = MedicalCodingEnvironment()
    obs = env.reset(task_id="easy_demographic")

    # Extract a span that's in the clinical note
    evidence = "34-year-old male"
    obs = env.step(MedicalCodingAction(
        action_type="extract_evidence",
        evidence_text=evidence,
    ))
    assert obs.reward >= 0
    assert evidence in obs.extracted_evidence


def test_extract_evidence_hallucinated_penalized():
    """Extracting text NOT in the clinical note should be penalized."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    obs = env.step(MedicalCodingAction(
        action_type="extract_evidence",
        evidence_text="This text is definitely not in the clinical note xyz123",
    ))
    assert obs.reward < 0


def test_clarification_relevant_positive():
    """A relevant clarifying question should yield positive reward."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    obs = env.step(MedicalCodingAction(
        action_type="ask_clarifying_question",
        question="Is the patient male or female?",
    ))
    # Should match triggers like "sex", "gender", "male", "female"
    assert obs.reward > 0
