"""
Tests for environment reset/step lifecycle and observation schema.
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.environment import MedicalCodingEnvironment
from models import MedicalCodingAction, MedicalCodingObservation

TASK_IDS = [
    "easy_demographic",
    "medium_ncci_conflict",
    "medium_excludes1",
    "hard_specificity_untraceable",
    "expert_multi_error",
]


def test_reset_returns_valid_observation():
    """Reset should return a valid MedicalCodingObservation for every task."""
    env = MedicalCodingEnvironment()
    for task_id in TASK_IDS:
        obs = env.reset(task_id=task_id)
        assert isinstance(obs, MedicalCodingObservation)
        assert obs.task_id == task_id
        assert obs.clinical_note != ""
        assert len(obs.proposed_codes) > 0
        assert obs.done is False
        assert obs.grader_score is None


def test_easy_demographic_patient_is_male():
    """Easy task patient should be male (maternity code mismatch)."""
    env = MedicalCodingEnvironment()
    obs = env.reset(task_id="easy_demographic")
    assert obs.patient_demographics["sex"] == "male"
    assert obs.patient_demographics["age"] == 34


def test_query_guideline_returns_tool_result():
    """Querying a valid proposed code should return guideline text."""
    env = MedicalCodingEnvironment()
    obs = env.reset(task_id="easy_demographic")
    action = MedicalCodingAction(action_type="query_guideline", code="O80")
    obs = env.step(action)
    assert obs.tool_result != ""
    assert "O80" in obs.codes_queried
    assert obs.reward > 0  # first valid query = +0.10


def test_query_hallucinated_code_penalized():
    """Querying a code NOT in proposed set should get -0.50 penalty."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")
    action = MedicalCodingAction(action_type="query_guideline", code="X99.99")
    obs = env.step(action)
    assert obs.reward == -0.50


def test_submit_audit_ends_episode():
    """Submitting audit should end the episode and set grader_score."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")
    action = MedicalCodingAction(action_type="submit_audit")
    obs = env.step(action)
    assert obs.done is True
    assert obs.grader_score is not None
    assert 0.0 <= obs.grader_score <= 1.0


def test_full_correct_easy_task():
    """Complete the easy task correctly: query O80 -> flag -> submit."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    # Query guideline for O80
    obs = env.step(MedicalCodingAction(action_type="query_guideline", code="O80"))
    assert obs.reward > 0

    # Flag the error
    obs = env.step(MedicalCodingAction(
        action_type="flag_error",
        code="O80",
        error_type="demographic_mismatch",
        justification="O80 is a female-only maternity code. Patient is male.",
    ))
    assert obs.reward > 0

    # Submit
    obs = env.step(MedicalCodingAction(action_type="submit_audit"))
    assert obs.done is True
    assert obs.grader_score is not None
    assert obs.grader_score >= 0.5  # should pass threshold


def test_episode_metrics_populated_on_done():
    """Episode metrics should be populated when done=True."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")
    obs = env.step(MedicalCodingAction(action_type="submit_audit"))
    assert obs.done is True
    assert obs.episode_metrics is not None
    assert "trajectory_length" in obs.episode_metrics
    assert "flag_precision" in obs.episode_metrics
    assert "flag_recall" in obs.episode_metrics


def test_step_after_done_raises_or_noop():
    """Stepping after episode is done should not crash."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")
    env.step(MedicalCodingAction(action_type="submit_audit"))
    # A second step should either raise or return a done observation
    try:
        obs = env.step(MedicalCodingAction(action_type="query_guideline", code="O80"))
        # If it doesn't raise, it should still be done
        assert obs.done is True
    except Exception:
        pass  # Raising is also acceptable


def test_all_tasks_have_proposed_codes():
    """Every task should have at least 2 proposed codes."""
    env = MedicalCodingEnvironment()
    for task_id in TASK_IDS:
        obs = env.reset(task_id=task_id)
        assert len(obs.proposed_codes) >= 2, f"{task_id} has too few codes"
