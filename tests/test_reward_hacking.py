"""
Tests for reward hacking resistance.

Verifies that degenerate strategies (flag everything, skip investigation)
are penalized by the grader and reward function.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.environment import MedicalCodingEnvironment
from models import MedicalCodingAction


def test_flag_everything_penalized():
    """Flagging all codes should be penalized by false positive deductions."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    # Flag every code as demographic_mismatch
    for code in ["E11.9", "Z00.00", "O80"]:
        env.step(MedicalCodingAction(
            action_type="flag_error",
            code=code,
            error_type="demographic_mismatch",
            justification="Flagging everything.",
        ))

    obs = env.step(MedicalCodingAction(action_type="submit_audit"))

    # With 2 false positives (E11.9, Z00.00), the FP penalty should reduce score
    # significantly even though O80 was correctly flagged
    assert obs.grader_score is not None
    assert obs.grader_score < 0.8, \
        f"Flag-everything strategy should be penalized, got {obs.grader_score}"


def test_submit_without_flags_low_score():
    """Submitting immediately without flagging anything should score low."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")
    obs = env.step(MedicalCodingAction(action_type="submit_audit"))
    assert obs.grader_score is not None
    assert obs.grader_score < 0.3, \
        f"Empty audit should score low, got {obs.grader_score}"


def test_flag_without_investigation_penalized():
    """Flagging without prior investigation should get no coherence bonus."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    # Flag without querying first
    obs_no_query = env.step(MedicalCodingAction(
        action_type="flag_error",
        code="O80",
        error_type="demographic_mismatch",
        justification="O80 is maternity code, patient is male.",
    ))

    # Reset and flag WITH querying first
    env2 = MedicalCodingEnvironment()
    env2.reset(task_id="easy_demographic")
    env2.step(MedicalCodingAction(action_type="query_guideline", code="O80"))
    obs_with_query = env2.step(MedicalCodingAction(
        action_type="flag_error",
        code="O80",
        error_type="demographic_mismatch",
        justification="O80 is maternity code, patient is male.",
    ))

    # The investigated flag should have higher reward
    assert obs_with_query.reward > obs_no_query.reward


def test_hallucinated_code_flag_heavily_penalized():
    """Flagging a code NOT in the proposed set should get -0.50."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="easy_demographic")

    obs = env.step(MedicalCodingAction(
        action_type="flag_error",
        code="A01.00",
        error_type="untraceable_code",
        justification="Hallucinated code.",
    ))
    assert obs.reward == -0.50


def test_expert_flag_all_heavily_penalized():
    """On expert task with 6 codes, flagging everything should score poorly."""
    env = MedicalCodingEnvironment()
    env.reset(task_id="expert_multi_error")

    codes = ["E10.9", "E11.9", "93306", "93308", "99214", "M79.621"]
    for code in codes:
        env.step(MedicalCodingAction(
            action_type="flag_error",
            code=code,
            error_type="demographic_mismatch",
            justification="Flagging everything.",
        ))

    obs = env.step(MedicalCodingAction(action_type="submit_audit"))
    assert obs.grader_score is not None
    # 4 false positives * 0.15 = 0.60 penalty
    assert obs.grader_score < 0.5, \
        f"Expert flag-everything should be heavily penalized, got {obs.grader_score}"
