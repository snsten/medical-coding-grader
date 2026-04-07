"""
Medical Coding Auditor Environment Client.

Provides a typed EnvClient for connecting to the MedicalCodingEnvironment server
over HTTP/WebSocket. Supports both direct URL connections and Docker-based launch.
"""

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MedicalCodingAction, MedicalCodingObservation


class MedicalCodingEnv(
    EnvClient[MedicalCodingAction, MedicalCodingObservation, State]
):
    """
    Client for the Medical Coding Auditor Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated session.

    Example — connect to a running server:
        >>> env = MedicalCodingEnv(base_url="http://localhost:7860")
        >>> result = await env.reset(task_id="easy_demographic")
        >>> obs = result.observation
        >>> print(obs.clinical_note)
        >>>
        >>> result = await env.step(
        ...     MedicalCodingAction(action_type="query_guideline", code="O80")
        ... )
        >>> print(result.observation.tool_result)

    Example — launch from Docker image:
        >>> env = await MedicalCodingEnv.from_docker_image("medical-coding-env:latest")
        >>> try:
        ...     result = await env.reset(task_id="medium_ncci_conflict")
        ...     result = await env.step(
        ...         MedicalCodingAction(
        ...             action_type="check_ncci_edits",
        ...             code1="93306",
        ...             code2="93307",
        ...         )
        ...     )
        ... finally:
        ...     await env.close()
    """

    def _step_payload(self, action: MedicalCodingAction) -> Dict[str, Any]:
        """Serialize MedicalCodingAction to JSON payload for the step endpoint."""
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
        }
        if action.code is not None:
            payload["code"] = action.code
        if action.code1 is not None:
            payload["code1"] = action.code1
        if action.code2 is not None:
            payload["code2"] = action.code2
        if action.error_type is not None:
            payload["error_type"] = action.error_type
        if action.justification is not None:
            payload["justification"] = action.justification
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MedicalCodingObservation]:
        """Parse server response into StepResult[MedicalCodingObservation]."""
        obs_data = payload.get("observation", {})
        observation = MedicalCodingObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            patient_demographics=obs_data.get("patient_demographics", {}),
            clinical_note=obs_data.get("clinical_note", ""),
            proposed_codes=obs_data.get("proposed_codes", {}),
            draft_report=obs_data.get("draft_report", []),
            tool_result=obs_data.get("tool_result", ""),
            step_count=obs_data.get("step_count", 0),
            codes_queried=obs_data.get("codes_queried", []),
            pairs_checked=obs_data.get("pairs_checked", []),
            last_action_error=obs_data.get("last_action_error"),
            grader_score=obs_data.get("grader_score"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
