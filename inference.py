"""
Inference Script — Medical Coding Auditor Environment
=======================================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL      The API endpoint for the LLM (default: "<your-active-endpoint>").
    MODEL_NAME        The model identifier to use for inference (default: "<your-active-model>").
    HF_TOKEN          Your Hugging Face / API key (used as API key for LLM calls). No default.
    LOCAL_IMAGE_NAME  (optional) Docker image name if launching env via from_docker_image().
    ENV_BASE_URL      (optional) Base URL of the running env server (default: http://localhost:7860).

STDOUT FORMAT (required by hackathon spec):
    [START] task=<task_id> env=medical_coding model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "medical_coding"

TASK_IDS = [
    "easy_demographic",
    "medium_ncci_conflict",
    "medium_excludes1",
    "hard_specificity_untraceable",
    "expert_multi_error",
]

MAX_STEPS_PER_TASK: Dict[str, int] = {
    "easy_demographic": 10,
    "medium_ncci_conflict": 15,
    "medium_excludes1": 15,
    "hard_specificity_untraceable": 20,
    "expert_multi_error": 25,
}

SUCCESS_THRESHOLD = 0.5
TEMPERATURE = 0.2
MAX_TOKENS = 400

# ---------------------------------------------------------------------------
# Logging (required stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Medical Coding Auditor performing a pre-bill compliance review.

Your task: review proposed ICD-10-CM and CPT billing codes against the patient's
clinical note and demographics. Identify coding errors WITHOUT assigning new codes.

Available tools (respond with exactly ONE JSON object per turn):

1. query_guideline — look up official coding guidelines for a specific code.
   {"action_type": "query_guideline", "code": "<CODE>"}

2. check_ncci_edits — check if two CPT codes have a CMS NCCI PTP bundling conflict.
   {"action_type": "check_ncci_edits", "code1": "<CPT1>", "code2": "<CPT2>"}

3. flag_error — record a confirmed coding error in the audit report.
   {"action_type": "flag_error", "code": "<CODE>", "error_type": "<TYPE>", "justification": "<REASON>"}
   error_type values:
     demographic_mismatch  — code inapplicable to patient's sex or age
     excludes1_conflict    — two mutually exclusive ICD-10-CM diagnosis codes
     ncci_edit             — CMS NCCI PTP bundling violation
     specificity_error     — wrong 7th character or insufficient code specificity
     untraceable_code      — code does not exist in any official code set

4. ask_clarifying_question — ask the physician for missing clinical information when the note is ambiguous.
   {"action_type": "ask_clarifying_question", "question": "<YOUR QUESTION>"}
   Use when you need to confirm: patient sex/age, encounter type (initial vs. follow-up), or diagnosis history.

5. submit_audit — submit the completed audit report when you are done.
   {"action_type": "submit_audit"}

RULES:
- Only query or flag codes that appear in the proposed_codes list.
- Query each code's guidelines before flagging it.
- For CPT code pairs, use check_ncci_edits to find bundling conflicts.
- If query_guideline returns "CODE NOT FOUND" or "INVALID/UNTRACEABLE", flag it as untraceable_code.
- Check patient sex — maternity (O-codes) ONLY apply to female patients.
- Check 7th character on injury codes against the clinical note (initial vs. subsequent encounter).
- Use ask_clarifying_question if demographics or encounter type are unclear before flagging.
- Submit the audit only after reviewing all proposed codes.
- Respond with ONLY a valid JSON object — no markdown, no extra text.
""").strip()


def build_user_prompt(obs: Dict[str, Any], step: int, history: List[str]) -> str:
    proposed = obs.get("proposed_codes", {})
    codes_summary = "\n".join(
        f"  {code}: {info.get('description', '')} [{info.get('code_type', '')}]"
        for code, info in proposed.items()
    )
    draft = obs.get("draft_report", [])
    draft_summary = json.dumps(draft, indent=2) if draft else "  (none yet)"
    queried = obs.get("codes_queried", [])
    pairs_checked = obs.get("pairs_checked", [])
    last_result = obs.get("tool_result", "")[:700]

    history_block = "\n".join(history[-5:]) if history else "None"

    return textwrap.dedent(f"""
    === STEP {step} ===
    PATIENT: age={obs.get('patient_demographics', {}).get('age')}, sex={obs.get('patient_demographics', {}).get('sex')}
    CLINICAL NOTE:
    {obs.get('clinical_note', '')[:900]}

    PROPOSED CODES TO AUDIT:
    {codes_summary}

    Already queried: {queried}
    NCCI pairs checked: {pairs_checked}

    DRAFT AUDIT REPORT:
    {draft_summary}

    LAST TOOL RESULT:
    {last_result}

    RECENT HISTORY:
    {history_block}

    What is your next action? Reply with exactly one JSON object.
    """).strip()


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON action from the LLM response."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end])
            if isinstance(obj, dict) and "action_type" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None


def get_llm_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    history: List[str],
) -> Dict[str, Any]:
    """Call the LLM and parse its action."""
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        action = parse_action(text)
        if action:
            return action
        print(f"[DEBUG] Could not parse action from LLM output: {text[:300]}", flush=True)
    except Exception as exc:
        print(f"[DEBUG] LLM error at step {step}: {exc}", flush=True)

    # Fallback: query unchecked codes, then submit
    queried = obs.get("codes_queried", [])
    unchecked = [c for c in obs.get("proposed_codes", {}) if c not in queried]
    if unchecked:
        return {"action_type": "query_guideline", "code": unchecked[0]}
    return {"action_type": "submit_audit"}


# ---------------------------------------------------------------------------
# WebSocket-based environment runner
# ---------------------------------------------------------------------------

async def run_task_websocket(
    ws_url: str,
    task_id: str,
    client: OpenAI,
) -> float:
    """
    Run one task episode over a WebSocket connection to maintain state.
    Returns the grader_score from the terminal observation (already in [0.0, 1.0]).
    """
    import websockets

    max_steps = MAX_STEPS_PER_TASK[task_id]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
            # Reset with task_id
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
            raw = await ws.recv()
            response = json.loads(raw)
            obs_payload = response.get("data", {})
            obs = obs_payload.get("observation", obs_payload)
            done = obs_payload.get("done", False)

            history: List[str] = []

            for step in range(1, max_steps + 1):
                if done:
                    break

                action_dict = get_llm_action(client, obs, step, history)
                action_str = json.dumps(action_dict)

                # Send step
                await ws.send(json.dumps({"type": "step", "data": action_dict}))
                raw = await ws.recv()
                step_response = json.loads(raw)
                step_data = step_response.get("data", {})

                obs = step_data.get("observation", obs)
                reward = step_data.get("reward") or 0.0
                done = step_data.get("done", False)
                error = obs.get("last_action_error") if isinstance(obs, dict) else None

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                history.append(
                    f"Step {step}: {action_str[:120]} → reward={reward:+.2f}"
                    + (f" [ERROR: {error}]" if error else "")
                )

                if done:
                    # Use grader_score from terminal observation as the score
                    grader = obs.get("grader_score") if isinstance(obs, dict) else None
                    if grader is not None:
                        score = float(grader)
                    break

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] WebSocket task {task_id} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# HTTP-based fallback runner (stateless — demonstrates single-step interactions)
# ---------------------------------------------------------------------------

async def run_task_http(
    base_url: str,
    task_id: str,
    client: OpenAI,
) -> float:
    """
    HTTP-based stateless runner. Since the OpenEnv HTTP endpoints are stateless
    (each request creates a fresh environment), this runner passes the full
    state from the reset observation into a stateful local environment instance
    for demonstration.
    """
    import httpx
    # Import environment directly for stateful HTTP-like simulation
    import sys
    import os
    env_dir = os.path.dirname(os.path.abspath(__file__))
    if env_dir not in sys.path:
        sys.path.insert(0, env_dir)

    from server.environment import MedicalCodingEnvironment
    from models import MedicalCodingAction

    env = MedicalCodingEnvironment()
    max_steps = MAX_STEPS_PER_TASK[task_id]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs_obj = env.reset(task_id=task_id)
        obs = obs_obj.model_dump()
        done = obs_obj.done
        history: List[str] = []

        for step in range(1, max_steps + 1):
            if done:
                break

            action_dict = get_llm_action(client, obs, step, history)
            action_str = json.dumps(action_dict)

            try:
                action = MedicalCodingAction(**action_dict)
                step_obs = env.step(action)
                obs = step_obs.model_dump()
                reward = step_obs.reward or 0.0
                done = step_obs.done
                error = step_obs.last_action_error
            except Exception as e:
                reward = -0.2
                done = False
                error = str(e)
                print(f"[DEBUG] Step parse error: {e}", flush=True)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str[:120]} → reward={reward:+.2f}"
                + (f" [ERROR: {error}]" if error else "")
            )

            if done:
                # Use grader_score from terminal observation as the score
                if step_obs.grader_score is not None:
                    score = float(step_obs.grader_score)
                break

    except Exception as exc:
        print(f"[DEBUG] HTTP task {task_id} error: {exc}", flush=True)

    finally:
        success = score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run baseline inference across all three tasks."""
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # If LOCAL_IMAGE_NAME is set, launch env from Docker image
    if LOCAL_IMAGE_NAME:
        from client import MedicalCodingEnv
        env = await MedicalCodingEnv.from_docker_image(LOCAL_IMAGE_NAME)
        base_url = env.base_url.rstrip("/")
    else:
        base_url = ENV_BASE_URL.rstrip("/")

    # Try WebSocket mode first, fall back to HTTP-local mode
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    # Check if websockets library is available
    try:
        import websockets
        use_ws = True
    except ImportError:
        use_ws = False
        print("[DEBUG] websockets not installed, using HTTP/local mode", flush=True)

    task_scores: Dict[str, float] = {}

    for task_id in TASK_IDS:
        if use_ws:
            score = await run_task_websocket(ws_url, task_id, client)
        else:
            score = await run_task_http(base_url, task_id, client)
        task_scores[task_id] = score

    print("\n" + "=" * 60, flush=True)
    print("BASELINE SCORES SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for task_id, score in task_scores.items():
        status = "PASS" if score >= SUCCESS_THRESHOLD else "FAIL"
        print(f"  [{status}] {task_id}: {score:.3f}", flush=True)
    avg = sum(task_scores.values()) / len(task_scores)
    print(f"  Average: {avg:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
