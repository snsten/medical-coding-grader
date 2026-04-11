"""
Gradio Landing Page for the Medical Coding Auditor Environment.

Provides a custom UI dashboard for Hugging Face Spaces with:
  - Overview & Scenarios: task descriptions, difficulty, expected errors
  - Leaderboard: baseline model comparison table
  - Playground: interactive episode runner for manual exploration

Mounted at /ui by app.py when gradio is available.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

import gradio as gr

try:
    from .environment import MedicalCodingEnvironment
except ImportError:
    from server.environment import MedicalCodingEnvironment

try:
    from ..models import MedicalCodingAction
except ImportError:
    try:
        from models import MedicalCodingAction
    except ImportError:
        from medical_coding_env.models import MedicalCodingAction


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
_sessions: Dict[str, MedicalCodingEnvironment] = {}

# ---------------------------------------------------------------------------
# Task data for the Overview tab
# ---------------------------------------------------------------------------
TASKS_INFO = [
    {
        "id": "easy_demographic",
        "difficulty": "Easy",
        "description": "Identify a demographic mismatch — maternity code (O80) on a male patient.",
        "expected_errors": "O80 -> demographic_mismatch",
        "max_steps": 10,
    },
    {
        "id": "medium_ncci_conflict",
        "difficulty": "Medium",
        "description": "Identify a CMS NCCI PTP bundling violation between echocardiography codes (93306 + 93307).",
        "expected_errors": "93306 -> ncci_edit",
        "max_steps": 15,
    },
    {
        "id": "medium_excludes1",
        "difficulty": "Medium",
        "description": "Identify an Excludes1 conflict between COPD (J44.1) and asthma (J45.20) codes.",
        "expected_errors": "J44.1 -> excludes1_conflict",
        "max_steps": 15,
    },
    {
        "id": "hard_specificity_untraceable",
        "difficulty": "Hard",
        "description": "Identify a 7th character specificity error (S52.501A) and an untraceable code (Z99.999).",
        "expected_errors": "S52.501A -> specificity_error, Z99.999 -> untraceable_code",
        "max_steps": 20,
    },
    {
        "id": "expert_multi_error",
        "difficulty": "Expert",
        "description": "Identify two errors in a complex encounter: Excludes1 (E10.9+E11.9) and NCCI (93306+93308).",
        "expected_errors": "E10.9 -> excludes1_conflict, 93306 -> ncci_edit",
        "max_steps": 25,
    },
]

LEADERBOARD_DATA = [
    ["Qwen/Qwen2.5-72B-Instruct", "~0.75", "~0.65", "~0.65", "~0.55", "~0.40", "~0.60"],
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.task-card {
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background: #0d1117;
}
.difficulty-easy { color: #3fb950; font-weight: bold; }
.difficulty-medium { color: #d29922; font-weight: bold; }
.difficulty-hard { color: #f85149; font-weight: bold; }
.difficulty-expert { color: #bc4dff; font-weight: bold; }
.action-output {
    font-family: monospace;
    font-size: 13px;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
}
"""

DIFFICULTY_COLORS = {
    "Easy": "difficulty-easy",
    "Medium": "difficulty-medium",
    "Hard": "difficulty-hard",
    "Expert": "difficulty-expert",
}


# ---------------------------------------------------------------------------
# Playground logic
# ---------------------------------------------------------------------------
def playground_reset(task_id: str) -> tuple:
    """Reset environment for playground session."""
    session_id = str(uuid.uuid4())
    env = MedicalCodingEnvironment()
    obs = env.reset(task_id=task_id)
    _sessions[session_id] = env
    obs_dict = obs.model_dump()

    demographics = obs_dict.get("patient_demographics", {})
    info = (
        f"Session: {session_id}\n"
        f"Task: {task_id} ({obs_dict.get('difficulty', '')})\n"
        f"Patient: {demographics.get('age', '?')}yo {demographics.get('sex', '?')}\n"
        f"Proposed codes: {', '.join(obs_dict.get('proposed_codes', {}).keys())}\n\n"
        f"--- Clinical Note ---\n{obs_dict.get('clinical_note', '')[:1200]}"
    )

    return session_id, info, "", "Episode started. Use the action panel to interact."


def playground_step(
    session_id: str,
    action_type: str,
    code: str,
    code1: str,
    code2: str,
    error_type: str,
    justification: str,
    question: str,
    evidence_text: str,
) -> tuple:
    """Execute one step in the playground."""
    env = _sessions.get(session_id)
    if env is None:
        return session_id, "No active session. Reset first.", ""

    action_data: Dict[str, Any] = {"action_type": action_type}
    if code.strip():
        action_data["code"] = code.strip()
    if code1.strip():
        action_data["code1"] = code1.strip()
    if code2.strip():
        action_data["code2"] = code2.strip()
    if error_type.strip() and error_type != "(none)":
        action_data["error_type"] = error_type
    if justification.strip():
        action_data["justification"] = justification.strip()
    if question.strip():
        action_data["question"] = question.strip()
    if evidence_text.strip():
        action_data["evidence_text"] = evidence_text.strip()

    try:
        action = MedicalCodingAction(**action_data)
        obs = env.step(action)
    except Exception as e:
        return session_id, f"Error: {e}", ""

    obs_dict = obs.model_dump()
    tool_result = obs_dict.get("tool_result", "")
    reward = obs_dict.get("reward", 0.0)
    done = obs_dict.get("done", False)
    error = obs_dict.get("last_action_error")
    step_count = obs_dict.get("step_count", 0)

    status_parts = [
        f"Step {step_count} | Reward: {reward:+.2f} | Done: {done}",
    ]
    if error:
        status_parts.append(f"Error: {error}")
    if done:
        grader = obs_dict.get("grader_score")
        if grader is not None:
            status_parts.append(f"Grader Score: {grader:.3f}")
        metrics = obs_dict.get("episode_metrics")
        if metrics:
            status_parts.append(f"Metrics: {json.dumps(metrics, indent=2)}")
        # Cleanup session
        _sessions.pop(session_id, None)

    draft = obs_dict.get("draft_report", [])
    draft_str = json.dumps(draft, indent=2) if draft else "(none)"

    result = (
        f"--- Tool Result ---\n{tool_result}\n\n"
        f"--- Draft Report ---\n{draft_str}\n\n"
        f"Codes queried: {obs_dict.get('codes_queried', [])}\n"
        f"NCCI pairs checked: {obs_dict.get('pairs_checked', [])}\n"
        f"Evidence extracted: {obs_dict.get('extracted_evidence', [])}\n"
        f"Clarifications: {obs_dict.get('clarifications_asked', [])}"
    )

    return session_id, result, "\n".join(status_parts)


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(
        title="Medical Coding Auditor - OpenEnv",
    ) as demo:

        gr.Markdown(
            "# Medical Coding Auditor\n"
            "A real-world RL environment simulating hospital pre-bill compliance review. "
            "The AI agent reviews ICD-10-CM and CPT billing codes against clinical notes "
            "to identify coding errors."
        )

        with gr.Tabs():
            # ----------------------------------------------------------
            # Tab 1: Overview & Scenarios
            # ----------------------------------------------------------
            with gr.Tab("Overview & Scenarios"):
                gr.Markdown("## Tasks\n5 tasks across 4 difficulty levels testing different medical coding error types.")

                tasks_md = ""
                for t in TASKS_INFO:
                    color_class = DIFFICULTY_COLORS.get(t["difficulty"], "")
                    tasks_md += (
                        f"### {t['id']}\n"
                        f"**Difficulty:** <span class='{color_class}'>{t['difficulty']}</span> "
                        f"| **Max steps:** {t['max_steps']}\n\n"
                        f"{t['description']}\n\n"
                        f"**Expected errors:** `{t['expected_errors']}`\n\n---\n\n"
                    )
                gr.Markdown(tasks_md)

                gr.Markdown(
                    "## Action Space\n\n"
                    "| Action | Required Fields | Description |\n"
                    "|---|---|---|\n"
                    "| `query_guideline` | `code` | Look up ICD-10-CM or CPT coding guidelines |\n"
                    "| `check_ncci_edits` | `code1`, `code2` | Check NCCI PTP bundling conflict |\n"
                    "| `flag_error` | `code`, `error_type`, `justification` | Record a coding error |\n"
                    "| `ask_clarifying_question` | `question` | Ask physician for clarification |\n"
                    "| `extract_evidence` | `evidence_text` | Highlight clinical note text as evidence |\n"
                    "| `submit_audit` | _(none)_ | Submit audit report for grading |\n"
                )

                gr.Markdown(
                    "## Error Types\n\n"
                    "| Type | Description | Rarity |\n"
                    "|---|---|---|\n"
                    "| `demographic_mismatch` | Code inapplicable to patient sex/age | 1.0x |\n"
                    "| `excludes1_conflict` | Mutually exclusive ICD-10 codes | 1.2x |\n"
                    "| `ncci_edit` | NCCI PTP bundling violation | 1.2x |\n"
                    "| `untraceable_code` | Code doesn't exist in any code set | 1.3x |\n"
                    "| `specificity_error` | Wrong 7th character / insufficient specificity | 1.5x |\n"
                )

            # ----------------------------------------------------------
            # Tab 2: Leaderboard
            # ----------------------------------------------------------
            with gr.Tab("Leaderboard"):
                gr.Markdown("## Baseline Scores\nEvaluated at temperature 0.2 with dense reward shaping.")
                gr.Dataframe(
                    headers=[
                        "Model",
                        "easy_demographic",
                        "medium_ncci",
                        "medium_excludes1",
                        "hard_specificity",
                        "expert_multi",
                        "Average",
                    ],
                    value=LEADERBOARD_DATA,
                    interactive=False,
                )
                gr.Markdown(
                    "*Scores reflect grader output (0.0-1.0) including efficiency bonus and FP penalties. "
                    "Submit your agent's scores to the leaderboard!*"
                )

            # ----------------------------------------------------------
            # Tab 3: Playground
            # ----------------------------------------------------------
            with gr.Tab("Playground"):
                gr.Markdown(
                    "## Interactive Playground\n"
                    "Step through a task manually to explore the environment."
                )

                with gr.Row():
                    task_dropdown = gr.Dropdown(
                        choices=[t["id"] for t in TASKS_INFO],
                        value="easy_demographic",
                        label="Task",
                    )
                    reset_btn = gr.Button("Reset", variant="primary")

                session_id_state = gr.State("")

                case_info = gr.Textbox(
                    label="Case Information",
                    lines=12,
                    interactive=False,
                )

                gr.Markdown("### Action Panel")
                with gr.Row():
                    action_type_dropdown = gr.Dropdown(
                        choices=[
                            "query_guideline",
                            "check_ncci_edits",
                            "flag_error",
                            "ask_clarifying_question",
                            "extract_evidence",
                            "submit_audit",
                        ],
                        value="query_guideline",
                        label="Action Type",
                    )
                    step_btn = gr.Button("Execute Step", variant="secondary")

                with gr.Row():
                    code_input = gr.Textbox(label="code", placeholder="e.g. O80, E11.9")
                    code1_input = gr.Textbox(label="code1", placeholder="CPT code 1")
                    code2_input = gr.Textbox(label="code2", placeholder="CPT code 2")

                with gr.Row():
                    error_type_input = gr.Dropdown(
                        choices=[
                            "(none)",
                            "demographic_mismatch",
                            "excludes1_conflict",
                            "ncci_edit",
                            "specificity_error",
                            "untraceable_code",
                        ],
                        value="(none)",
                        label="error_type",
                    )
                    justification_input = gr.Textbox(
                        label="justification",
                        placeholder="Clinical rationale for the error",
                    )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="question",
                        placeholder="Clarifying question for physician",
                    )
                    evidence_input = gr.Textbox(
                        label="evidence_text",
                        placeholder="Exact text span from clinical note",
                    )

                status_output = gr.Textbox(label="Status", lines=3, interactive=False)
                result_output = gr.Textbox(
                    label="Result",
                    lines=15,
                    interactive=False,
                    elem_classes=["action-output"],
                )

                reset_btn.click(
                    fn=playground_reset,
                    inputs=[task_dropdown],
                    outputs=[session_id_state, case_info, result_output, status_output],
                )
                step_btn.click(
                    fn=playground_step,
                    inputs=[
                        session_id_state,
                        action_type_dropdown,
                        code_input,
                        code1_input,
                        code2_input,
                        error_type_input,
                        justification_input,
                        question_input,
                        evidence_input,
                    ],
                    outputs=[session_id_state, result_output, status_output],
                )

            # ----------------------------------------------------------
            # Tab 4: API Reference
            # ----------------------------------------------------------
            with gr.Tab("API Reference"):
                gr.Markdown(
                    "## API Endpoints\n\n"
                    "### OpenEnv Standard\n"
                    "```\n"
                    "POST /reset          Reset environment (body: {task_id: string})\n"
                    "POST /step           Execute action (body: MedicalCodingAction)\n"
                    "GET  /state          Get current state\n"
                    "GET  /health         Health check\n"
                    "WS   /ws             WebSocket session\n"
                    "```\n\n"
                    "### Session-Based (HF Spaces Compatible)\n"
                    "```\n"
                    "POST /api/reset      Create session (body: {task_id})\n"
                    "                     Returns: {session_id, observation}\n\n"
                    "POST /api/call_tool  Execute action (body: {session_id, action_type, ...})\n"
                    "                     Returns: {observation, reward, done}\n\n"
                    "POST /api/close      Terminate session (body: {session_id})\n"
                    "```\n\n"
                    "### Example Usage\n"
                    "```python\n"
                    "import httpx\n\n"
                    "# Create session\n"
                    "resp = httpx.post(BASE_URL + '/api/reset',\n"
                    "                  json={'task_id': 'easy_demographic'})\n"
                    "data = resp.json()\n"
                    "session_id = data['session_id']\n\n"
                    "# Query a code\n"
                    "resp = httpx.post(BASE_URL + '/api/call_tool', json={\n"
                    "    'session_id': session_id,\n"
                    "    'action_type': 'query_guideline',\n"
                    "    'code': 'O80'\n"
                    "})\n"
                    "print(resp.json()['observation']['tool_result'])\n"
                    "```\n"
                )

    return demo
