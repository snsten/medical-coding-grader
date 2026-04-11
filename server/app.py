"""
FastAPI application for the Medical Coding Auditor Environment.

Exposes the MedicalCodingEnvironment over HTTP and WebSocket endpoints
compatible with the OpenEnv EnvClient, plus session-based API endpoints
for HF Spaces compatibility (avoids WebSocket proxy timeout issues).

Endpoints:
    # OpenEnv standard (via create_app)
    POST /reset   — Reset the environment (accepts optional task_id in body)
    POST /step    — Execute an action
    GET  /state   — Get current environment state
    GET  /schema  — Get action/observation JSON schemas
    GET  /health  — Health check
    WS   /ws      — WebSocket endpoint for persistent sessions

    # Session-based API (HF Spaces compatible)
    POST /api/reset      — Create a new session, returns {session_id, observation}
    POST /api/call_tool  — Execute action in session, returns {observation, reward, done}
    POST /api/close      — Terminate session and free memory

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import asyncio
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from exc

try:
    from ..models import MedicalCodingAction, MedicalCodingObservation
    from .environment import MedicalCodingEnvironment
except ImportError:
    from models import MedicalCodingAction, MedicalCodingObservation
    from server.environment import MedicalCodingEnvironment


# ---------------------------------------------------------------------------
# OpenEnv standard app (provides /reset, /step, /state, /schema, /health, /ws)
# ---------------------------------------------------------------------------
app = create_app(
    MedicalCodingEnvironment,
    MedicalCodingAction,
    MedicalCodingObservation,
    env_name="medical_coding",
    max_concurrent_envs=10,
)


# ---------------------------------------------------------------------------
# Session-based API — HF Spaces compatible (no WebSocket needed)
# ---------------------------------------------------------------------------
_sessions: Dict[str, MedicalCodingEnvironment] = {}
_sessions_lock = asyncio.Lock()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class CallToolRequest(BaseModel):
    session_id: str
    action_type: str
    code: Optional[str] = None
    code1: Optional[str] = None
    code2: Optional[str] = None
    error_type: Optional[str] = None
    justification: Optional[str] = None
    question: Optional[str] = None
    evidence_text: Optional[str] = None


class CloseRequest(BaseModel):
    session_id: str


@app.post("/api/reset")
async def api_reset(req: ResetRequest) -> Dict[str, Any]:
    """Create a new session and reset the environment."""
    session_id = str(uuid.uuid4())
    env = MedicalCodingEnvironment()

    task_id = req.task_id or "easy_demographic"
    obs = env.reset(task_id=task_id)

    async with _sessions_lock:
        _sessions[session_id] = env

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
    }


@app.post("/api/call_tool")
async def api_call_tool(req: CallToolRequest) -> Dict[str, Any]:
    """Execute an action within an existing session."""
    async with _sessions_lock:
        env = _sessions.get(req.session_id)

    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")

    action_data = {"action_type": req.action_type}
    if req.code is not None:
        action_data["code"] = req.code
    if req.code1 is not None:
        action_data["code1"] = req.code1
    if req.code2 is not None:
        action_data["code2"] = req.code2
    if req.error_type is not None:
        action_data["error_type"] = req.error_type
    if req.justification is not None:
        action_data["justification"] = req.justification
    if req.question is not None:
        action_data["question"] = req.question
    if req.evidence_text is not None:
        action_data["evidence_text"] = req.evidence_text

    try:
        action = MedicalCodingAction(**action_data)
        obs = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }

    # Auto-cleanup completed sessions
    if obs.done:
        async with _sessions_lock:
            _sessions.pop(req.session_id, None)

    return result


@app.post("/api/close")
async def api_close(req: CloseRequest) -> Dict[str, str]:
    """Terminate a session and free memory."""
    async with _sessions_lock:
        removed = _sessions.pop(req.session_id, None)

    if removed is None:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")

    return {"status": "closed", "session_id": req.session_id}


# ---------------------------------------------------------------------------
# Gradio landing page mount (if gradio is available)
# ---------------------------------------------------------------------------
try:
    import gradio as gr
    from .gradio_landing import create_gradio_app
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")
except ImportError:
    pass
except Exception:
    # Gradio UI is optional — don't break the server if it fails
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
