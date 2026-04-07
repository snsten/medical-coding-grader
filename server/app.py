"""
FastAPI application for the Medical Coding Auditor Environment.

Exposes the MedicalCodingEnvironment over HTTP and WebSocket endpoints
compatible with the OpenEnv EnvClient.

Endpoints:
    POST /reset   — Reset the environment (accepts optional task_id in body)
    POST /step    — Execute an action
    GET  /state   — Get current environment state
    GET  /schema  — Get action/observation JSON schemas
    GET  /health  — Health check
    WS   /ws      — WebSocket endpoint for persistent sessions

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

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


app = create_app(
    MedicalCodingEnvironment,
    MedicalCodingAction,
    MedicalCodingObservation,
    env_name="medical_coding",
    max_concurrent_envs=10,
)


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
    main()  # runs with argparse defaults when called without arguments
