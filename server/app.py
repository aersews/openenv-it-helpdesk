# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Helpdesk Env Environment.

This module creates an HTTP server that exposes the HelpdeskEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import HelpdeskAction, HelpdeskObservation
    from .helpdesk_env_environment import HelpdeskEnvironment
except ImportError:
    from models import HelpdeskAction, HelpdeskObservation
    from server.helpdesk_env_environment import HelpdeskEnvironment


from fastapi import Request
import yaml
import os

# Create the app with web interface and README integration
app = create_app(
    HelpdeskEnvironment,
    HelpdeskAction,
    HelpdeskObservation,
    env_name="helpdesk_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

@app.get("/tasks")
def list_tasks():
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config.get("tasks", [])

@app.post("/grader")
async def evaluate_task(request: Request):
    data = await request.json()
    episode = data.get("episode", data)
    
    try:
        from graders import _compute_score
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from graders import _compute_score

    score = _compute_score(episode)
    return {"score": score}

def main():
    """
    Entry point for direct execution.
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == '__main__':
    main()
