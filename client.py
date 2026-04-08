# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpdesk Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import HelpdeskAction, HelpdeskObservation


class HelpdeskEnv(
    EnvClient[HelpdeskAction, HelpdeskObservation, State]
):
    """
    Client for the Helpdesk Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with HelpdeskEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(HelpdeskAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = HelpdeskEnv.from_docker_image("helpdesk_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(HelpdeskAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: HelpdeskAction) -> Dict:
        """
        Convert HelpdeskAction to JSON payload for step message.

        Args:
            action: HelpdeskAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[HelpdeskObservation]:
        """
        Parse server response into StepResult[HelpdeskObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with HelpdeskObservation
        """
        obs_data = payload.get("observation", {})
        observation = HelpdeskObservation(
            command_result=obs_data.get("command_result", ""),
            open_tickets_summary=obs_data.get("open_tickets_summary", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
