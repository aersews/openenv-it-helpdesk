# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Helpdesk Env Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Dict


class HelpdeskAction(Action):
    """Action for the Helpdesk Env environment."""
    tool_name: str = Field(..., description="The tool to use. Options: 'get_tickets', 'read_ticket', 'assign_ticket', 'resolve_ticket', 'run_diagnostic', 'restart_service'.")
    tool_args: Dict[str, str] = Field(default_factory=dict, description="Arguments for the tool (e.g. {'ticket_id': 'T-1001', 'department': 'hardware_support'}).")


class HelpdeskObservation(Observation):
    """Observation from the Helpdesk Env environment."""
    command_result: str = Field(default="", description="The text output of the tool used.")
    open_tickets_summary: str = Field(default="", description="Short summary of currently open tickets to keep context.")
