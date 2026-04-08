---
title: Helpdesk Env
emoji: 📞
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# IT Helpdesk & Diagnostics OpenEnv Environment

An OpenEnv RL execution environment that simulates an IT support helpdesk. The AI agent must interact with a ticketing system and simulated technical backends to resolve user issues.

## Environment Overview

This environment evaluates an agent's ability to act as an IT professional by reading tickets, diagnosing underlying system problems through technical tools, and performing resolving actions.

### 3 Built-In Scenarios (Graded)
The environment randomly selects one of three issues upon `reset()`:
1. **Easy**: "Laptop very slow". The agent must recognize a client hardware issue and correctly assign it to the `hardware_support` department.
2. **Medium**: "Account locked". The agent must check the directory and use the `restart_service` tool to unlock the user account for `johndoe`.
3. **Hard**: "Website is down!". The agent must use the `run_diagnostic` tool to uncover that the database is offline, then restart the correct `database` service to restore operations.

## Action Space
Agents interact via the `HelpdeskAction` Pydantic model:
- `tool_name` (str): Options are `get_tickets`, `read_ticket`, `assign_ticket`, `resolve_ticket`, `run_diagnostic`, `restart_service`.
- `tool_args` (dict): Keyword arguments for the tool (e.g. `{"ticket_id": "T-1001", "department": "hardware_support"}`).

## Observation Space
The environment returns a `HelpdeskObservation` containing:
- `command_result` (str): The standard output of the tool execution.
- `open_tickets_summary` (str): A summary of the currently open tickets.
- Standard OpenEnv fields: `reward` and `done`.

## Scoring
The environment has a meaningful reward function that grants `0.5` points for correctly diagnosing or escalating, and `0.5` for completely resolving the core issue (Total possible reward = 1.0).

---

## Setup & Testing

### 1. Run the Environment Server
```bash
uv sync  # Install dependencies
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 2. Run the Evaluator
Run the standardized evaluator (inference endpoint compatible with OpenAI schema).
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="sk-proj-xyz..."
python inference.py
```
*(The script will output `[START]`, `[STEP]`, and `[END]` strictly).*
