import os
import json
import asyncio
from openai import OpenAI
from helpdesk_env.client import HelpdeskEnv, HelpdeskAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.environ.get("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-key"
)

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    # Ensure action/error don't introduce newlines if possible
    safe_action = action.replace('\n', ' ') if action else ""
    print(f"[STEP] step={step} action={safe_action!r} reward={reward} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]):
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}", flush=True)

def get_model_message(obs, step, last_reward, history):
    history_str = "\n".join(history[-5:]) if history else "None"
    prompt = (
        f"Past History:\n{history_str}\n\n"
        f"Observation: {obs.command_result}\n"
        f"Tickets: {obs.open_tickets_summary}\n\n"
        "You are an IT helpdesk agent. Resolve the tickets. Output exactly ONE valid JSON object per turn and no other text. Example:\n"
        "{\"tool_name\": \"tool\", \"tool_args\": {\"arg\": \"val\"}}\n\n"
        "Tools available:\n"
        "- get_tickets (no args)\n"
        "- read_ticket (args: ticket_id)\n"
        "- run_diagnostic (args: system) (systems: portal, database, logs, johndoe)\n"
        "- restart_service (args: service) (services: database, johndoe_account)\n"
        "- assign_ticket (args: ticket_id, department) (Valid departments: hardware_support, IT, networks)\n"
        "- resolve_ticket (args: ticket_id)\n"
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ""

def run_agent():
    base_url = "http://127.0.0.1:8000"
    
    tasks = [
        {"id": "task_easy", "scenario": "easy", "max_steps": 10},
        {"id": "task_medium", "scenario": "medium", "max_steps": 12},
        {"id": "task_hard", "scenario": "hard", "max_steps": 15}
    ]
    
    env_client = HelpdeskEnv(base_url=base_url)
    with env_client.sync() as env:
        for t in tasks:
            task_id = t["id"]
            scenario = t["scenario"]
            max_steps = t["max_steps"]
            print(f"[DEBUG] Beginning task mapping to: {task_id}")
            
            obs_res = env.reset(scenario=scenario)
            obs = obs_res.observation
            state = env.state()
            
            log_start(task=task_id, env="helpdesk_env", model=MODEL_NAME)
            
            history = []
            rewards = []
            steps_taken = 0
            score = 0.0
            success = False
            
            last_reward = 0.0
            done = False
            
            for step_num in range(1, max_steps + 1):
                if done:
                    break
                    
                message = get_model_message(obs, step_num, last_reward, history)
                
                import re
                md_match = re.search(r'```(?:json)?\s*(.*?)\s*```', message, re.DOTALL)
                json_candidate = md_match.group(1) if md_match else message
                
                action_data = None
                for line in json_candidate.split('\n'):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            action_data = json.loads(line)
                            break
                        except Exception:
                            continue
                            
                if not action_data:
                    match = re.search(r'\{.*\}', json_candidate, re.DOTALL)
                    json_str = match.group(0) if match else json_candidate
                    try:
                        action_data = json.loads(json_str)
                    except Exception:
                        action_data = {"tool_name": "error", "tool_args": {"error": "unparseable json"}}
                        
                error = None
                try:
                    action = HelpdeskAction(**action_data)
                except Exception as e:
                    action = HelpdeskAction(tool_name="error", tool_args={"error": str(e)})
                    error = str(e)
                    
                obs_res = env.step(action)
                obs = obs_res.observation
                reward = obs_res.reward or 0.0
                done = obs_res.done
                
                rewards.append(reward)
                steps_taken = step_num
                last_reward = reward
                
                log_step(step=step_num, action=message, reward=reward, done=done, error=error)
                history.append(f"Step {step_num}: {message} -> reward {reward:+.2f}")
                
            # Score formula required between 0.0 and 1.0
            # Grader computes real score, but we do a pseudo-score here for the log_end requirement
            total_reward = sum(rewards)
            score = 0.1 + (total_reward * 0.8) # mimicking what grader logic does loosely to preserve constraints
            score = min(max(score, 0.1), 0.9)  # clamp strictly within (0,1)
            success = score >= 0.5
            
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    run_agent()
