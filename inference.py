import os
import json
from openai import OpenAI
from helpdesk_env.client import HelpdeskEnv, HelpdeskAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.environ.get("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-key"
)

# Start the environment context manager without base_url so it runs a local server automatically or connects if provided
# But since this is inference, it might want to connect to a space. 
# We'll use local by default for testing.

def run_agent():
    # Run the server locally using .sync() without base_url handles this typically if configured right
    # However we'll assume the environment is running locally on 8000 for now.
    base_url = "http://127.0.0.1:8000"
    
    env_client = HelpdeskEnv(base_url=base_url)
    with env_client.sync() as env:
        obs_res = env.reset()
        obs = obs_res.observation
        state = env.state()
        
        print(f"[START] Episode {state.episode_id}")
        
        total_reward = 0.0
        for step_num in range(15):
            prompt = (
                f"Observation: {obs.command_result}\n"
                f"Tickets: {obs.open_tickets_summary}\n\n"
                "You are an IT helpdesk agent. Resolve the tickets. Output exactly ONE valid JSON object per turn and no other text. Example:\n"
                "{\"tool_name\": \"tool\", \"tool_args\": {\"arg\": \"val\"}}\n\n"
                "Tools available:\n"
                "- get_tickets (no args)\n"
                "- read_ticket (args: ticket_id)\n"
                "- run_diagnostic (args: system)\n"
                "- restart_service (args: service)\n"
                "- assign_ticket (args: ticket_id, department)\n"
                "- resolve_ticket (args: ticket_id)\n"
            )
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content or ""
                print(f"[DEBUG] Raw LLM reply: {repr(content)}")
                
                # Try to extract from Markdown first
                import re
                md_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if md_match:
                    json_candidate = md_match.group(1)
                else:
                    json_candidate = content

                action_data = None
                # Since Llama might output multiple json actions separated by newline
                for line in json_candidate.split('\n'):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            action_data = json.loads(line)
                            break
                        except Exception:
                            continue
                
                if not action_data:
                    # Fallback to grabbing everything between first { and last }
                    match = re.search(r'\{.*\}', json_candidate, re.DOTALL)
                    json_str = match.group(0) if match else json_candidate
                    action_data = json.loads(json_str)

                action = HelpdeskAction(**action_data)
            except Exception as e:
                action = HelpdeskAction(tool_name="error", tool_args={"error": str(e)})
                
            print(f"[STEP] step={step_num} obs={obs.command_result} action={action.model_dump_json()} reward={obs_res.reward}")
            
            obs_res = env.step(action)
            obs = obs_res.observation
            total_reward += obs_res.reward
            
            if obs_res.done:
                # Print final step
                print(f"[STEP] step={step_num+1} obs={obs.command_result} action=None reward={obs_res.reward}")
                break
                
        print(f"[END] Final Reward: {total_reward}")

if __name__ == "__main__":
    run_agent()
