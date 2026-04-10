import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import HelpdeskAction, HelpdeskObservation
except ImportError:
    from models import HelpdeskAction, HelpdeskObservation

EASY_TASK = "easy"
MEDIUM_TASK = "medium"
HARD_TASK = "hard"

class HelpdeskEnvironment(Environment):
    """
    An IT Helpdesk and Diagnostics RL Environment.
    The agent gets an issue (easy, medium, hard) and uses tools to diagnose and resolve it.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.reset()
        
    def reset(self, scenario: str = None) -> HelpdeskObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_difficulty = scenario or random.choice([EASY_TASK, MEDIUM_TASK, HARD_TASK])
        self.tickets = {}
        self.system_state = {}
        self.total_reward = 0.0
        
        if self.task_difficulty == EASY_TASK:
            self.tickets = {
                "T-1001": {"title": "Laptop very slow", "body": "My laptop is taking 10 minutes to boot.", "status": "open", "department": "unassigned"}
            }
        elif self.task_difficulty == MEDIUM_TASK:
            self.tickets = {
                "T-2001": {"title": "Account locked", "body": "I cannot login, username: johndoe.", "status": "open", "department": "unassigned"}
            }
            self.system_state["johndoe_locked"] = True
        elif self.task_difficulty == HARD_TASK:
            self.tickets = {
                "T-3001": {"title": "Website is down!", "body": "I'm getting a 500 error on the company portal.", "status": "open", "department": "unassigned"}
            }
            self.system_state["portal_status"] = "500 Internal Server Error"
            self.system_state["db_status"] = "down"
            
        return self._make_obs("Environment reset. You are an IT helpdesk agent. Use 'get_tickets' to begin.")

    def _make_obs(self, result: str, reward: float = 0.0, done: bool = False) -> HelpdeskObservation:
        summary = ", ".join([f"{k} ({v['status']})" for k,v in self.tickets.items() if v['status'] == 'open'])
        if not summary:
            summary = "No open tickets."
        
        self.total_reward += reward
        final_reward = reward
        
        return HelpdeskObservation(
            command_result=result,
            open_tickets_summary=summary,
            done=done,
            reward=final_reward
        )

    def step(self, action: HelpdeskAction) -> HelpdeskObservation:
        self._state.step_count += 1
        
        if self._state.step_count > 15:
            return self._make_obs("Max steps reached.", reward=0.0, done=True)
            
        tool = action.tool_name
        args = action.tool_args
        
        if tool == "get_tickets":
            res = "Tickets:\\n"
            for k, v in self.tickets.items():
                res += f"- {k}: {v['title']} (Status: {v['status']}, Dept: {v['department']})\\n"
            return self._make_obs(res)
            
        elif tool == "read_ticket":
            tid = args.get("ticket_id")
            if tid in self.tickets:
                t = self.tickets[tid]
                return self._make_obs(f"Ticket {tid}:\\nTitle: {t['title']}\\nBody: {t['body']}")
            return self._make_obs(f"Ticket {tid} not found.")
            
        elif tool == "assign_ticket":
            tid = args.get("ticket_id")
            dept = args.get("department")
            if tid in self.tickets:
                self.tickets[tid]["department"] = dept
                reward = 0.0
                if self.task_difficulty == EASY_TASK and dept == "hardware_support" and tid == "T-1001":
                    reward = 0.5 # Partial reward for correct assignment
                return self._make_obs(f"Ticket {tid} assigned to {dept}.", reward=reward)
            return self._make_obs(f"Ticket {tid} not found.")
            
        elif tool == "run_diagnostic":
            system = args.get("system")
            if self.task_difficulty == HARD_TASK:
                if system == "portal":
                    return self._make_obs(f"Portal status: {self.system_state.get('portal_status')}")
                elif system == "database":
                    return self._make_obs(f"Database status: {self.system_state.get('db_status')}")
                elif system == "logs":
                    return self._make_obs(f"Logs: Connection refused to database backend.")
            elif self.task_difficulty == MEDIUM_TASK:
                if system == "johndoe":
                    status = "Locked" if self.system_state.get("johndoe_locked") else "Active"
                    return self._make_obs(f"User johndoe status: {status}")
            return self._make_obs(f"Diagnostic on {system} returned typical results.")
            
        elif tool == "restart_service":
            svc = args.get("service")
            if svc == "database" and self.task_difficulty == HARD_TASK:
                self.system_state["db_status"] = "running"
                self.system_state["portal_status"] = "200 OK"
                return self._make_obs("Database restarted successfully.", reward=0.5)
            elif svc == "johndoe_account" and self.task_difficulty == MEDIUM_TASK:
                self.system_state["johndoe_locked"] = False
                return self._make_obs("User account unlocked.", reward=0.5)
            return self._make_obs(f"Service {svc} restarted but had no effect.")
            
        elif tool == "resolve_ticket":
            tid = args.get("ticket_id")
            if tid in self.tickets:
                t = self.tickets[tid]
                reward = 0.0
                done = False
                
                if self.task_difficulty == EASY_TASK:
                    if t["department"] == "hardware_support":
                        reward = 0.5
                        done = True
                elif self.task_difficulty == MEDIUM_TASK:
                    if not self.system_state.get("johndoe_locked"):
                        reward = 0.5
                        done = True
                elif self.task_difficulty == HARD_TASK:
                    if self.system_state.get("db_status") == "running":
                        reward = 0.5
                        done = True
                        
                t["status"] = "resolved"
                
                if done:
                    return self._make_obs(f"Ticket {tid} resolved correctly. Episode finished.", reward=reward, done=True)
                else:
                    return self._make_obs(f"Ticket {tid} resolved, but the underlying issue was not fixed or assigned correctly.", reward=0.0, done=True)
            return self._make_obs(f"Ticket {tid} not found.")

        return self._make_obs(f"Unknown tool: {tool}")

    @property
    def state(self) -> State:
        return self._state
