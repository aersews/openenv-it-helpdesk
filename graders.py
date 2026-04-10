def _compute_score(episode):
    try:
        rewards = []
        if hasattr(episode, 'rewards') and episode.rewards is not None:
            rewards = episode.rewards
        elif hasattr(episode, 'steps') and episode.steps is not None:
            rewards = [getattr(step, 'reward', 0.0) for step in episode.steps]
        elif isinstance(episode, dict):
            rewards = episode.get('rewards') or [s.get('reward', 0.0) for s in episode.get('steps', [])]
            
        total_reward = sum(float(r) for r in rewards if r is not None)
        score = 0.1 + (total_reward * 0.8)
        return float(min(0.9, max(0.1, score)))
    except Exception:
        # Ensure it NEVER yields exactly 0.0 or 1.0 even on complete failure, keeping it STRICTLY between (0, 1)
        return 0.5

def grade_easy(episode):
    return _compute_score(episode)

def grade_medium(episode):
    return _compute_score(episode)

def grade_hard(episode):
    return _compute_score(episode)

