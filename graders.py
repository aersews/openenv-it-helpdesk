def grade_easy(episode):
    total_reward = sum(r for r in episode.rewards if r is not None) if hasattr(episode, 'rewards') else 0.0
    score = 0.1 + (total_reward * 0.8)
    return min(0.9, max(0.1, score))

def grade_medium(episode):
    total_reward = sum(r for r in episode.rewards if r is not None) if hasattr(episode, 'rewards') else 0.0
    score = 0.1 + (total_reward * 0.8)
    return min(0.9, max(0.1, score))

def grade_hard(episode):
    total_reward = sum(r for r in episode.rewards if r is not None) if hasattr(episode, 'rewards') else 0.0
    score = 0.1 + (total_reward * 0.8)
    return min(0.9, max(0.1, score))

