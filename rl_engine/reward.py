from typing import Any

class RewardEngine:
    def __init__(self, config: dict[str, Any]):
        self.config = config.get("reward", {})
        self.progress_weight = self.config.get("progress_weight", 1.0)
        self.collision_penalty = self.config.get("collision_penalty", -5.0)
        self.energy_penalty = self.config.get("energy_penalty", -0.01)
        self.completion_bonus = self.config.get("completion_bonus", 10.0)

    def calculate_rewards(
        self, prev_distances: dict[str, float], curr_distances: dict[str, float],
        collisions: dict[str, bool], thrusts: dict[str, float], all_reached: bool
    ) -> dict[str, float]:
        rewards = {}
        for drone_id in curr_distances:
            delta_dist = prev_distances.get(drone_id, 0.0) - curr_distances.get(drone_id, 0.0)
            r = self.progress_weight * delta_dist
            if collisions.get(drone_id, False):
                r += self.collision_penalty
            r += self.energy_penalty * thrusts.get(drone_id, 0.0)
            if all_reached:
                r += self.completion_bonus
            rewards[drone_id] = float(r)
        return rewards
