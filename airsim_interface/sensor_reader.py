from typing import Any
import numpy as np

class SensorReader:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        env_cfg = config.get("environment", {})
        self.noise_std = env_cfg.get("sensor_noise_std", 0.02)
        self.sensor_range = env_cfg.get("sensor_range", 20.0)

    def read_sensors(
        self, drone_id: str, true_pos: np.ndarray, true_vel: np.ndarray,
        true_orientation: np.ndarray, battery: float, obstacles: list[dict[str, Any]]
    ) -> dict[str, np.ndarray]:
        pos_noise = np.random.normal(0, self.noise_std, size=3)
        vel_noise = np.random.normal(0, self.noise_std, size=3)
        gps = (true_pos + pos_noise).astype(np.float32)
        imu = np.concatenate([true_vel + vel_noise, true_orientation]).astype(np.float32)
        bat = np.array([battery], dtype=np.float32)
        lidar_points = []
        for obs in obstacles:
            obs_pos = obs["pos"]
            dist = float(np.linalg.norm(obs_pos - true_pos))
            if dist <= self.sensor_range:
                lidar_points.append(obs_pos - true_pos)
        lidar = np.zeros((1, 3), dtype=np.float32) if len(lidar_points) == 0 else np.array(lidar_points, dtype=np.float32)
        return {
            "gps": gps,
            "imu": imu,
            "battery": bat,
            "lidar": lidar,
            "rgb": np.zeros((64, 64, 3), dtype=np.uint8)
        }
