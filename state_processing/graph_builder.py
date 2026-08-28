from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from state_processing.state import StateProcessor

class GraphBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        env_cfg = config.get("environment", {})
        sim_cfg = config.get("simulation", {})
        m_cfg = config.get("model", {})
        self.edge_rule = env_cfg.get("edge_rule", "distance_threshold")
        self.radius = env_cfg.get("edge_threshold_radius", 15.0)
        self.knn_k = env_cfg.get("knn_k", 4)
        self.max_nodes = sim_cfg.get("max_nodes", 30)
        self.node_dim = m_cfg.get("node_dim", 16)
        self.processor = StateProcessor(config)

    def build_graph(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        drones_dict = raw_state.get("drones", {})
        obstacles_list = raw_state.get("obstacles", [])
        targets_dict = raw_state.get("targets", {})
        node_features_list = []
        positions_list = []
        node_types_list = []
        drone_indices = []

        for idx, (d_id, d_data) in enumerate(drones_dict.items()):
            feat = self.processor.process_agent_state(d_data, idx)
            node_features_list.append(feat)
            positions_list.append(d_data["pos"])
            node_types_list.append(0)
            drone_indices.append(idx)

        for idx, obs_data in enumerate(obstacles_list):
            feat = self.processor.process_obstacle_state(obs_data, idx)
            node_features_list.append(feat)
            positions_list.append(obs_data["pos"])
            node_types_list.append(1)

        for idx, (t_id, t_data) in enumerate(targets_dict.items()):
            feat = self.processor.process_target_state(t_data, idx)
            node_features_list.append(feat)
            positions_list.append(t_data["pos"])
            node_types_list.append(2)

        actual_num_nodes = len(node_features_list)
        x = np.zeros((self.max_nodes, self.node_dim), dtype=np.float32)
        if actual_num_nodes > 0:
            x[:min(actual_num_nodes, self.max_nodes)] = np.array(node_features_list[:self.max_nodes], dtype=np.float32)
        mask = np.zeros(self.max_nodes, dtype=bool)
        mask[:min(actual_num_nodes, self.max_nodes)] = True
        node_types = np.zeros(self.max_nodes, dtype=np.int64)
        if actual_num_nodes > 0:
            node_types[:min(actual_num_nodes, self.max_nodes)] = np.array(node_types_list[:self.max_nodes], dtype=np.int64)
        positions = np.zeros((self.max_nodes, 3), dtype=np.float32)
        if actual_num_nodes > 0:
            positions[:min(actual_num_nodes, self.max_nodes)] = np.array(positions_list[:self.max_nodes], dtype=np.float32)

        diff = positions[:, None, :] - positions[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1).astype(np.float32)
        edges = []
        valid_count = min(actual_num_nodes, self.max_nodes)
        for i in range(valid_count):
            for j in range(valid_count):
                if i == j:
                    continue
                if self.edge_rule == "distance_threshold" and dist_matrix[i, j] <= self.radius:
                    edges.append((i, j))
        edge_index = np.empty((2, 0), dtype=np.int64) if len(edges) == 0 else np.array(edges, dtype=np.int64).T

        return {
            "x": torch.from_numpy(x).unsqueeze(0),
            "edge_index": torch.from_numpy(edge_index),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "node_types": torch.from_numpy(node_types).unsqueeze(0),
            "drone_indices": drone_indices,
            "dist_matrix": torch.from_numpy(dist_matrix).unsqueeze(0)
        }
