from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from graph_transformer.model import GraphTransformer

class MAPPOAgent(nn.Module):
    def __init__(self, config: Dict[str, Any], action_dim: int = 5):
        super().__init__()
        self.config = config
        m_cfg = config.get("model", {})
        self.encoder = GraphTransformer(
            node_dim=m_cfg.get("node_dim", 16),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 3),
            num_heads=m_cfg.get("num_heads", 4),
            dropout=m_cfg.get("dropout", 0.1)
        )
        hidden_dim = m_cfg.get("hidden_dim", 64)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, node_features: torch.Tensor, mask: torch.Tensor, drone_indices: List[int], dist_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        node_embeds, attn_weights = self.encoder(node_features, mask, dist_matrix=dist_matrix)
        drone_embeds = node_embeds[:, drone_indices, :]
        logits = self.actor(drone_embeds)
        pooled = drone_embeds.mean(dim=1)
        value = self.critic(pooled)
        return logits, value, attn_weights
