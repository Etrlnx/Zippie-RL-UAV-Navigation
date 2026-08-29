from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from graph_transformer.layers import GraphTransformerLayer

class GraphTransformer(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self, node_features: torch.Tensor, mask: torch.Tensor, dist_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self.input_proj(node_features)
        key_padding_mask = ~mask if mask is not None else None
        all_attn_weights = []
        for layer in self.layers:
            h, attn = layer(h, dist_matrix=dist_matrix, key_padding_mask=key_padding_mask)
            all_attn_weights.append(attn)
        return h, all_attn_weights
