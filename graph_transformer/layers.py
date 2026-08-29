from typing import Optional, Tuple
import torch
import torch.nn as nn

class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.gamma = nn.Parameter(torch.full((num_heads, 1, 1), 0.1))

    def forward(
        self, x: torch.Tensor, dist_matrix: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = x.shape
        if dist_matrix is not None:
            log_dist = torch.log1p(dist_matrix)
            dist_bias = -torch.abs(self.gamma).unsqueeze(0) * log_dist.unsqueeze(1)
            attn_mask = dist_bias.view(batch_size * self.num_heads, num_nodes, num_nodes)
        else:
            attn_mask = torch.zeros((batch_size * self.num_heads, num_nodes, num_nodes), device=x.device)

        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).repeat(1, self.num_heads, num_nodes, 1)
            pad_mask = pad_mask.view(batch_size * self.num_heads, num_nodes, num_nodes)
            attn_mask = attn_mask.masked_fill(pad_mask, float("-inf"))

        attn_out, attn_weights = self.mha(x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, attn_weights
