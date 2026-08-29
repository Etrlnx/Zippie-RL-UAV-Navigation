import tomllib
import pytest
import torch
from rl_engine.agent import MAPPOAgent
from graph_transformer.model import GraphTransformer


@pytest.fixture
def config():
    with open("configs/default.toml", "rb") as f:
        return tomllib.load(f)


def test_graph_transformer_forward(config):
    m_cfg = config["model"]
    gt = GraphTransformer(
        node_dim=m_cfg["node_dim"],
        hidden_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        num_heads=m_cfg["num_heads"]
    )
    x = torch.randn(2, 30, m_cfg["node_dim"])
    mask = torch.ones(2, 30, dtype=torch.bool)
    dist_m = torch.rand(2, 30, 30) * 10.0
    embeds, attns = gt(x, mask, dist_matrix=dist_m)
    assert embeds.shape == (2, 30, m_cfg["hidden_dim"])
    assert len(attns) == m_cfg["num_layers"]


def test_mappo_agent_forward(config):
    agent = MAPPOAgent(config, action_dim=7)
    x = torch.randn(2, 30, config["model"]["node_dim"])
    mask = torch.ones(2, 30, dtype=torch.bool)
    dist_m = torch.rand(2, 30, 30) * 10.0
    logits, val, attns = agent(x, mask, [0, 1, 2], dist_matrix=dist_m)
    assert logits.shape == (2, 3, 7)
    assert val.shape == (2, 1)
