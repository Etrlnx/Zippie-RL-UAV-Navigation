import tomllib

import pytest
import torch

from explainability.attention_viz import AttentionVisualizer
from explainability.neighbor_analysis import NeighborAnalyzer


@pytest.fixture
def config():
    with open("configs/default.toml", "rb") as f:
        return tomllib.load(f)


def test_attention_rollout_and_neighbor_ranking(config):
    viz = AttentionVisualizer(config)
    analyzer = NeighborAnalyzer(config)

    dummy_attns = [torch.softmax(torch.randn(1, 4, 10, 10), dim=-1) for _ in range(3)]
    rollout = viz.extract_attention_rollout(dummy_attns)
    assert rollout.shape == (10, 10)

    meta = [{"label": f"node_{i}", "type": "drone" if i < 3 else "obstacle"} for i in range(10)]
    ranking = analyzer.rank_influential_neighbors(0, rollout, meta, top_k=3)
    assert len(ranking) == 3
    assert ranking[0]["attention_weight"] >= ranking[1]["attention_weight"]
