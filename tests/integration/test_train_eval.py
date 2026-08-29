"""Integration test for full train -> eval pipeline."""
import tempfile
import tomllib
import torch
from pathlib import Path

from airsim_interface.env import AirSimEnv
from rl_engine.agent import MAPPOAgent
from rl_engine.trainer import MAPPOTrainer
from evaluation.baselines import BaselineRunner, MLPBaselineAgent


def test_train_eval_pipeline():
    """Test that training runs and evaluation works end-to-end."""
    with open("configs/default.toml", "rb") as f:
        config = tomllib.load(f)

    # Reduce steps for fast test
    config["rl"]["total_timesteps"] = 100
    config["rl"]["n_steps"] = 32
    config["rl"]["batch_size"] = 16
    config["rl"]["n_epochs"] = 1
    config["simulation"]["max_steps"] = 50
    config["experiment"]["checkpoint_dir"] = "tests/integration/checkpoints"
    config["experiment"]["log_dir"] = "tests/integration/logs"

    env = AirSimEnv(config, seed=42)
    agent = MAPPOAgent(config)
    trainer = MAPPOTrainer(config, env, agent)

    # Run a few training iterations
    trainer.train(max_iterations=2)

    # Check checkpoint was saved
    checkpoint_path = Path(config["experiment"]["checkpoint_dir"]) / "model_iter_1.pt"
    assert checkpoint_path.exists(), "Checkpoint not saved"

    # Load checkpoint and evaluate
    agent.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    runner = BaselineRunner(config)
    results = runner.run_evaluation(agent, env, num_episodes=2)

    # Verify metrics structure
    assert "success_rate" in results
    assert "path_length" in results
    assert "collision_rate" in results
    assert "energy_consumption" in results
    assert "completion_time" in results
    assert "inference_latency_ms" in results
    assert "unseen_generalization_rate" in results

    # Verify values are floats
    for v in results.values():
        assert isinstance(v, float)


def test_baseline_evaluation():
    """Test that MLP baseline evaluation works."""
    with open("configs/default.toml", "rb") as f:
        config = tomllib.load(f)

    config["simulation"]["max_steps"] = 50

    env = AirSimEnv(config, seed=42)
    runner = BaselineRunner(config)

    mlp_agent = MLPBaselineAgent(num_agents=env.num_agents)
    results = runner.run_evaluation(mlp_agent, env, num_episodes=2)

    assert "success_rate" in results
    assert isinstance(results["success_rate"], float)