import tomllib

import pytest

from airsim_interface.env import AirSimEnv


@pytest.fixture
def config():
    with open("configs/default.toml", "rb") as f:
        return tomllib.load(f)


def test_env_reset(config):
    env = AirSimEnv(config)
    obs = env.reset()
    assert len(obs["drones"]) == config["simulation"]["num_agents"]
    assert len(obs["obstacles"]) == 10
    assert len(obs["targets"]) == config["simulation"]["num_agents"]


def test_env_step(config):
    env = AirSimEnv(config)
    env.reset()
    actions = {"drone_0": 1, "drone_1": 2, "drone_2": 3}
    obs, rewards, done, info = env.step(actions)
    assert len(rewards) == config["simulation"]["num_agents"]
    assert isinstance(done, bool)
    assert "collisions" in info
