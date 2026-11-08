from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autopark.environment import ParkingEnv


def test_observation_and_action_spaces() -> None:
    env = ParkingEnv()
    obs, info = env.reset(seed=11)
    assert obs.shape == (ParkingEnv.OBSERVATION_SIZE,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (2,)
    assert info["feasible_slots"] >= 2
    env.close()


def test_target_is_empty_and_feasible() -> None:
    env = ParkingEnv(difficulty=0.58)
    for seed in range(100):
        _, info = env.reset(seed=seed)
        selected = next(r for r in env.slot_records if r["id"] == env.target_slot_id)
        assert selected["feasible"] is True
        assert selected["occupied"] is False
        assert selected["selected"] is True
        assert info["feasible_slots"] >= 2
    env.close()


def test_expert_parks_in_randomized_maps() -> None:
    env = ParkingEnv(difficulty=0.58)
    successes = 0
    for episode in range(100):
        env.set_difficulty(0.10 + 0.48 * ((episode % 17) / 16.0))
        obs, _ = env.reset(seed=30_000 + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, info = env.step(env.expert_action())
            assert env.observation_space.contains(obs)
        successes += int(info["success"])
    env.close()
    assert successes >= 95, f"Expert only completed {successes}/100 scenarios"


def test_collision_terminates_episode() -> None:
    env = ParkingEnv(difficulty=0.0)
    env.reset(seed=9)
    # Place an obstacle over the vehicle to trigger the physical collision check.
    env.obstacles = [env.car.__class__(env.car.x, env.car.y, env.car.heading, 0.0)]
    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["collision"] is True
    assert reward < -100.0
    env.close()
