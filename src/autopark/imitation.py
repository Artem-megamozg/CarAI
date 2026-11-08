"""Expert demonstrations and behavior-cloning initialization for AutoPark RL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autopark.environment import ParkingEnv


HIDDEN_DIMS = (256, 256, 128)


class BehaviorCloningActor(nn.Module):
    def __init__(self, observation_size: int, action_size: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = observation_size
        for width in HIDDEN_DIMS:
            layers.extend((nn.Linear(in_features, width), nn.Tanh()))
            in_features = width
        layers.append(nn.Linear(in_features, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def collect_expert_demonstrations(
    env: ParkingEnv,
    *,
    episodes: int = 900,
    seed: int = 42,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect only complete expert trajectories across randomized lots."""
    if episodes < 1:
        raise ValueError("episodes must be positive")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    successful = 0
    failures = 0
    total_steps = 0
    step_limit = max_steps or env.max_steps

    for episode in range(episodes):
        # Sample difficulties from the curriculum range, not only easy maps.
        difficulty = float(np.random.default_rng(seed + episode).uniform(0.10, 0.58))
        env.set_difficulty(difficulty)
        obs, _ = env.reset(seed=seed + episode)
        episode_obs: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []
        terminated = truncated = False
        info: dict[str, Any] = {}

        while not (terminated or truncated) and len(episode_actions) < step_limit:
            action = env.expert_action()
            episode_obs.append(np.asarray(obs, dtype=np.float32).copy())
            episode_actions.append(np.asarray(action, dtype=np.float32).copy())
            obs, _, terminated, truncated, info = env.step(action)

        total_steps += len(episode_actions)
        if bool(info.get("success", False)):
            successful += 1
            observations.extend(episode_obs)
            actions.extend(episode_actions)
        else:
            failures += 1

        if (episode + 1) % 100 == 0 or episode + 1 == episodes:
            rate = successful / (episode + 1)
            print(
                f"Expert demonstrations: {episode + 1}/{episodes} | "
                f"successful={successful} | success_rate={rate:.1%}"
            )

    if not observations:
        raise RuntimeError("Expert generated no successful trajectories")

    obs_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    stats = {
        "requested_episodes": int(episodes),
        "successful_episodes": int(successful),
        "failed_episodes": int(failures),
        "expert_success_rate": float(successful / episodes),
        "accepted_transitions": int(len(obs_array)),
        "mean_steps_all_episodes": float(total_steps / episodes),
    }
    return obs_array, action_array, stats


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


def initialize_sb3_actor(
    sb3_model: Any,
    behavior_actor: BehaviorCloningActor,
    algorithm: str,
) -> None:
    """Copy BC weights into the actor layers of PPO or SAC."""
    if algorithm == "ppo":
        destinations = [
            *_linear_layers(sb3_model.policy.mlp_extractor.policy_net),
            *_linear_layers(sb3_model.policy.action_net),
        ]
        if hasattr(sb3_model.policy, "log_std"):
            with torch.no_grad():
                sb3_model.policy.log_std.fill_(-1.35)
    elif algorithm == "sac":
        actor = sb3_model.policy.actor
        destinations = [
            *_linear_layers(actor.latent_pi),
            *_linear_layers(actor.mu),
        ]
        if hasattr(actor, "log_std"):
            with torch.no_grad():
                actor.log_std.fill_(-2.0)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    sources = _linear_layers(behavior_actor.net)
    if len(destinations) != len(sources):
        raise RuntimeError(
            f"Actor layer mismatch: source={len(sources)}, destination={len(destinations)}"
        )
    with torch.no_grad():
        for source, destination in zip(sources, destinations):
            if source.weight.shape != destination.weight.shape:
                raise RuntimeError(
                    f"Actor shape mismatch: {tuple(source.weight.shape)} != "
                    f"{tuple(destination.weight.shape)}"
                )
            destination.weight.copy_(source.weight)
            destination.bias.copy_(source.bias)


def train_behavior_cloning(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    algorithm: str = "ppo",
    epochs: int = 35,
    batch_size: int = 512,
    learning_rate: float = 3.0e-4,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[BehaviorCloningActor, dict[str, float]]:
    if algorithm not in {"ppo", "sac"}:
        raise ValueError("algorithm must be 'ppo' or 'sac'")
    if len(observations) != len(actions) or len(observations) < 100:
        raise ValueError("The demonstration dataset is empty or too small")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(observations))
    split = max(1, int(len(indices) * 0.9))
    train_idx, validation_idx = indices[:split], indices[split:]

    x = torch.as_tensor(observations, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.float32)
    if algorithm == "sac":
        # SAC uses a tanh-squashed Gaussian policy; regress its pre-tanh mean.
        y = torch.atanh(torch.clamp(y, -0.97, 0.97))

    train_loader = DataLoader(
        TensorDataset(x[train_idx], y[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    actor = BehaviorCloningActor(observations.shape[1]).to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    best_validation = float("inf")
    best_state = None
    for epoch in range(epochs):
        actor.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = actor(batch_x)
            loss = loss_fn(prediction, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        actor.eval()
        with torch.no_grad():
            val_x = x[validation_idx].to(device)
            val_y = y[validation_idx].to(device)
            validation_loss = float(loss_fn(actor(val_x), val_y).cpu())
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in actor.state_dict().items()
            }
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                f"BC epoch {epoch + 1:02d}/{epochs} | "
                f"train_loss={np.mean(train_losses):.5f} | "
                f"val_loss={validation_loss:.5f}"
            )

    if best_state is not None:
        actor.load_state_dict(best_state)
    actor.eval()
    return actor, {"best_validation_loss": float(best_validation)}


def save_demonstrations(
    path: str | Path,
    observations: np.ndarray,
    actions: np.ndarray,
    stats: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        observations=observations,
        actions=actions,
    )
