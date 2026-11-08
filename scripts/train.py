"""Pretrain from expert demonstrations, then fine-tune with PPO or SAC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from autopark.environment import ParkingEnv
from autopark.imitation import (
    collect_expert_demonstrations,
    initialize_sb3_actor,
    save_demonstrations,
    train_behavior_cloning,
)


class CurriculumCallback(BaseCallback):
    def __init__(self, env: ParkingEnv, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.total_timesteps = max(1, int(total_timesteps))
        self.last_difficulty = -1.0

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        difficulty = 0.10 + 0.48 * progress
        if abs(difficulty - self.last_difficulty) >= 0.025:
            self.env.set_difficulty(difficulty)
            self.last_difficulty = difficulty
            if self.verbose:
                print(f"\nCurriculum difficulty -> {difficulty:.2f}")
        return True


class SuccessAwareEvalCallback(BaseCallback):
    """Select checkpoints by success rate first, not mean reward alone."""

    def __init__(
        self,
        eval_env: ParkingEnv,
        best_model_path: Path,
        *,
        eval_freq: int = 40_000,
        episodes: int = 24,
        seed: int = 90_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_path = best_model_path
        self.eval_freq = max(1, int(eval_freq))
        self.episodes = max(1, int(episodes))
        self.seed = int(seed)
        self.eval_calls = 0
        self.best_key = (-1.0, -1.0, -1.0, -1e9)
        self.metrics_path = ROOT / "reports" / "training" / "success_evaluations.jsonl"
        self.best_metrics_path = ROOT / "reports" / "training" / "best_model_metrics.json"
        if self.best_metrics_path.exists():
            try:
                previous = json.loads(self.best_metrics_path.read_text(encoding="utf-8"))
                self.best_key = (
                    float(previous["success_rate"]),
                    -float(previous["collision_rate"]),
                    -float(previous["outside_rate"]),
                    float(previous["mean_reward"]),
                )
            except (KeyError, ValueError, json.JSONDecodeError):
                pass

    def evaluate(self) -> dict[str, float]:
        successes = collisions = outside = 0
        rewards: list[float] = []
        distances: list[float] = []
        angle_errors: list[float] = []
        for episode in range(self.episodes):
            obs, _ = self.eval_env.reset(seed=self.seed + self.eval_calls * self.episodes + episode)
            terminated = truncated = False
            total_reward = 0.0
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += float(reward)
            successes += int(bool(info.get("success")))
            collisions += int(bool(info.get("collision")))
            outside += int(bool(info.get("outside_bounds")))
            rewards.append(total_reward)
            distances.append(float(info.get("distance", np.nan)))
            angle_errors.append(float(info.get("angle_error_deg", np.nan)))

        result = {
            "timesteps": int(self.num_timesteps),
            "episodes": int(self.episodes),
            "success_rate": successes / self.episodes,
            "collision_rate": collisions / self.episodes,
            "outside_rate": outside / self.episodes,
            "timeout_rate": max(0.0, 1.0 - (successes + collisions + outside) / self.episodes),
            "mean_reward": float(np.mean(rewards)),
            "mean_distance_m": float(np.mean(distances)),
            "mean_angle_error_deg": float(np.mean(angle_errors)),
        }
        return result

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        self.eval_calls += 1
        metrics = self.evaluate()
        self.logger.record("eval/success_rate", metrics["success_rate"])
        self.logger.record("eval/collision_rate", metrics["collision_rate"])
        self.logger.record("eval/mean_reward", metrics["mean_reward"])
        self.logger.record("eval/mean_distance_m", metrics["mean_distance_m"])

        key = (
            metrics["success_rate"],
            -metrics["collision_rate"],
            -metrics["outside_rate"],
            metrics["mean_reward"],
        )
        improved = key > self.best_key
        if improved:
            self.best_key = key
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_model_path))
            metrics["saved_as_best"] = 1.0
            self.best_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            metrics["saved_as_best"] = 0.0

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if self.verbose:
            suffix = " | NEW BEST" if improved else ""
            print(
                "\nEval @ {:,}: success={:.1%}, collision={:.1%}, timeout={:.1%}, "
                "dist={:.2f}m, angle={:.1f}deg{}".format(
                    metrics["timesteps"],
                    metrics["success_rate"],
                    metrics["collision_rate"],
                    metrics["timeout_rate"],
                    metrics["mean_distance_m"],
                    metrics["mean_angle_error_deg"],
                    suffix,
                )
            )
        return True


class PeriodicCheckpointCallback(BaseCallback):
    def __init__(self, directory: Path, algorithm: str, frequency: int, verbose: int = 0):
        super().__init__(verbose)
        self.directory = directory
        self.algorithm = algorithm
        self.frequency = max(1, int(frequency))

    def _on_step(self) -> bool:
        if self.n_calls % self.frequency == 0:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{self.algorithm}_step_{self.num_timesteps:,}".replace(",", "")
            self.model.save(str(path))
            if self.verbose:
                print(f"\nCheckpoint saved: {path}.zip")
        return True


def evaluate_model(
    model: Any,
    env: ParkingEnv,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    successes = collisions = outside = 0
    rewards: list[float] = []
    distances: list[float] = []
    angles: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        total_reward = 0.0
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
        successes += int(bool(info.get("success")))
        collisions += int(bool(info.get("collision")))
        outside += int(bool(info.get("outside_bounds")))
        rewards.append(total_reward)
        distances.append(float(info["distance"]))
        angles.append(float(info["angle_error_deg"]))
    return {
        "episodes": float(episodes),
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "outside_rate": outside / episodes,
        "timeout_rate": max(0.0, 1.0 - (successes + collisions + outside) / episodes),
        "mean_reward": float(np.mean(rewards)),
        "mean_distance_m": float(np.mean(distances)),
        "mean_angle_error_deg": float(np.mean(angles)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoPark v0.3: expert behavior cloning + PPO/SAC fine-tuning"
    )
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--demo-episodes", type=int, default=900)
    parser.add_argument("--bc-epochs", type=int, default=35)
    parser.add_argument("--eval-freq", type=int, default=40_000)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--checkpoint-freq", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--resume", action="store_true", help="Continue from models/final_model.zip")
    parser.add_argument("--skip-pretraining", action="store_true", help="Start RL without BC initialization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps < 1:
        raise SystemExit("--timesteps must be positive")
    if args.demo_episodes < 1 and not args.skip_pretraining and not args.resume:
        raise SystemExit("--demo-episodes must be positive")

    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports" / "training"
    best_path = models_dir / "best_model.zip"
    final_path = models_dir / "final_model.zip"
    for directory in (models_dir, reports_dir, ROOT / "reports" / "tensorboard", models_dir / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for stale_log in (
            reports_dir / "success_evaluations.jsonl",
            reports_dir / "best_model_metrics.json",
        ):
            stale_log.unlink(missing_ok=True)

    base_env = ParkingEnv(difficulty=0.10)
    train_env = Monitor(base_env)
    eval_env = ParkingEnv(difficulty=0.58)

    if args.algorithm == "ppo":
        ModelClass = PPO
        policy_kwargs = {
            "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
            "activation_fn": torch.nn.Tanh,
        }
    else:
        ModelClass = SAC
        policy_kwargs = {
            "net_arch": [256, 256, 128],
            "activation_fn": torch.nn.Tanh,
        }

    if args.resume:
        if not final_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing {final_path}")
        print(f"Resuming {args.algorithm.upper()} from {final_path}")
        model = ModelClass.load(str(final_path), env=train_env, device=args.device)
    else:
        if not args.skip_pretraining:
            print("\n[1/4] Collecting expert parking demonstrations...")
            demo_env = ParkingEnv(difficulty=0.25)
            demo_obs, demo_actions, demo_stats = collect_expert_demonstrations(
                demo_env,
                episodes=args.demo_episodes,
                seed=args.seed,
            )
            save_demonstrations(
                reports_dir / "expert_demonstrations.npz",
                demo_obs,
                demo_actions,
                demo_stats,
            )
            (reports_dir / "expert_demonstrations.json").write_text(
                json.dumps(demo_stats, indent=2), encoding="utf-8"
            )
            print(f"Expert success rate: {demo_stats['expert_success_rate']:.1%}")
            if demo_stats["expert_success_rate"] < 0.90:
                raise RuntimeError(
                    "Expert baseline success rate is below 90%; refusing to train on weak demonstrations."
                )

            print("\n[2/4] Behavior cloning pretraining...")
            bc_actor, bc_metrics = train_behavior_cloning(
                demo_obs,
                demo_actions,
                algorithm=args.algorithm,
                epochs=args.bc_epochs,
                device="cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu",
                seed=args.seed,
            )
            torch.save(bc_actor.state_dict(), models_dir / "bc_actor.pt")
            (reports_dir / "behavior_cloning.json").write_text(
                json.dumps({**demo_stats, **bc_metrics}, indent=2), encoding="utf-8"
            )
        else:
            bc_actor = None

        print("\n[3/4] Initializing RL model...")
        if args.algorithm == "ppo":
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=1.0e-4,
                n_steps=2048,
                batch_size=512,
                n_epochs=6,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.15,
                ent_coef=0.003,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(ROOT / "reports" / "tensorboard"),
                verbose=1,
                seed=args.seed,
                device=args.device,
            )
        else:
            model = SAC(
                "MlpPolicy",
                train_env,
                learning_rate=3.0e-4,
                buffer_size=500_000,
                learning_starts=1_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                ent_coef="auto_0.1",
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(ROOT / "reports" / "tensorboard"),
                verbose=1,
                seed=args.seed,
                device=args.device,
            )
        if bc_actor is not None:
            initialize_sb3_actor(model, bc_actor, args.algorithm)
            print("Behavior-cloned actor weights copied into RL policy.")

        print("\n[4/4] Measuring the initialized policy on unseen parking maps...")
        baseline = evaluate_model(model, eval_env, episodes=100, seed=120_000)
        print(json.dumps(baseline, indent=2))
        model.save(str(best_path))
        baseline_with_time = {"timesteps": 0, **baseline, "saved_as_best": 1.0}
        (reports_dir / "best_model_metrics.json").write_text(
            json.dumps(baseline_with_time, indent=2), encoding="utf-8"
        )
        (reports_dir / "bc_baseline_evaluation.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )

    eval_callback = SuccessAwareEvalCallback(
        eval_env,
        best_path,
        eval_freq=args.eval_freq,
        episodes=args.eval_episodes,
        verbose=1,
    )
    callbacks = CallbackList(
        [
            eval_callback,
            CurriculumCallback(base_env, args.timesteps, verbose=1),
            PeriodicCheckpointCallback(
                models_dir / "checkpoints", args.algorithm, args.checkpoint_freq, verbose=1
            ),
        ]
    )

    metadata = {
        "version": "0.3",
        "algorithm": args.algorithm,
        "observation_size": ParkingEnv.OBSERVATION_SIZE,
        "action_size": 2,
        "behavior_cloning": not args.skip_pretraining,
        "expert_demo_episodes": args.demo_episodes if not args.skip_pretraining else 0,
        "timesteps_requested": args.timesteps,
        "seed": args.seed,
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print(f"AUTOPARK RL v0.3 | {args.algorithm.upper()} + BEHAVIOR CLONING")
    print(f"RL timesteps       : {args.timesteps:,}")
    print(f"Expert episodes    : {metadata['expert_demo_episodes']:,}")
    print(f"Evaluation          : every {args.eval_freq:,} steps / {args.eval_episodes} episodes")
    print("Parking bays        : 12 with obstacle-aware bay selection")
    print("Selection criterion : success rate, then collision/outside rates, then reward")
    print(f"Device              : {args.device}")
    print("=" * 76)
    print("Training begins. Best model is selected by completed parking rate, not reward alone.\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not args.resume,
            tb_log_name=f"autopark_v03_{args.algorithm}",
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted; saving current policy...")
    finally:
        model.save(str(final_path))
        train_env.close()
        eval_env.close()
        print(f"\nFinal model: {final_path}")
        print(f"Best model : {best_path}")
        print(f"Metadata   : {models_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
