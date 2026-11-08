"""Evaluate an AutoPark PPO/SAC policy or the built-in expert controller."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC

from autopark.environment import ParkingEnv


def resolve_model(path_text: str | None) -> Path:
    if path_text:
        path = Path(path_text)
        return path if path.is_absolute() else ROOT / path
    candidates = [ROOT / "models" / "best_model.zip", ROOT / "models" / "final_model.zip"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No model found. Train first or pass --model PATH")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AutoPark v0.3")
    parser.add_argument("--model", default=None)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--expert", action="store_true", help="Evaluate the rule-based expert baseline")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--difficulty", type=float, default=0.58)
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")

    model = None
    model_label = "expert_controller"
    if not args.expert:
        model_path = resolve_model(args.model)
        model_cls = PPO if args.algorithm == "ppo" else SAC
        model = model_cls.load(str(model_path), device="auto")
        model_label = str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path)

    env = ParkingEnv(difficulty=args.difficulty)
    rows: list[dict[str, object]] = []
    selected_slots: Counter[str] = Counter()

    for episode in range(args.episodes):
        obs, reset_info = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        terminated = truncated = False
        info = reset_info
        while not (terminated or truncated):
            action = env.expert_action() if args.expert else model.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

        selected_slots[str(info["target_slot"])] += 1
        rows.append({
            "episode": episode + 1,
            "target_slot": info["target_slot"],
            "free_slots": info["free_slots"],
            "feasible_slots": info["feasible_slots"],
            "occupied_slots": info["occupied_slots"],
            "success": bool(info["success"]),
            "collision": bool(info["collision"]),
            "outside_bounds": bool(info["outside_bounds"]),
            "steps": int(info["steps"]),
            "distance_m": float(info["distance"]),
            "angle_error_deg": float(info["angle_error_deg"]),
            "bearing_error_deg": float(info["bearing_error_deg"]),
            "total_reward": total_reward,
        })
        if (episode + 1) % 20 == 0 or episode + 1 == args.episodes:
            print(f"Evaluated {episode + 1}/{args.episodes} episodes")

    env.close()
    out_dir = ROOT / "reports" / "evaluation"
    fig_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / ("expert_episodes.csv" if args.expert else "episodes.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    success_rate = float(np.mean([bool(row["success"]) for row in rows]))
    collision_rate = float(np.mean([bool(row["collision"]) for row in rows]))
    outside_rate = float(np.mean([bool(row["outside_bounds"]) for row in rows]))
    summary = {
        "model": model_label,
        "mode": "expert" if args.expert else args.algorithm,
        "episodes": args.episodes,
        "difficulty": args.difficulty,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "outside_bounds_rate": outside_rate,
        "timeout_rate": max(0.0, 1.0 - success_rate - collision_rate - outside_rate),
        "mean_steps": float(np.mean([int(row["steps"]) for row in rows])),
        "mean_final_distance_m": float(np.mean([float(row["distance_m"]) for row in rows])),
        "mean_final_angle_error_deg": float(np.mean([float(row["angle_error_deg"]) for row in rows])),
        "mean_reward": float(np.mean([float(row["total_reward"]) for row in rows])),
        "selected_bay_counts": dict(selected_slots),
    }
    summary_path = out_dir / ("expert_summary.json" if args.expert else "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = ["Success", "Collision", "Outside", "Timeout"]
    values = [
        success_rate * 100,
        collision_rate * 100,
        outside_rate * 100,
        max(0.0, 100.0 * (1.0 - success_rate - collision_rate - outside_rate)),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Episodes (%)")
    ax.set_title(f"AutoPark v0.3 {'expert' if args.expert else args.algorithm.upper()} — {args.episodes} maps")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    chart_path = fig_dir / ("expert_evaluation.png" if args.expert else "evaluation_results.png")
    fig.savefig(chart_path, dpi=170)
    plt.close(fig)

    print("\nEvaluation summary")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Chart:   {chart_path}")


if __name__ == "__main__":
    main()
