"""Watch a trained policy or the built-in expert in the Pygame simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pygame
from stable_baselines3 import PPO, SAC

from autopark.environment import ParkingEnv
from autopark.simulation.renderer import ParkingRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoPark RL v0.3 visualizer")
    parser.add_argument("--model", default="models/best_model.zip")
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--expert", action="store_true", help="Run the tested expert controller")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fps", type=int, default=35)
    parser.add_argument("--difficulty", type=float, default=0.58)
    args = parser.parse_args()

    model = None
    if not args.expert:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train first or use --expert."
            )
        model_cls = PPO if args.algorithm == "ppo" else SAC
        model = model_cls.load(str(model_path), device="auto")

    env = ParkingEnv(difficulty=args.difficulty)
    pygame.init()
    renderer = ParkingRenderer()
    clock = pygame.time.Clock()

    episode = 1
    observation, _ = env.reset()
    trajectory: list[tuple[float, float]] = [(env.car.x, env.car.y)]
    total_reward = 0.0
    status = "EXPERT READY" if args.expert else "AGENT READY"
    done = False
    paused = False
    terminal_frames = 0
    deterministic = not args.stochastic
    running = True
    mode_name = "EXPERT CONTROLLER" if args.expert else f"{args.algorithm.upper()} POLICY"

    print(f"Mode: {mode_name}")
    print("Controls: SPACE pause | N next episode | R reset | ESC exit")
    print(
        f"Selected bay: {env.target_slot_id}; free="
        f"{sum(not r['occupied'] for r in env.slot_records)}, "
        f"feasible={sum(r['feasible'] for r in env.slot_records)}"
    )

    while running:
        reset_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_n, pygame.K_r):
                    reset_requested = True

        if reset_requested:
            episode += 1
            observation, _ = env.reset()
            trajectory = [(env.car.x, env.car.y)]
            total_reward = 0.0
            status = "NEW SCENARIO"
            done = False
            terminal_frames = 0

        if running and not paused and not done:
            action = env.expert_action() if args.expert else model.predict(
                observation, deterministic=deterministic
            )[0]
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            trajectory.append((env.car.x, env.car.y))
            if len(trajectory) > 900:
                trajectory = trajectory[-900:]

            if terminated or truncated:
                done = True
                terminal_frames = 0
                if info.get("success"):
                    status = f"SUCCESS  |  {info['target_slot']}  |  {info['steps']} steps"
                elif info.get("collision"):
                    status = f"COLLISION  |  {info['target_slot']}"
                elif info.get("outside_bounds"):
                    status = "OUTSIDE BOUNDS"
                else:
                    status = f"TIME LIMIT  |  {info['target_slot']}"
                print(
                    f"Episode {episode}: {status}; reward={total_reward:.1f}; "
                    f"distance={info['distance']:.2f}m; angle={info['angle_error_deg']:.1f}deg"
                )
        elif done and not paused:
            terminal_frames += 1
            if terminal_frames >= args.fps * 2:
                episode += 1
                observation, _ = env.reset()
                trajectory = [(env.car.x, env.car.y)]
                total_reward = 0.0
                status = "NEW SCENARIO"
                done = False
                terminal_frames = 0

        renderer.draw(
            env,
            episode=episode,
            reward=total_reward,
            status=status,
            trajectory=trajectory,
            paused=paused,
            deterministic=deterministic,
            mode=mode_name,
        )
        clock.tick(max(10, args.fps))

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
