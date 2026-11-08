# AutoPark RL v0.3

**Multi-bay autonomous parking with expert demonstrations, behavior cloning, and reinforcement learning.**

AutoPark RL is a 2D vehicle-parking research project. A parking manager generates a lot with 12 bays, checks whether empty spaces have safe geometry and a clear entry corridor, then selects a feasible destination. The learning agent controls throttle/brake and steering to align with the chosen bay, enter it, and stop.

Version 0.3 addresses the failure mode observed in v0.2: the PPO policy learned to avoid collisions but did not discover the 90-degree turn and controlled stop required for parking. The new pipeline first creates successful expert trajectories, pretrains the policy with behavior cloning, then fine-tunes it using PPO or SAC. Evaluation checkpoints are ranked by **parking success rate first**, not by reward alone.

## What is included

- 12 parking bays arranged in two rows, with randomized parked vehicles.
- Feasibility checks for bay occupancy, vehicle fit, final-pose clearance, and entry-corridor clearance.
- Obstacle-aware parking-bay selection that prefers feasible spaces with more clearance.
- Kinematic bicycle vehicle model, collision detection, and bounds checks.
- A tested feedback-controller expert used to generate demonstrations.
- Behavior-cloning pretraining for the actor network.
- PPO by default; SAC is available for a second experiment.
- Curriculum that increases the parked-car density during RL fine-tuning.
- Success-aware checkpoint selection, CSV/JSON metrics, charts, and Pygame visualization.

## Requirements

- Windows 10/11 (commands below use PowerShell)
- Python 3.11, 64-bit
- About 1–2 GB free disk space for dependencies, logs, and model checkpoints
- CPU is supported; CUDA is optional when a compatible PyTorch build is installed

## Install on Windows

Open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use Command Prompt instead:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Recommended run

First run the expert baseline. This checks the environment geometry without training a neural network:

```powershell
python scripts/evaluate.py --expert --episodes 200
```

Then collect expert demonstrations, pretrain the neural-network actor, and fine-tune with PPO for two million environment steps:

```powershell
python scripts/train.py --timesteps 2000000
```

The training command performs these steps automatically:

1. Collects 900 expert episodes from randomized parking lots.
2. Keeps successful trajectories and writes the demonstration dataset to `reports/training/`.
3. Trains a behavior-cloning actor and copies its weights into the PPO policy.
4. Evaluates the initialized policy on 100 unseen scenarios.
5. Fine-tunes the policy with PPO and a difficulty curriculum.
6. Evaluates regularly and saves the best policy using this priority: success rate, collision rate, outside-bounds rate, then mean reward.

The demo collection and BC phase are not RL timesteps; the `--timesteps` value applies to PPO/SAC fine-tuning.

## Watch the result

Show the learned policy:

```powershell
python scripts/visualize.py
```

Show the reference expert controller:

```powershell
python scripts/visualize.py --expert
```

Controls: **SPACE** pause, **N** next scenario, **R** reset, **ESC** exit.

## Evaluate

Evaluate the best learned PPO model on 200 new randomized maps:

```powershell
python scripts/evaluate.py --episodes 200
```

Evaluate the expert baseline on the same difficulty range:

```powershell
python scripts/evaluate.py --expert --episodes 200
```

Outputs:

- `reports/evaluation/episodes.csv` — per-episode details
- `reports/evaluation/summary.json` — aggregate metrics
- `reports/figures/evaluation_results.png` — learned-policy chart
- `reports/evaluation/expert_episodes.csv` and `expert_summary.json` — expert baseline

Important metrics are success rate, collision rate, outside-bounds rate, timeout rate, final position error, and final angle error. Compare the learned policy with the expert baseline; a good reward curve alone is not evidence of successful parking.

## Try SAC

PPO is the default. SAC is included as an alternative continuous-control algorithm and receives the same behavior-cloning initialization:

```powershell
python scripts/train.py --algorithm sac --timesteps 2000000
python scripts/evaluate.py --algorithm sac --episodes 200
python scripts/visualize.py --algorithm sac
```

PPO and SAC both write `models/best_model.zip` and `models/final_model.zip`. To compare them safely, train one algorithm in a separate copy of the project so its model files do not overwrite the other run.

## Continue training

Continue the current algorithm from `models/final_model.zip` for one additional million RL steps:

```powershell
python scripts/train.py --timesteps 1000000 --resume
```

Do not use `--resume` with a model created by v0.2. The observation space and policy initialization changed in v0.3. Start a fresh run instead.

## Useful options

```powershell
python scripts/train.py --timesteps 2000000 --demo-episodes 1200 --bc-epochs 45
python scripts/train.py --algorithm sac --timesteps 1500000 --eval-freq 30000
python scripts/evaluate.py --episodes 500 --difficulty 0.58
python scripts/visualize.py --stochastic
```

## Project structure

```text
AutoPark-RL/
├── configs/
├── models/                       # generated model weights (not committed)
├── reports/
│   ├── evaluation/
│   ├── figures/
│   ├── tensorboard/
│   └── training/
├── scripts/
│   ├── train.py                  # expert -> BC -> PPO/SAC
│   ├── evaluate.py
│   └── visualize.py
├── src/autopark/
│   ├── environment/parking_env.py
│   ├── imitation.py
│   ├── physics.py
│   └── simulation/renderer.py
└── tests/
```

## Reproducibility and limitations

- Training uses a fixed seed by default (`--seed 42`); environment maps are randomized from that seed.
- The expert baseline is a deterministic feedback controller, not a neural network. It provides demonstration labels and a reference success rate.
- The neural network is considered successful only when it reaches the success criteria in the environment: position inside the bay, orientation error below 8 degrees, and speed below 0.16 m/s.
- PPO/SAC success rates must be reported from unseen evaluation episodes. Do not claim the expert's success rate as the learned policy's result.
- The simulator uses a kinematic bicycle model and does not model tire slip, suspension, camera perception, or full vehicle dynamics.

## License

MIT. See [LICENSE](LICENSE).
