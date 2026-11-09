# AutoPark RL v0.4 — Robust Multi-Mode Autonomous Parking

**Python · Reinforcement Learning · Behavior Cloning · PPO · Sensor Fusion · Computer Vision · Pygame**

AutoPark RL is a research simulator in which a policy learns to park a vehicle in randomized lots. Version 0.4 extends the earlier single-task agent into four parking manoeuvres, with a sensor-fusion observation stack and a top-down virtual camera.

> This is a 2D kinematic research simulator. The virtual camera is synthetic and the perception model uses semantic colour segmentation. The project is not a real-vehicle control system and has not been validated for road use.

## What is included

### Four parking modes

| Mode | Manoeuvre | Scenario |
|---|---|---|
| `perpendicular` | Forward-in | Standard 90-degree bay entry |
| `reverse` | Reverse-in | Back into a perpendicular bay |
| `parallel` | Reverse-in | Curved reverse manoeuvre into a curbside bay |
| `tight` | Forward-in | Narrower perpendicular bay with denser neighbouring cars |

Training uses a mixed environment, so the mode is randomized per episode and explicitly encoded in the agent observation. Evaluation reports each mode separately rather than hiding weak modes in one aggregate score.

### Perception and sensor fusion

The policy receives a 75-dimensional observation containing:

- **16-ray LiDAR** around the vehicle, with an 8 m range.
- **4 ultrasonic sensors**: front, rear, left and right, with a 4 m range.
- Vehicle pose, signed speed, target-relative position, heading error and path progress.
- A one-hot parking-mode code and forward/reverse gear indicator.
- Occupancy and feasibility of the candidate bays.
- **Virtual RGB camera (288 × 216)** with a lightweight image-segmentation stage that detects parked vehicles in marked bay regions. The camera occupancy estimate is fused into the observation.

The Pygame viewer displays the top-down lot, selected target, route trace, LiDAR rays, ultrasonic distances and the camera/perception panel.

### Learning pipeline

1. A geometric path-following expert generates successful trajectories for all four modes.
2. Behavior Cloning (BC) pretrains the policy on the expert demonstrations.
3. **PPO only** fine-tunes the policy in randomized environments. SAC is intentionally not included.
4. A curriculum increases parking-lot occupancy during training.
5. Evaluation checkpoints are ranked by aggregate success rate first, then weakest-mode success, safety rates and reward.

## Project structure

```text
AutoPark-RL-v04/
├── configs/
│   └── default.yaml
├── models/                       # generated locally; not included in the archive
│   └── checkpoints/
├── reports/
│   ├── evaluation/
│   ├── figures/
│   ├── tensorboard/
│   └── training/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── record_demo.py
│   └── make_report.py
├── src/autopark/
│   ├── environment/parking_env.py
│   ├── physics.py
│   ├── sensors.py
│   ├── vision.py
│   ├── imitation.py
│   └── simulation/renderer.py
├── tests/
├── requirements.txt
└── README.md
```

## Installation — Windows / PowerShell

Open the project folder in PyCharm, then open its Terminal. Use Python 3.11 (64-bit):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No `.bat` files are required. All operations are launched from the terminal.

## 1. Verify the simulator and expert controller

Run this before starting a long training job:

```powershell
python scripts/evaluate.py --expert --episodes 100
```

By default, `--mode all` evaluates 100 episodes **per mode** (400 in total). The expert is a deterministic geometric path-following controller, not a neural network. It is a reference baseline for checking that generated tasks are physically feasible.

You can isolate one mode:

```powershell
python scripts/evaluate.py --expert --mode parallel --episodes 200
```

## 2. Train BC + PPO across all four modes

```powershell
python scripts/train.py --timesteps 3000000
```

Default run:

- 1,200 balanced expert episodes (300 per mode).
- 30 epochs of Behavior Cloning.
- 3,000,000 PPO environment steps.
- Evaluation every 50,000 steps, 12 episodes per mode.
- Checkpoint every 250,000 steps.
- Difficulty curriculum from 0.08 to 0.62.

Training runs on CPU by default. If a CUDA-enabled PyTorch installation is available, use:

```powershell
python scripts/train.py --timesteps 3000000 --device cuda
```

Continue a previous v0.4 PPO training run:

```powershell
python scripts/train.py --timesteps 1000000 --resume
```

The best policy is saved to `models/best_model.zip`; the latest policy is saved to `models/final_model.zip`. PPO checkpoints are written to `models/checkpoints/`.

## 3. Evaluate the trained policy

```powershell
python scripts/evaluate.py --episodes 200
```

With `--mode all`, this means **200 episodes for each of the four modes** (800 total). Outputs:

- `reports/evaluation/episodes.csv`
- `reports/evaluation/summary.json`
- `reports/evaluation/summary_by_mode.json`
- `reports/figures/evaluation_v04.png`

Evaluate an individual mode:

```powershell
python scripts/evaluate.py --mode tight --episodes 500
```

Increase the density of parked vehicles for a stress test:

```powershell
python scripts/evaluate.py --mode all --episodes 300 --difficulty 0.75
```

Difficulty is a randomized occupancy level, not a formal real-world safety rating.

## 4. Watch the policy

```powershell
python scripts/visualize.py
```

Useful alternatives:

```powershell
python scripts/visualize.py --mode parallel
python scripts/visualize.py --mode tight --difficulty 0.65
python scripts/visualize.py --expert --mode reverse
```

Keyboard controls: `SPACE` pause, `N` next scenario, `R` reset, `1` perpendicular, `2` reverse, `3` parallel, `4` tight, `ESC` quit.

## 5. Record portfolio GIFs

After training, generate one GIF per parking mode:

```powershell
python scripts/record_demo.py --mode all
```

The GIFs are saved in `reports/figures/`. To record the expert baseline instead:

```powershell
python scripts/record_demo.py --mode all --expert
```

## 6. Generate an experiment report

After evaluation:

```powershell
python scripts/make_report.py
```

This creates `reports/evaluation/REPORT.md` with overall and per-mode metrics.

## 7. Run tests

```powershell
python -m pytest -q
```

Tests cover environment shapes, four-mode task creation, expert completion, sensor ranges, camera occupancy segmentation and mutually exclusive episode outcomes.

## Metrics

| Metric | Interpretation |
|---|---|
| Success Rate | Share of episodes where the vehicle stops within position and heading tolerances |
| Collision Rate | Share of episodes ending in an obstacle collision |
| Outside Bounds Rate | Share of episodes ending outside the lot boundary |
| Timeout Rate | Share of episodes reaching the step limit |
| Mean Steps | Average time-to-park in simulation steps |
| Position Error | Final Euclidean distance to target pose |
| Angle Error | Final absolute heading error |
| Camera Occupancy Match | Whether camera-derived occupied-bay count matches the simulated ground truth |

The four terminal outcomes are mutually exclusive in v0.4, so their rates should sum to 100%.

## Compatibility note

The v0.4 observation contains 75 values and the v0.3 policy used a different observation shape. **v0.3 model weights are not compatible with v0.4.** Keep the v0.3 checkpoint as a historical baseline and train a fresh v0.4 model. The archive contains source code only; generated weights and large datasets are intentionally excluded.

## Research limitations and next steps

- Camera perception currently uses a synthetic bird's-eye RGB feed and semantic colour segmentation; it is not a trained camera detector for natural images.
- Vehicle dynamics use a kinematic bicycle model, not a full tire/friction or 3D physics model.
- The expert route controller is a demonstrator and feasibility baseline; the learned PPO policy remains the model under evaluation.
- Strong simulator metrics do not imply transfer to a physical vehicle.

Potential follow-up experiments: noisy sensors, sensor dropouts, camera occlusion, more diverse parking-lot layouts, random vehicle dimensions, domain randomization, and a learned image-based perception front end.
