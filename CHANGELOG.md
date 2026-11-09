# Changelog

## 0.4.0

- Added four scenarios: perpendicular, reverse, parallel and tight parking.
- Added randomized mixed-mode task generation and occupancy curriculum.
- Added 16-ray LiDAR and four bumper-mounted ultrasonic sensors.
- Added a 288x216 synthetic RGB bird's-eye camera and semantic occupied-bay segmentation.
- Fused camera-derived occupancy, bay feasibility and range readings into a 75-value observation.
- Kept PPO as the only reinforcement-learning algorithm; removed SAC support.
- Added balanced expert demonstrations, Behavior Cloning initialization and success-aware PPO checkpoint selection.
- Added per-mode evaluation, camera occupancy checks, Pygame overlays, GIF recording and Markdown report generation.
- Kept all model weights and generated datasets outside the source archive.
