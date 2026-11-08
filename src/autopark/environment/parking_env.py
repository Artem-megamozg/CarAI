"""Obstacle-aware multi-bay parking environment for AutoPark RL v0.3."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from autopark.physics import (
    CarState,
    VehicleParams,
    normalize_angle,
    rectangle_corners,
    rectangles_intersect,
    step_bicycle,
)


class ParkingEnv(gym.Env):
    """A multi-bay parking task with a built-in expert demonstrator.

    The parking manager chooses an available bay after checking the parked-car
    geometry and a straight entry corridor. The agent then turns into the aisle,
    aligns with the bay, enters it, and stops inside the marked footprint.
    """

    metadata = {"render_modes": []}

    ARENA_WIDTH = 20.0
    ARENA_HEIGHT = 15.0
    LANE_Y = 7.35
    SLOT_X = tuple(2.9 + 2.82 * i for i in range(6))
    SLOT_Y = (2.65, 11.95)
    SLOT_WIDTH = 2.72
    SLOT_LENGTH = 5.25
    ROW_NAMES = ("A", "B")
    DT = 0.1
    MAX_STEPS = 500
    N_SLOTS = 12
    OBSERVATION_SIZE = 12 + 3 * N_SLOTS

    def __init__(self, max_steps: int = MAX_STEPS, difficulty: float = 0.25):
        super().__init__()
        self.max_steps = int(max_steps)
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.vehicle = VehicleParams()
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.car = CarState(6.0, self.LANE_Y, 0.0, 0.0)
        self.obstacles: list[CarState] = []
        self.occupancy = np.zeros((2, 6), dtype=np.int8)
        self.slot_records: list[dict[str, Any]] = []
        self.target_row = 0
        self.target_col = 0
        self.target_slot_id = "A1"
        self.target_x = self.SLOT_X[0]
        self.target_y = self.SLOT_Y[0]
        self.target_heading = -math.pi / 2.0
        self.approach_sign = 1.0
        self.vertical_sign = -1.0
        self.turn_sign = -1.0
        self.staging_x = self.target_x
        self.staging_y = self.LANE_Y - 3.9
        self.stage = 0
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.previous_potential = 0.0
        self.selected_score = 0.0
        self.last_collision = False

    def set_difficulty(self, difficulty: float) -> None:
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    @staticmethod
    def _slot_id(row: int, col: int) -> str:
        return f"{ParkingEnv.ROW_NAMES[row]}{col + 1}"

    def _car_polygon(self, state: CarState, margin: float = 0.0) -> np.ndarray:
        return rectangle_corners(
            state.x,
            state.y,
            state.heading,
            self.vehicle.length + 2.0 * margin,
            self.vehicle.width + 2.0 * margin,
        )

    def _inside_arena(self, polygon: np.ndarray) -> bool:
        return bool(
            np.all(polygon[:, 0] >= 0.12)
            and np.all(polygon[:, 0] <= self.ARENA_WIDTH - 0.12)
            and np.all(polygon[:, 1] >= 0.12)
            and np.all(polygon[:, 1] <= self.ARENA_HEIGHT - 0.12)
        )

    def _pose_is_clear(
        self,
        state: CarState,
        obstacles: list[CarState],
        margin: float = 0.0,
    ) -> bool:
        polygon = self._car_polygon(state, margin=margin)
        if not self._inside_arena(polygon):
            return False
        return not any(
            rectangles_intersect(polygon, self._car_polygon(obstacle))
            for obstacle in obstacles
        )

    def _approach_is_clear(
        self,
        x: float,
        y: float,
        heading: float,
        obstacles: list[CarState],
    ) -> bool:
        for t in np.linspace(0.0, 1.0, 22):
            sample_y = self.LANE_Y + (y - self.LANE_Y) * float(t)
            if not self._pose_is_clear(
                CarState(x, sample_y, heading, 0.0), obstacles, margin=0.04
            ):
                return False
        return True

    def _build_slot_records(
        self,
        occupancy: np.ndarray,
        obstacles: list[CarState],
        start_x: float,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        fits = (
            self.SLOT_WIDTH >= self.vehicle.width + 0.18
            and self.SLOT_LENGTH >= self.vehicle.length + 0.18
        )
        for row in range(2):
            heading = -math.pi / 2.0 if row == 0 else math.pi / 2.0
            for col, x in enumerate(self.SLOT_X):
                y = self.SLOT_Y[row]
                occupied = bool(occupancy[row, col])
                goal_clear = self._pose_is_clear(
                    CarState(float(x), y, heading, 0.0), obstacles, margin=0.08
                )
                entry_clear = self._approach_is_clear(x, y, heading, obstacles)
                feasible = bool(not occupied and fits and goal_clear and entry_clear)

                adjacent = int(col > 0 and bool(occupancy[row, col - 1]))
                adjacent += int(col < 5 and bool(occupancy[row, col + 1]))
                adjacent += int(bool(occupancy[1 - row, col]))
                distance = abs(float(x) - start_x)
                # Penalize tight neighbors; use distance only as a tie-breaker.
                score = 1.2 * adjacent + 0.055 * distance
                records.append(
                    {
                        "id": self._slot_id(row, col),
                        "row": row,
                        "col": col,
                        "x": float(x),
                        "y": float(y),
                        "heading": float(heading),
                        "occupied": occupied,
                        "feasible": feasible,
                        "score": float(score),
                        "occupied_neighbors": adjacent,
                        "distance_from_start": float(distance),
                    }
                )
        return records

    def _make_map(self, reference_x: float) -> None:
        """Generate a randomized lot and select a reachable, empty bay."""
        occupancy_probability = 0.10 + 0.40 * self.difficulty
        chosen_records = None
        chosen_obstacles: list[CarState] = []
        chosen_candidates: list[dict[str, Any]] = []

        for _ in range(100):
            occupancy = (
                self.np_random.random((2, 6)) < occupancy_probability
            ).astype(np.int8)
            if int((occupancy == 0).sum()) < 4:
                continue

            obstacles: list[CarState] = []
            for row in range(2):
                heading = -math.pi / 2.0 if row == 0 else math.pi / 2.0
                for col, x in enumerate(self.SLOT_X):
                    if occupancy[row, col]:
                        obstacles.append(CarState(float(x), self.SLOT_Y[row], heading, 0.0))

            records = self._build_slot_records(occupancy, obstacles, reference_x)
            candidates = [r for r in records if r["feasible"]]
            if len(candidates) >= 2:
                chosen_records, chosen_obstacles = records, obstacles
                chosen_candidates = candidates
                self.occupancy = occupancy
                break
        else:
            # Keep the environment valid even in a rare over-constrained draw.
            occupancy = np.zeros((2, 6), dtype=np.int8)
            obstacles = []
            records = self._build_slot_records(occupancy, obstacles, reference_x)
            chosen_records = records
            chosen_obstacles = obstacles
            chosen_candidates = [r for r in records if r["feasible"]]
            self.occupancy = occupancy

        if not chosen_candidates:
            raise RuntimeError("Parking map generator failed to produce a feasible bay")

        selected = min(
            chosen_candidates,
            key=lambda r: r["score"] + float(self.np_random.uniform(0.0, 0.9)),
        )
        self.obstacles = chosen_obstacles
        self.slot_records = chosen_records or []
        self.target_row = int(selected["row"])
        self.target_col = int(selected["col"])
        self.target_slot_id = str(selected["id"])
        self.target_x = float(selected["x"])
        self.target_y = float(selected["y"])
        self.target_heading = float(selected["heading"])
        self.vertical_sign = -1.0 if self.target_row == 0 else 1.0
        self.selected_score = float(selected["score"])

        turn_radius = self.vehicle.wheelbase / math.tan(self.vehicle.max_steering)
        viable = []
        for sign in (1.0, -1.0):
            start_candidate = self.target_x - sign * turn_radius
            if 2.5 <= start_candidate <= self.ARENA_WIDTH - 2.5:
                viable.append((abs(start_candidate - reference_x), sign, start_candidate))
        if not viable:
            raise RuntimeError(f"No safe turning start for bay {self.target_slot_id}")
        _, self.approach_sign, base_start_x = min(viable, key=lambda item: item[0])
        self.turn_sign = self.vertical_sign * self.approach_sign
        self.staging_x = self.target_x
        self.staging_y = self.LANE_Y + self.vertical_sign * turn_radius

        for record in self.slot_records:
            record["selected"] = record["id"] == self.target_slot_id
            record["start_x"] = float(base_start_x)

    def _target_distance(self) -> float:
        return math.hypot(self.target_x - self.car.x, self.target_y - self.car.y)

    def _staging_distance(self) -> float:
        return math.hypot(self.staging_x - self.car.x, self.staging_y - self.car.y)

    def _angle_error(self) -> float:
        return normalize_angle(self.target_heading - self.car.heading)

    def _bearing_error(self) -> float:
        dx = self.target_x - self.car.x
        dy = self.target_y - self.car.y
        if abs(dx) + abs(dy) < 1e-8:
            return self._angle_error()
        return normalize_angle(math.atan2(dy, dx) - self.car.heading)

    def _potential(self) -> float:
        if self.stage == 0:
            return 0.75 * self._staging_distance() + 1.8 * abs(self._angle_error())
        return self._target_distance() + 0.8 * abs(self._angle_error())

    def _observation(self) -> np.ndarray:
        waypoint_x = self.staging_x if self.stage == 0 else self.target_x
        waypoint_y = self.staging_y if self.stage == 0 else self.target_y
        angle_error = self._angle_error()
        bearing_error = self._bearing_error()
        values: list[float] = [
            2.0 * self.car.x / self.ARENA_WIDTH - 1.0,
            2.0 * self.car.y / self.ARENA_HEIGHT - 1.0,
            math.sin(self.car.heading),
            math.cos(self.car.heading),
            self.car.speed / self.vehicle.max_speed,
            (waypoint_x - self.car.x) / self.ARENA_WIDTH,
            (waypoint_y - self.car.y) / self.ARENA_HEIGHT,
            math.sin(angle_error),
            math.cos(angle_error),
            float(self.stage),
            math.sin(bearing_error),
            math.cos(bearing_error),
        ]
        values.extend(float(r["occupied"]) for r in self.slot_records)
        values.extend(float(r["feasible"]) for r in self.slot_records)
        values.extend(float(r.get("selected", False)) for r in self.slot_records)
        return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)

    def _check_collision(self) -> tuple[bool, bool]:
        polygon = self._car_polygon(self.car)
        if not self._inside_arena(polygon):
            return False, True
        for obstacle in self.obstacles:
            if rectangles_intersect(polygon, self._car_polygon(obstacle)):
                return True, False
        return False, False

    def _is_success(self) -> bool:
        dx = self.car.x - self.target_x
        dy = self.car.y - self.target_y
        longitudinal = abs(dx * math.cos(self.target_heading) + dy * math.sin(self.target_heading))
        lateral = abs(-dx * math.sin(self.target_heading) + dy * math.cos(self.target_heading))
        return bool(
            longitudinal < 0.42
            and lateral < 0.32
            and abs(self._angle_error()) < math.radians(8.0)
            and abs(self.car.speed) < 0.16
        )

    def expert_action(self) -> np.ndarray:
        """Return a tested feedback-control action for demonstrations/baselines."""
        if self.stage == 0:
            angle_error = self._angle_error()
            steering = self.turn_sign if abs(angle_error) > math.radians(3.0) else 0.0
            target_speed = 0.65
        else:
            dx = self.target_x - self.car.x
            dy = self.target_y - self.car.y
            longitudinal = dx * math.cos(self.target_heading) + dy * math.sin(self.target_heading)
            lateral = -dx * math.sin(self.target_heading) + dy * math.cos(self.target_heading)
            heading_correction = math.atan2(lateral, max(abs(longitudinal), 0.55))
            steering = float(np.clip(2.4 * (self._angle_error() + heading_correction), -1.0, 1.0))
            distance = self._target_distance()
            target_speed = min(
                0.65,
                math.sqrt(max(0.0, 2.0 * 1.4 * max(longitudinal - 0.10, 0.0))),
            )
            # Do not freeze just outside the success-angle tolerance: keep a
            # very slow correction movement so the bicycle model can align.
            if distance < 0.35 and abs(self._angle_error()) > math.radians(5.5):
                target_speed = 0.18
        throttle = float(np.clip(2.0 * (target_speed - self.car.speed), -1.0, 1.0))
        return np.asarray([throttle, steering], dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        reference_x = float(self.np_random.uniform(2.5, 17.5))
        self._make_map(reference_x)

        if "target_slot" in options:
            requested = str(options["target_slot"]).upper()
            selected = next(
                (r for r in self.slot_records if r["id"] == requested and r["feasible"]),
                None,
            )
            if selected is None:
                raise ValueError(f"Requested target slot {requested!r} is not feasible")
            self.target_row = int(selected["row"])
            self.target_col = int(selected["col"])
            self.target_slot_id = str(selected["id"])
            self.target_x = float(selected["x"])
            self.target_y = float(selected["y"])
            self.target_heading = float(selected["heading"])
            self.vertical_sign = -1.0 if self.target_row == 0 else 1.0
            turn_radius = self.vehicle.wheelbase / math.tan(self.vehicle.max_steering)
            viable = [
                (abs(self.target_x - sign * turn_radius - reference_x), sign,
                 self.target_x - sign * turn_radius)
                for sign in (1.0, -1.0)
                if 2.5 <= self.target_x - sign * turn_radius <= self.ARENA_WIDTH - 2.5
            ]
            _, self.approach_sign, _ = min(viable, key=lambda item: item[0])
            self.turn_sign = self.vertical_sign * self.approach_sign
            self.staging_x = self.target_x
            self.staging_y = self.LANE_Y + self.vertical_sign * turn_radius
            self.selected_score = float(selected["score"])
            for record in self.slot_records:
                record["selected"] = record["id"] == self.target_slot_id

        turn_radius = self.vehicle.wheelbase / math.tan(self.vehicle.max_steering)
        base_start_x = self.target_x - self.approach_sign * turn_radius
        start_x = base_start_x + float(self.np_random.uniform(-0.08, 0.08))
        start_y = self.LANE_Y + float(self.np_random.uniform(-0.08, 0.08))
        if "start" in options:
            start = options["start"]
            start_x = float(start.get("x", start_x))
            start_y = float(start.get("y", start_y))
        start_heading = 0.0 if self.approach_sign > 0 else math.pi
        if "start" in options and "heading" in options["start"]:
            start_heading = float(options["start"]["heading"])
        else:
            start_heading += float(self.np_random.uniform(-0.015, 0.015))

        self.car = CarState(start_x, start_y, normalize_angle(start_heading), 0.0)
        if not self._pose_is_clear(self.car, self.obstacles):
            raise RuntimeError("Generated initial car pose intersects an obstacle or arena boundary")

        self.stage = 0
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_collision = False
        self.previous_potential = self._potential()
        return self._observation(), self._info(False, False, False)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.copy()
        self.car = step_bicycle(self.car, action, self.DT, self.vehicle)
        self.step_count += 1

        collision, outside = self._check_collision()
        success = self._is_success()
        stage_advanced = False
        if self.stage == 0:
            if (
                abs(self._angle_error()) < math.radians(4.0)
                and abs(self.car.speed) < 0.95
            ):
                self.stage = 1
                stage_advanced = True

        current_potential = self._potential()
        reward = 2.2 * (self.previous_potential - current_potential) - 0.025
        if self.stage == 0:
            reward += 0.035 * max(0.0, self.car.speed)
        else:
            reward += 0.06 * max(0.0, 1.0 - self._target_distance())
            reward -= 0.035 * abs(self.car.speed)
        if stage_advanced:
            reward += 18.0

        terminated = False
        truncated = False
        if collision:
            reward -= 160.0
            terminated = True
        elif outside:
            reward -= 160.0
            terminated = True
        elif success:
            reward += 180.0
            terminated = True
        elif self.step_count >= self.max_steps:
            reward -= 25.0
            truncated = True

        self.previous_potential = current_potential
        self.last_collision = collision
        info = self._info(success, collision, outside)
        info["stage_advanced"] = stage_advanced
        return self._observation(), float(reward), terminated, truncated, info

    def _info(self, success: bool, collision: bool, outside: bool) -> dict[str, Any]:
        return {
            "success": bool(success),
            "collision": bool(collision),
            "outside_bounds": bool(outside),
            "target_slot": self.target_slot_id,
            "free_slots": int(sum(not bool(r["occupied"]) for r in self.slot_records)),
            "feasible_slots": int(sum(bool(r["feasible"]) for r in self.slot_records)),
            "occupied_slots": int(len(self.obstacles)),
            "distance": float(self._target_distance()),
            "staging_distance": float(self._staging_distance()),
            "angle_error_deg": float(math.degrees(abs(self._angle_error()))),
            "bearing_error_deg": float(math.degrees(abs(self._bearing_error()))),
            "speed": float(self.car.speed),
            "stage": int(self.stage),
            "steps": int(self.step_count),
        }

    def close(self) -> None:
        return None
