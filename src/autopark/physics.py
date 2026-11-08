"""Kinematic bicycle model and oriented-rectangle geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class CarState:
    x: float
    y: float
    heading: float
    speed: float = 0.0


@dataclass(frozen=True)
class VehicleParams:
    length: float = 4.2
    width: float = 1.8
    wheelbase: float = 2.65
    max_speed: float = 3.0
    max_acceleration: float = 2.4
    max_steering: float = math.radians(34.0)
    rolling_resistance: float = 0.18


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def step_bicycle(
    state: CarState,
    action: np.ndarray,
    dt: float,
    params: VehicleParams,
) -> CarState:
    """Advance the car using signed throttle and normalized steering."""
    throttle = float(np.clip(action[0], -1.0, 1.0))
    steering_input = float(np.clip(action[1], -1.0, 1.0))

    acceleration = throttle * params.max_acceleration
    if abs(throttle) < 0.035 and abs(state.speed) > 1e-5:
        acceleration -= math.copysign(params.rolling_resistance, state.speed)

    speed = float(np.clip(
        state.speed + acceleration * dt,
        -params.max_speed,
        params.max_speed,
    ))

    steering_angle = steering_input * params.max_steering
    yaw_rate = speed / params.wheelbase * math.tan(steering_angle)
    heading = normalize_angle(state.heading + yaw_rate * dt)
    average_heading = normalize_angle(state.heading + 0.5 * yaw_rate * dt)
    x = state.x + speed * math.cos(average_heading) * dt
    y = state.y + speed * math.sin(average_heading) * dt
    return CarState(x=x, y=y, heading=heading, speed=speed)


def rectangle_corners(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
) -> np.ndarray:
    """Return the four world-coordinate corners of an oriented rectangle."""
    half_l, half_w = length / 2.0, width / 2.0
    local = np.array(
        [
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    c, s = math.cos(heading), math.sin(heading)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + np.array([x, y], dtype=np.float64)


def rectangles_intersect(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Separating Axis Theorem for two convex quadrilaterals."""
    for polygon in (poly_a, poly_b):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-10:
                continue
            axis /= norm
            projection_a = poly_a @ axis
            projection_b = poly_b @ axis
            if (
                float(projection_a.max()) < float(projection_b.min())
                or float(projection_b.max()) < float(projection_a.min())
            ):
                return False
    return True
