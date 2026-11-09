"""Kinematic bicycle model and oriented rectangle collision geometry."""
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
    max_speed: float = 2.2
    max_acceleration: float = 2.6
    max_steering: float = math.radians(34.0)
    rolling_resistance: float = 0.22


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def step_bicycle(state: CarState, action: np.ndarray, dt: float, params: VehicleParams) -> CarState:
    throttle = float(np.clip(action[0], -1.0, 1.0))
    steering = float(np.clip(action[1], -1.0, 1.0))
    acceleration = throttle * params.max_acceleration
    if abs(throttle) < 0.035 and abs(state.speed) > 1e-5:
        acceleration -= math.copysign(params.rolling_resistance, state.speed)
    speed = float(np.clip(state.speed + acceleration * dt, -params.max_speed, params.max_speed))
    steering_angle = steering * params.max_steering
    yaw_rate = speed / params.wheelbase * math.tan(steering_angle)
    heading = normalize_angle(state.heading + yaw_rate * dt)
    avg_heading = normalize_angle(state.heading + 0.5 * yaw_rate * dt)
    return CarState(
        state.x + speed * math.cos(avg_heading) * dt,
        state.y + speed * math.sin(avg_heading) * dt,
        heading,
        speed,
    )


def rectangle_corners(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[-hl,-hw],[hl,-hw],[hl,hw],[-hl,hw]], dtype=np.float64)
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c,-s],[s,c]], dtype=np.float64)
    return local @ rot.T + np.array([x,y], dtype=np.float64)


def rectangles_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-10:
                continue
            axis /= norm
            pa, pb = a @ axis, b @ axis
            if float(pa.max()) < float(pb.min()) or float(pb.max()) < float(pa.min()):
                return False
    return True


def point_in_oriented_rectangle(px: float, py: float, car: CarState, length: float, width: float) -> bool:
    dx, dy = px-car.x, py-car.y
    c, s = math.cos(car.heading), math.sin(car.heading)
    lx, ly = c*dx+s*dy, -s*dx+c*dy
    return abs(lx) <= length/2.0 and abs(ly) <= width/2.0
