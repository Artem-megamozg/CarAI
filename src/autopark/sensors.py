"""Ray-cast LiDAR and short-range ultrasonic sensor simulation."""
from __future__ import annotations
import math
from autopark.physics import CarState


def _ray_aabb_local(ox: float, oy: float, dx: float, dy: float, half_l: float, half_w: float, max_range: float) -> float:
    tmin, tmax = 0.0, max_range
    for origin, direction, lo, hi in ((ox,dx,-half_l,half_l),(oy,dy,-half_w,half_w)):
        if abs(direction) < 1e-10:
            if origin < lo or origin > hi:
                return max_range
            continue
        t1=(lo-origin)/direction; t2=(hi-origin)/direction
        if t1>t2: t1,t2=t2,t1
        tmin=max(tmin,t1); tmax=min(tmax,t2)
        if tmin>tmax: return max_range
    return float(tmin if tmin>=0 else tmax) if tmax>=0 else max_range


def ray_distance(origin: CarState, angle: float, obstacles: list[CarState], arena_w: float,
                 arena_h: float, car_length: float, car_width: float, max_range: float=8.0) -> float:
    dx,dy=math.cos(angle),math.sin(angle)
    best=max_range
    # Arena walls.
    if dx>1e-9: best=min(best,(arena_w-origin.x)/dx)
    elif dx<-1e-9: best=min(best,(0.0-origin.x)/dx)
    if dy>1e-9: best=min(best,(arena_h-origin.y)/dy)
    elif dy<-1e-9: best=min(best,(0.0-origin.y)/dy)
    # Oriented vehicles transformed into local rectangle coordinates.
    for obj in obstacles:
        c,s=math.cos(obj.heading),math.sin(obj.heading)
        rx,ry=origin.x-obj.x,origin.y-obj.y
        ox=c*rx+s*ry; oy=-s*rx+c*ry
        ldx=c*dx+s*dy; ldy=-s*dx+c*dy
        hit=_ray_aabb_local(ox,oy,ldx,ldy,car_length/2.0,car_width/2.0,best)
        if 0.05 < hit < best: best=hit
    return float(max(0.05,min(best,max_range)))


def collect_ranges(car: CarState, obstacles: list[CarState], arena_w: float, arena_h: float,
                   car_length: float, car_width: float, lidar_count: int=16, max_range: float=8.0) -> tuple[list[float],list[float]]:
    lidar=[ray_distance(car,car.heading+2*math.pi*i/lidar_count,obstacles,arena_w,arena_h,car_length,car_width,max_range)
           for i in range(lidar_count)]
    # Four bumper-mounted ultrasonic sensors: front, rear, left, right.
    sonar=[]
    c,s=math.cos(car.heading),math.sin(car.heading)
    bumper_origins=(
        CarState(car.x+c*(car_length/2.0-0.06),car.y+s*(car_length/2.0-0.06),car.heading,car.speed),
        CarState(car.x-c*(car_length/2.0-0.06),car.y-s*(car_length/2.0-0.06),car.heading,car.speed),
        CarState(car.x-s*(car_width/2.0-0.06),car.y+c*(car_width/2.0-0.06),car.heading,car.speed),
        CarState(car.x+s*(car_width/2.0-0.06),car.y-c*(car_width/2.0-0.06),car.heading,car.speed),
    )
    for origin,offset in zip(bumper_origins,(0.0,math.pi,math.pi/2,-math.pi/2)):
        sonar.append(ray_distance(origin,car.heading+offset,obstacles,arena_w,arena_h,car_length,car_width,4.0))
    return lidar,sonar
