"""Top-down virtual camera and lightweight semantic bay detector.

The simulator camera renders the parking lot as RGB pixels. The perception
module segments the blue parked-vehicle class inside each marked bay and
returns a camera occupancy estimate that is fused with LiDAR/ultrasonic data.
"""
from __future__ import annotations
import math
from typing import Iterable
import numpy as np
from PIL import Image, ImageDraw
from autopark.physics import CarState, rectangle_corners

CAMERA_WIDTH = 288
CAMERA_HEIGHT = 216
CAR_CLASS = np.array([62, 132, 224], dtype=np.int16)


def _world_to_pixel(x: float, y: float, width: int, height: int, arena_w: float, arena_h: float) -> tuple[int,int]:
    return int(x / arena_w * width), int(y / arena_h * height)


def _draw_car(draw: ImageDraw.ImageDraw, state: CarState, length: float, width: float,
              color: tuple[int,int,int], canvas: tuple[int,int], arena: tuple[float,float]) -> None:
    w,h = canvas; aw,ah = arena
    corners = rectangle_corners(state.x,state.y,state.heading,length,width)
    pts = [_world_to_pixel(float(x),float(y),w,h,aw,ah) for x,y in corners]
    draw.polygon(pts, fill=color, outline=(12,18,24))
    # Small windscreen makes vehicle orientation visible in the camera feed.
    c,s = math.cos(state.heading), math.sin(state.heading)
    local = [(length*.12,-width*.34),(length*.34,-width*.34),(length*.34,width*.34),(length*.12,width*.34)]
    glass=[]
    for lx,ly in local:
        glass.append(_world_to_pixel(state.x+lx*c-ly*s,state.y+lx*s+ly*c,w,h,aw,ah))
    draw.polygon(glass, fill=(32,52,70))


def render_camera(env, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT) -> np.ndarray:
    """Render one semantic RGB frame from the overhead virtual camera."""
    aw, ah = env.ARENA_WIDTH, env.ARENA_HEIGHT
    scale_x, scale_y = width/aw, height/ah
    im = Image.new("RGB", (width,height), (34,43,54))
    d = ImageDraw.Draw(im)
    # Main driving aisle.
    lane_top = int((env.LANE_Y-2.15)*scale_y)
    lane_bottom = int((env.LANE_Y+2.15)*scale_y)
    d.rectangle((0,lane_top,width,lane_bottom), fill=(49,59,71))
    cy = int(env.LANE_Y*scale_y)
    for x in range(0,width,18):
        d.line((x,cy,min(width,x+9),cy), fill=(119,132,146), width=1)
    # Marked bays; the selected bay is yellow and other free bays are green.
    for rec in env.slot_records:
        cx,cyw = float(rec["x"]),float(rec["y"])
        length,width_bay = float(rec["length"]),float(rec["width"])
        heading = float(rec["heading"])
        poly = rectangle_corners(cx,cyw,heading,length,width_bay)
        pts = [_world_to_pixel(float(x),float(y),width,height,aw,ah) for x,y in poly]
        if rec["occupied"]:
            fill=(52,83,115); outline=(77,130,188)
        elif rec["id"] == env.target_slot_id:
            fill=(96,91,43); outline=(255,197,75)
        elif rec["feasible"]:
            fill=(39,83,70); outline=(80,169,132)
        else:
            fill=(77,57,61); outline=(177,91,99)
        d.polygon(pts, fill=fill, outline=outline)
        p = _world_to_pixel(cx,cyw,width,height,aw,ah)
        d.text((p[0]+2,p[1]+2),str(rec["id"]),fill=(232,237,242))
    # Parked vehicles are rendered with a distinct blue semantic class.
    for obstacle in env.obstacles:
        _draw_car(d,obstacle,env.vehicle.length,env.vehicle.width,tuple(int(x) for x in CAR_CLASS),
                  (width,height),(aw,ah))
    # Ego vehicle is yellow and is deliberately excluded from parked-car segmentation.
    _draw_car(d,env.car,env.vehicle.length,env.vehicle.width,(245,190,65),
              (width,height),(aw,ah))
    # Camera reticle / target marker.
    tx,ty = _world_to_pixel(env.target_x,env.target_y,width,height,aw,ah)
    d.ellipse((tx-4,ty-4,tx+4,ty+4),outline=(255,220,126),width=1)
    return np.asarray(im,dtype=np.uint8)


def detect_bay_occupancy(frame: np.ndarray, slot_records: Iterable[dict], arena_w: float, arena_h: float) -> np.ndarray:
    """Estimate bay occupancy by blue-class pixel segmentation in each bay ROI."""
    h,w = frame.shape[:2]
    result=[]
    for rec in slot_records:
        cx=int(float(rec["x"])/arena_w*w); cy=int(float(rec["y"])/arena_h*h)
        # Sample a compact central ROI; it avoids most painted bay borders.
        rx=max(2,int(float(rec["length"])/arena_w*w*.13))
        ry=max(2,int(float(rec["width"])/arena_h*h*.20))
        roi=frame[max(0,cy-ry):min(h,cy+ry+1),max(0,cx-rx):min(w,cx+rx+1)].astype(np.int16)
        blue=(roi[:,:,2] > roi[:,:,0]+35) & (roi[:,:,2] > roi[:,:,1]+25)
        fraction=float(blue.mean()) if blue.size else 0.0
        result.append(1.0 if fraction > 0.08 else 0.0)
    return np.asarray(result,dtype=np.float32)
