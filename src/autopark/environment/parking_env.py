"""Four-mode multi-bay parking environment with sensor fusion and CV perception."""
from __future__ import annotations
import math
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from autopark.physics import (CarState, VehicleParams, normalize_angle, rectangle_corners,
                               rectangles_intersect, step_bicycle)
from autopark.sensors import collect_ranges
from autopark.vision import render_camera, detect_bay_occupancy

MODES = ("perpendicular", "reverse", "parallel", "tight")
MODE_INDEX = {name:i for i,name in enumerate(MODES)}


def _bezier(p0, p1, p2, p3, n=120):
    t=np.linspace(0.0,1.0,n,dtype=np.float64)[:,None]
    omt=1.0-t
    points=(omt**3)*np.asarray(p0)+(3*omt**2*t)*np.asarray(p1)+(3*omt*t**2)*np.asarray(p2)+(t**3)*np.asarray(p3)
    deriv=(3*omt**2)*(np.asarray(p1)-np.asarray(p0))+(6*omt*t)*(np.asarray(p2)-np.asarray(p1))+(3*t**2)*(np.asarray(p3)-np.asarray(p2))
    headings=np.arctan2(deriv[:,1],deriv[:,0])
    return points, headings


class ParkingEnv(gym.Env):
    """Agent parks a car in one of four scenarios.

    Observation is a fused low-dimensional vector: vehicle state, target-relative
    pose, mode encoding, 16 LiDAR rays, four ultrasonic ranges, bay occupancy,
    free-space feasibility, and occupancy detected from the virtual RGB camera.
    """
    metadata={"render_modes":[]}
    ARENA_WIDTH=24.0
    ARENA_HEIGHT=18.0
    LANE_Y=9.0
    DT=0.10
    MAX_STEPS=650
    LIDAR_COUNT=16
    MAX_SENSOR_RANGE=8.0
    MAX_SLOTS=12
    # 13 state + 4 mode + 2 gear + 16 lidar + 4 sonar + 3*12 slot/camera arrays = 75.
    OBSERVATION_SIZE=75

    def __init__(self, mode: str="mixed", max_steps: int=MAX_STEPS, difficulty: float=0.25,
                 seed: int|None=None):
        super().__init__()
        if mode not in (*MODES,"mixed","all"):
            raise ValueError(f"mode must be one of {MODES}, 'mixed' or 'all'")
        self.requested_mode=mode
        self.mode="perpendicular"
        self.max_steps=int(max_steps)
        self.difficulty=float(np.clip(difficulty,0.0,1.0))
        self.vehicle=VehicleParams()
        self.action_space=spaces.Box(low=np.array([-1.,-1.],np.float32),high=np.array([1.,1.],np.float32),dtype=np.float32)
        self.observation_space=spaces.Box(low=-1.0,high=1.0,shape=(self.OBSERVATION_SIZE,),dtype=np.float32)
        self.car=CarState(5.0,self.LANE_Y,0.0,0.0)
        self.obstacles:list[CarState]=[]
        self.slot_records:list[dict[str,Any]]=[]
        self.occupancy=np.zeros(self.MAX_SLOTS,dtype=np.int8)
        self.camera_occupancy=np.zeros(self.MAX_SLOTS,dtype=np.float32)
        self.target_slot_id="A1"
        self.target_x=3.0; self.target_y=3.0; self.target_heading=-math.pi/2
        self.target_length=5.2; self.target_width=2.55
        self.path=np.zeros((2,2),dtype=np.float64)
        self.path_headings=np.zeros(2,dtype=np.float64)
        self.path_cumulative=np.zeros(2,dtype=np.float64)
        self.reverse_motion=False
        self.step_count=0
        self.previous_potential=0.0
        self.last_action=np.zeros(2,dtype=np.float32)
        self.last_lidar=[8.0]*self.LIDAR_COUNT
        self.last_sonar=[4.0]*4
        self._camera_frame=None
        self._rng_seed=seed

    def set_difficulty(self,difficulty:float)->None:
        self.difficulty=float(np.clip(difficulty,0.0,1.0))

    @staticmethod
    def _bay_id(row:int,col:int)->str:
        return f"{'A' if row==0 else 'B'}{col+1}"

    def _mode_layout(self):
        if self.mode=="parallel":
            xs=[4.0,9.3,14.7,20.0]
            ys=[5.55,12.45]
            return [(r,c,x,ys[r],0.0 if r==0 else math.pi,5.15,2.55)
                    for r,y in enumerate(ys) for c,x in enumerate(xs)]
        xs=[3.0,6.6,10.2,13.8,17.4,21.0]
        ys=[3.0,15.0]
        width=2.18 if self.mode=="tight" else 2.55
        length=4.72 if self.mode=="tight" else 5.25
        return [(r,c,x,ys[r],-math.pi/2 if r==0 else math.pi/2,length,width)
                for r,y in enumerate(ys) for c,x in enumerate(xs)]

    def _poly(self,state:CarState,margin:float=0.0)->np.ndarray:
        return rectangle_corners(state.x,state.y,state.heading,
                                 self.vehicle.length+2*margin,self.vehicle.width+2*margin)

    def _inside_arena(self,poly:np.ndarray)->bool:
        return bool(np.all(poly[:,0]>=0.12) and np.all(poly[:,0]<=self.ARENA_WIDTH-.12)
                    and np.all(poly[:,1]>=0.12) and np.all(poly[:,1]<=self.ARENA_HEIGHT-.12))

    def _pose_clear(self,state:CarState,obstacles:list[CarState],margin:float=0.0)->bool:
        poly=self._poly(state,margin)
        if not self._inside_arena(poly): return False
        return not any(rectangles_intersect(poly,self._poly(o)) for o in obstacles)

    def _path_for_slot(self,rec:dict[str,Any]):
        tx,ty=rec["x"],rec["y"]
        if self.mode=="parallel":
            side=1.0 if tx<12.0 else -1.0
            p0=(tx+side*6.0,self.LANE_Y)
            p1=(tx+side*4.1,self.LANE_Y)
            p2=(tx+side*1.9,ty)
            p3=(tx,ty)
            points,tangents=_bezier(p0,p1,p2,p3)
            # Reverse into a curbside bay; the vehicle body faces opposite its travel vector.
            headings=np.asarray([normalize_angle(float(a)+math.pi) for a in tangents])
            target_heading=0.0 if side>0 else math.pi
            start_heading=target_heading
            reverse=True
        else:
            side=1.0 if tx<12.0 else -1.0
            vertical=-1.0 if rec["row"]==0 else 1.0
            p0=(tx+side*5.0,self.LANE_Y)
            p1=(tx+side*2.5,self.LANE_Y)
            p2=(tx,ty-vertical*2.4)
            p3=(tx,ty)
            points,tangents=_bezier(p0,p1,p2,p3)
            reverse=self.mode=="reverse"
            headings=np.asarray([normalize_angle(float(a)+(math.pi if reverse else 0.0)) for a in tangents])
            target_heading=float(headings[-1])
            start_heading=float(headings[0])
        return points,headings,start_heading,target_heading,reverse

    def _path_is_clear(self,points:np.ndarray,headings:np.ndarray,obstacles:list[CarState])->bool:
        # Subsample every second spline point and retain a small safety margin.
        for i in range(0,len(points),2):
            state=CarState(float(points[i,0]),float(points[i,1]),float(headings[i]),0.0)
            if not self._pose_clear(state,obstacles,margin=0.035):
                return False
        state=CarState(float(points[-1,0]),float(points[-1,1]),float(headings[-1]),0.0)
        return self._pose_clear(state,obstacles,margin=0.0)

    def _make_map(self,reference_x:float)->None:
        specs=self._mode_layout()
        prob=0.08+0.48*self.difficulty
        chosen=None
        for _ in range(100):
            occupancy=(self.np_random.random(len(specs))<prob).astype(np.int8)
            obstacles=[]
            records=[]
            for idx,(row,col,x,y,heading,length,width) in enumerate(specs):
                records.append({"id":self._bay_id(row,col),"row":row,"col":col,"x":float(x),"y":float(y),
                                "heading":float(heading),"length":float(length),"width":float(width),
                                "occupied":bool(occupancy[idx]),"feasible":False,"selected":False})
                if occupancy[idx]: obstacles.append(CarState(float(x),float(y),float(heading),0.0))
            candidates=[]
            for rec in records:
                if rec["occupied"]: continue
                # Vehicle fit, geometric clearance and the full approach path must all be valid.
                if rec["length"] < self.vehicle.length+0.22 or rec["width"] < self.vehicle.width+0.22:
                    continue
                points,heads,start_h,target_h,reverse=self._path_for_slot(rec)
                if not self._path_is_clear(points,heads,obstacles): continue
                rec["feasible"]=True
                neighbor_penalty=0.0
                for other in records:
                    if other["occupied"] and other["row"]==rec["row"] and abs(other["col"]-rec["col"])==1:
                        neighbor_penalty+=0.8
                distance=float(np.linalg.norm(points[0]-np.array([reference_x,self.LANE_Y])))
                rec["score"]=0.045*distance+neighbor_penalty
                candidates.append(rec)
            if len(candidates)>=2:
                chosen=(records,obstacles,candidates)
                break
        if chosen is None:
            # Fallback: completely empty but still randomized mode geometry.
            occupancy=np.zeros(len(specs),dtype=np.int8)
            obstacles=[]; records=[]; candidates=[]
            for idx,(row,col,x,y,heading,length,width) in enumerate(specs):
                rec={"id":self._bay_id(row,col),"row":row,"col":col,"x":float(x),"y":float(y),
                     "heading":float(heading),"length":float(length),"width":float(width),
                     "occupied":False,"feasible":False,"selected":False}
                points,heads,sh,th,rev=self._path_for_slot(rec)
                rec["feasible"]=self._path_is_clear(points,heads,obstacles)
                rec["score"]=0.04*float(np.linalg.norm(points[0]-np.array([reference_x,self.LANE_Y])))
                records.append(rec)
                if rec["feasible"]: candidates.append(rec)
            if len(candidates)<1: raise RuntimeError(f"Could not generate a feasible map for mode={self.mode}")
            chosen=(records,obstacles,candidates)
        records,obstacles,candidates=chosen
        scores=np.asarray([r.get("score",0.0) for r in candidates],dtype=np.float64)
        weights=np.exp(-(scores-scores.min())/0.7); weights/=weights.sum()
        selected=candidates[int(self.np_random.choice(len(candidates),p=weights))]
        selected["selected"]=True
        self.slot_records=records
        self.obstacles=obstacles
        self.target_slot_id=selected["id"]
        self.target_x=float(selected["x"]); self.target_y=float(selected["y"])
        self.target_length=float(selected["length"]); self.target_width=float(selected["width"])
        self.path,self.path_headings,start_heading,self.target_heading,self.reverse_motion=self._path_for_slot(selected)
        diffs=np.diff(self.path,axis=0)
        self.path_cumulative=np.concatenate([[0.0],np.cumsum(np.linalg.norm(diffs,axis=1))])
        self.car=CarState(float(self.path[0,0]),float(self.path[0,1]),float(start_heading),0.0)
        # Mild randomized pose noise encourages sensor robustness without breaking expert reachability.
        self.car.x+=float(self.np_random.uniform(-0.025,0.025))
        self.car.y+=float(self.np_random.uniform(-0.025,0.025))
        self.car.heading=normalize_angle(self.car.heading+float(self.np_random.uniform(-0.01,0.01)))
        if not self._pose_clear(self.car,self.obstacles):
            raise RuntimeError("Generated initial pose collides with an obstacle")
        self.camera_frame=render_camera(self)
        detected=detect_bay_occupancy(self.camera_frame,self.slot_records,self.ARENA_WIDTH,self.ARENA_HEIGHT)
        self.camera_occupancy=np.zeros(self.MAX_SLOTS,dtype=np.float32)
        self.camera_occupancy[:min(len(detected),self.MAX_SLOTS)]=detected[:self.MAX_SLOTS]
        self.bay_occupancy=np.zeros(self.MAX_SLOTS,dtype=np.float32)
        self.bay_feasible=np.zeros(self.MAX_SLOTS,dtype=np.float32)
        for i,r in enumerate(self.slot_records[:self.MAX_SLOTS]):
            self.bay_occupancy[i]=float(r["occupied"])
            self.bay_feasible[i]=float(r["feasible"])
        # For parallel mode only 8 marked bays exist; the remaining slots are padded as unavailable.

    def _target_distance(self)->float:
        return math.hypot(self.target_x-self.car.x,self.target_y-self.car.y)

    def _angle_error(self)->float:
        return normalize_angle(self.target_heading-self.car.heading)

    def _local_error(self)->tuple[float,float]:
        dx,dy=self.target_x-self.car.x,self.target_y-self.car.y
        c,s=math.cos(self.target_heading),math.sin(self.target_heading)
        longitudinal=dx*c+dy*s
        lateral=-dx*s+dy*c
        return longitudinal,lateral

    def _path_remaining(self)->float:
        distances=np.linalg.norm(self.path-np.array([self.car.x,self.car.y]),axis=1)
        idx=int(np.argmin(distances))
        return float(max(0.0,self.path_cumulative[-1]-self.path_cumulative[idx]))

    def _potential(self)->float:
        longitudinal,lateral=self._local_error()
        return self._target_distance()+0.85*abs(self._angle_error())+0.18*abs(lateral)

    def _observation(self)->np.ndarray:
        lidar,sonar=collect_ranges(self.car,self.obstacles,self.ARENA_WIDTH,self.ARENA_HEIGHT,
                                   self.vehicle.length,self.vehicle.width,self.LIDAR_COUNT,self.MAX_SENSOR_RANGE)
        self.last_lidar,self.last_sonar=lidar,sonar
        dx,dy=self.target_x-self.car.x,self.target_y-self.car.y
        c,s=math.cos(self.car.heading),math.sin(self.car.heading)
        local_forward=(c*dx+s*dy)/12.0
        local_left=(-s*dx+c*dy)/12.0
        angle_err=self._angle_error()
        mode_onehot=[1.0 if self.mode==m else 0.0 for m in MODES]
        longitudinal,lateral=self._local_error()
        path_remaining=self._path_remaining()/20.0
        progress=1.0-path_remaining
        state=[
            2*self.car.x/self.ARENA_WIDTH-1,
            2*self.car.y/self.ARENA_HEIGHT-1,
            math.sin(self.car.heading),math.cos(self.car.heading),
            self.car.speed/self.vehicle.max_speed,
            float(np.clip(local_forward,-1,1)),float(np.clip(local_left,-1,1)),
            math.sin(angle_err),math.cos(angle_err),
            float(np.clip(self._target_distance()/15.0,0,1)),
            float(np.clip(path_remaining,0,1)),float(np.clip(progress,0,1)),
            float(np.clip(lateral/5.0,-1,1)),
        ]
        gear=[0.0,1.0] if self.reverse_motion else [1.0,0.0]
        values=state+mode_onehot+gear
        values += [float(np.clip(v/self.MAX_SENSOR_RANGE,0,1)) for v in lidar]
        values += [float(np.clip(v/4.0,0,1)) for v in sonar]
        values += self.bay_occupancy.tolist()
        values += self.bay_feasible.tolist()
        values += self.camera_occupancy.tolist()
        arr=np.asarray(values,dtype=np.float32)
        if arr.shape!=(self.OBSERVATION_SIZE,):
            raise RuntimeError(f"Observation shape mismatch: got {arr.shape}, expected {(self.OBSERVATION_SIZE,)}")
        return np.clip(arr,-1.0,1.0)

    def expert_action(self)->np.ndarray:
        """Pure-pursuit path-tracking controller used for BC demonstrations and baseline."""
        distances=np.linalg.norm(self.path-np.array([self.car.x,self.car.y]),axis=1)
        nearest=int(np.argmin(distances))
        lookahead=0.85+0.18*abs(self.car.speed)
        desired_s=min(float(self.path_cumulative[-1]),float(self.path_cumulative[nearest])+lookahead)
        target_idx=int(np.searchsorted(self.path_cumulative,desired_s,side="left"))
        target_idx=min(target_idx,len(self.path)-1)
        target=self.path[target_idx]
        dx,dy=float(target[0]-self.car.x),float(target[1]-self.car.y)
        travel_heading=normalize_angle(self.car.heading+(math.pi if self.reverse_motion else 0.0))
        desired_travel=math.atan2(dy,dx)
        alpha=normalize_angle(desired_travel-travel_heading)
        distance=max(0.45,math.hypot(dx,dy))
        curvature=2.0*math.sin(alpha)/distance
        steering_angle=math.atan(self.vehicle.wheelbase*curvature)
        # With signed negative velocity, the bicycle yaw response reverses.
        if self.reverse_motion:
            steering_angle=-steering_angle
        steering=float(np.clip(steering_angle/self.vehicle.max_steering,-1.0,1.0))
        remaining=self._target_distance()
        target_speed=min(1.25,math.sqrt(max(0.0,2.0*1.0*max(remaining-0.18,0.0))))
        if remaining<0.38:
            target_speed=0.0
        signed_target=-target_speed if self.reverse_motion else target_speed
        throttle=float(np.clip(1.9*(signed_target-self.car.speed),-1.0,1.0))
        # Correct the final heading gently when the position is already close.
        if remaining<0.75 and (abs(self._angle_error())>math.radians(2.5) or remaining>0.18):
            err=normalize_angle(self.target_heading-self.car.heading)
            longitudinal,lateral=self._local_error()
            # Choose the gear that moves toward the target; maintain a tiny rolling
            # speed so steering can still change the heading at the final pose.
            direction = 1.0 if longitudinal >= 0.0 else -1.0
            if abs(longitudinal)<0.035:
                direction = 1.0
            correction=2.5*err/self.vehicle.max_steering
            if direction < 0.0: correction=-correction
            steering=float(np.clip(correction,-0.58,0.58))
            if abs(self._angle_error())>math.radians(2.5):
                magnitude=0.20 if remaining>0.16 else 0.13
            else:
                magnitude=min(0.16,max(0.08,remaining*0.45))
            signed_target=direction*magnitude
            throttle=float(np.clip(2.0*(signed_target-self.car.speed),-0.6,0.6))
        return np.asarray([throttle,steering],dtype=np.float32)

    def _check_collision(self)->tuple[bool,bool]:
        poly=self._poly(self.car)
        if not self._inside_arena(poly): return False,True
        for obstacle in self.obstacles:
            if rectangles_intersect(poly,self._poly(obstacle)): return True,False
        return False,False

    def _is_success(self)->bool:
        longitudinal,lateral=self._local_error()
        return bool(abs(longitudinal)<0.43 and abs(lateral)<0.31 and
                    abs(self._angle_error())<math.radians(7.0) and abs(self.car.speed)<0.16)

    def reset(self,*,seed:int|None=None,options:dict[str,Any]|None=None):
        super().reset(seed=seed)
        options=options or {}
        if self.requested_mode in ("mixed","all"):
            self.mode=MODES[int(self.np_random.integers(0,len(MODES)))]
        else:
            self.mode=self.requested_mode
        if "mode" in options:
            if options["mode"] not in MODES: raise ValueError(f"Unknown mode: {options['mode']}")
            self.mode=str(options["mode"])
        reference_x=float(self.np_random.uniform(3.0,self.ARENA_WIDTH-3.0))
        self._make_map(reference_x)
        self.step_count=0
        self.last_action=np.zeros(2,dtype=np.float32)
        self.previous_potential=self._potential()
        return self._observation(),self._info(False,False,False,False)

    def step(self,action:np.ndarray):
        action=np.asarray(action,dtype=np.float32).reshape(2)
        action=np.clip(action,self.action_space.low,self.action_space.high)
        self.last_action=action.copy()
        self.car=step_bicycle(self.car,action,self.DT,self.vehicle)
        self.step_count+=1
        collision,outside=self._check_collision()
        success=self._is_success()
        current_potential=self._potential()
        reward=3.5*(self.previous_potential-current_potential)-0.012
        reward-=0.006*abs(self.car.speed)
        if self._target_distance()<1.0:
            reward+=0.035*(1.0-abs(self.car.speed)/self.vehicle.max_speed)
        terminated=False; truncated=False
        if collision:
            reward-=180.0; terminated=True
        elif outside:
            reward-=150.0; terminated=True
        elif success:
            reward+=220.0; terminated=True
        elif self.step_count>=self.max_steps:
            reward-=45.0; truncated=True
        self.previous_potential=current_potential
        info=self._info(success,collision,outside,truncated)
        return self._observation(),float(reward),terminated,truncated,info

    def _info(self,success:bool,collision:bool,outside:bool,timeout:bool)->dict[str,Any]:
        return {
            "success":bool(success),"collision":bool(collision),"outside_bounds":bool(outside),"timeout":bool(timeout),
            "mode":self.mode,"target_slot":self.target_slot_id,
            "free_slots":int(sum(not r["occupied"] for r in self.slot_records)),
            "feasible_slots":int(sum(r["feasible"] for r in self.slot_records)),
            "occupied_slots":int(len(self.obstacles)),"distance":float(self._target_distance()),
            "angle_error_deg":float(math.degrees(abs(self._angle_error()))),
            "speed":float(self.car.speed),"steps":int(self.step_count),
            "lidar_min_m":float(min(self.last_lidar)) if self.last_lidar else self.MAX_SENSOR_RANGE,
            "camera_detected_occupied":int(self.camera_occupancy.sum()),
            "ground_truth_occupied":int(len(self.obstacles)),
            "camera_occupancy_match":bool(np.array_equal(self.camera_occupancy,self.bay_occupancy)),
        }

    def close(self):
        return None
