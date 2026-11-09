"""Pygame visualizer for vehicle, parking modes, camera and sensor overlays."""
from __future__ import annotations
import math
import pygame
from PIL import Image
from autopark.physics import rectangle_corners
from autopark.vision import render_camera

class ParkingRenderer:
    WIDTH,HEIGHT=1500,980
    LEFT,TOP,SCALE=55,65,42
    BG=(14,19,26); ASPHALT=(34,43,54); ROAD=(48,58,70); WHITE=(239,244,249)
    MUTED=(153,167,182); GREEN=(72,207,151); RED=(225,91,98); YELLOW=(255,193,77); BLUE=(75,145,225)
    def __init__(self):
        pygame.init(); self.screen=pygame.display.set_mode((self.WIDTH,self.HEIGHT))
        pygame.display.set_caption("AutoPark RL v0.4 | Four-mode autonomous parking")
        self.font=pygame.font.SysFont("segoeui",20); self.small=pygame.font.SysFont("segoeui",15)
        self.tiny=pygame.font.SysFont("segoeui",12); self.large=pygame.font.SysFont("segoeui",32,bold=True)
    def world(self,x,y): return int(self.LEFT+x*self.SCALE),int(self.TOP+y*self.SCALE)
    def text(self,s,x,y,color=None,font=None):
        self.screen.blit((font or self.font).render(str(s),True,color or self.WHITE),(x,y))
    def draw_car(self,state,length,width,color):
        pts=[self.world(float(x),float(y)) for x,y in rectangle_corners(state.x,state.y,state.heading,length,width)]
        pygame.draw.polygon(self.screen,color,pts); pygame.draw.polygon(self.screen,(8,13,19),pts,2)
        c,s=math.cos(state.heading),math.sin(state.heading)
        def lp(lx,ly): return self.world(state.x+lx*c-ly*s,state.y+lx*s+ly*c)
        glass=[lp(length*.1,-width*.34),lp(length*.34,-width*.34),lp(length*.34,width*.34),lp(length*.1,width*.34)]
        pygame.draw.polygon(self.screen,(34,55,73),glass)
    def draw(self,env,episode,reward,status,trajectory,paused=False,expert=False):
        self.screen.fill(self.BG)
        self.text("AUTOPARK RL  /  v0.4",45,18,self.WHITE,self.large)
        self.text(f"MODE  {env.mode.upper()}   |   TARGET  {env.target_slot_id}   |   {'EXPERT BASELINE' if expert else 'PPO POLICY'}",48,48,self.MUTED,self.small)
        arena=pygame.Rect(self.LEFT,self.TOP,int(env.ARENA_WIDTH*self.SCALE),int(env.ARENA_HEIGHT*self.SCALE))
        pygame.draw.rect(self.screen,self.ASPHALT,arena,border_radius=8)
        lane_top=self.world(0,env.LANE_Y-2.15)[1]; lane_bot=self.world(0,env.LANE_Y+2.15)[1]
        pygame.draw.rect(self.screen,self.ROAD,(arena.left,lane_top,arena.width,lane_bot-lane_top))
        cy=self.world(0,env.LANE_Y)[1]
        for x in range(arena.left+10,arena.right-10,48): pygame.draw.line(self.screen,(110,123,137),(x,cy),(x+22,cy),2)
        for rec in env.slot_records:
            poly=rectangle_corners(rec["x"],rec["y"],rec["heading"],rec["length"],rec["width"])
            pts=[self.world(float(x),float(y)) for x,y in poly]
            if rec["occupied"]: fill=(46,68,93); edge=self.BLUE
            elif rec["selected"]: fill=(86,79,40); edge=self.YELLOW
            elif rec["feasible"]: fill=(35,73,62); edge=self.GREEN
            else: fill=(68,48,54); edge=self.RED
            pygame.draw.polygon(self.screen,fill,pts); pygame.draw.polygon(self.screen,edge,pts,2)
            p=self.world(rec["x"],rec["y"]); self.text(rec["id"],p[0]-10,p[1]-8,self.WHITE,self.tiny)
        # Expert route / agent trajectory.
        for i in range(1,len(env.path),5):
            p0=self.world(*env.path[i-1]); p1=self.world(*env.path[i]); pygame.draw.line(self.screen,(82,143,160),p0,p1,1)
        pts=[self.world(x,y) for x,y in trajectory]
        if len(pts)>1: pygame.draw.lines(self.screen,self.YELLOW,False,pts,3)
        for obstacle in env.obstacles: self.draw_car(obstacle,env.vehicle.length,env.vehicle.width,self.BLUE)
        self.draw_car(env.car,env.vehicle.length,env.vehicle.width,(244,185,64))
        # LiDAR rays (16) from front/side sensor origin.
        for i,rng in enumerate(env.last_lidar):
            angle=env.car.heading+2*math.pi*i/env.LIDAR_COUNT
            end=self.world(env.car.x+rng*math.cos(angle),env.car.y+rng*math.sin(angle))
            start=self.world(env.car.x,env.car.y)
            pygame.draw.line(self.screen,(54,190,190),start,end,1)
            pygame.draw.circle(self.screen,(87,221,210),end,3)
        # Side dashboard.
        panel_x=arena.right+35
        pygame.draw.rect(self.screen,(24,31,42),(panel_x,65,360,arena.height+80),border_radius=10)
        self.text("VEHICLE TELEMETRY",panel_x+18,82,self.WHITE,self.font)
        self.text(f"Speed: {env.car.speed:+.2f} m/s",panel_x+18,122)
        self.text(f"Goal distance: {env._target_distance():.2f} m",panel_x+18,154)
        self.text(f"Angle error: {math.degrees(abs(env._angle_error())):.1f} deg",panel_x+18,186)
        self.text(f"Steps: {env.step_count}/{env.max_steps}",panel_x+18,218)
        self.text(f"Free / feasible: {sum(not r['occupied'] for r in env.slot_records)} / {sum(r['feasible'] for r in env.slot_records)}",panel_x+18,250,self.GREEN)
        self.text("ULTRASONIC (m)",panel_x+18,296,self.WHITE,self.small)
        labels=["FRONT","REAR","LEFT","RIGHT"]
        for i,(label,value) in enumerate(zip(labels,env.last_sonar)):
            self.text(f"{label:>6}: {value:4.1f}",panel_x+18,322+i*25,(88,215,205),self.small)
        self.text("LIDAR 16-RAY",panel_x+18,432,self.WHITE,self.small)
        for i in range(4):
            vals=env.last_lidar[i*4:(i+1)*4]
            self.text("  ".join(f"{v:.1f}" for v in vals),panel_x+18,456+i*22,(88,215,205),self.tiny)
        self.text("CAMERA / PERCEPTION",panel_x+18,558,self.WHITE,self.small)
        frame=render_camera(env,288,216)
        camera_surface=pygame.image.fromstring(frame.tobytes(),(288,216),"RGB")
        camera_surface=pygame.transform.scale(camera_surface,(324,243))
        self.screen.blit(camera_surface,(panel_x+18,584))
        self.text(f"Camera occupied estimate: {int(env.camera_occupancy.sum())}",panel_x+18,840,self.GREEN,self.small)
        status_color=self.GREEN if "SUCCESS" in status else self.RED if any(k in status for k in ("COLLISION","OUTSIDE")) else self.YELLOW
        self.text(status,48,arena.bottom+20,status_color,self.font)
        self.text("Keys: SPACE pause | N next | R reset | 1-4 mode | ESC quit",48,self.HEIGHT-28,self.MUTED,self.small)
        if paused:
            overlay=pygame.Surface((self.WIDTH,self.HEIGHT),pygame.SRCALPHA); overlay.fill((0,0,0,150)); self.screen.blit(overlay,(0,0))
            surf=self.large.render("PAUSED",True,self.WHITE); self.screen.blit(surf,surf.get_rect(center=(self.WIDTH//2,self.HEIGHT//2)))
        pygame.display.flip()
