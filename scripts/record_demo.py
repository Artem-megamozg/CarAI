"""Record one GIF per parking mode (PPO model or expert baseline)."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
import imageio.v2 as imageio
import numpy as np
import pygame
from stable_baselines3 import PPO
from autopark.environment import ParkingEnv,MODES
from autopark.simulation.renderer import ParkingRenderer


def record(mode,model,expert,difficulty,fps,output):
    env=ParkingEnv(mode=mode,difficulty=difficulty); renderer=ParkingRenderer()
    obs,_=env.reset(seed=44000+MODES.index(mode)); trajectory=[(env.car.x,env.car.y)]
    reward_sum=0.0; done=False; status="RUNNING"; frames=[]; max_steps=env.max_steps
    for step in range(max_steps):
        action=env.expert_action() if expert else model.predict(obs,deterministic=True)[0]
        obs,r,terminated,truncated,info=env.step(action); reward_sum+=float(r)
        trajectory.append((env.car.x,env.car.y))
        if terminated or truncated:
            done=True
            status=(f"SUCCESS | {info['steps']} steps" if info['success'] else
                    "COLLISION" if info['collision'] else "OUTSIDE BOUNDS" if info['outside_bounds'] else "TIMEOUT")
        if step%2==0 or done:
            renderer.draw(env,1,reward_sum,status,trajectory,False,expert)
            surf=pygame.display.get_surface()
            frame=pygame.surfarray.array3d(surf).transpose(1,0,2)
            # Downsample for a manageable GIF while retaining readable telemetry.
            im=pygame.surfarray.make_surface(frame[::2,::2].transpose(1,0,2))
            frames.append(pygame.surfarray.array3d(im).transpose(1,0,2))
        if done: break
    if not done:
        status="TIMEOUT"
    for _ in range(max(1,fps//2)):
        renderer.draw(env,1,reward_sum,status,trajectory,False,expert)
        surf=pygame.display.get_surface(); frame=pygame.surfarray.array3d(surf).transpose(1,0,2)
        im=pygame.surfarray.make_surface(frame[::2,::2].transpose(1,0,2)); frames.append(pygame.surfarray.array3d(im).transpose(1,0,2))
    output.parent.mkdir(parents=True,exist_ok=True)
    imageio.mimsave(output,frames,duration=1/max(1,fps//2),loop=0)
    print(f"{mode}: {status}; saved {output}")
    env.close()


def main():
    p=argparse.ArgumentParser(description="Create AutoPark portfolio GIFs")
    p.add_argument("--model",default="models/best_model.zip"); p.add_argument("--expert",action="store_true")
    p.add_argument("--mode",choices=("all",*MODES),default="all"); p.add_argument("--difficulty",type=float,default=.55)
    p.add_argument("--fps",type=int,default=12); p.add_argument("--output",default="reports/figures")
    args=p.parse_args(); model=None
    if not args.expert:
        path=Path(args.model); path=path if path.is_absolute() else ROOT/path
        if not path.exists(): raise FileNotFoundError(path)
        model=PPO.load(str(path),device="auto")
    pygame.init(); modes=MODES if args.mode=="all" else (args.mode,)
    try:
        for mode in modes:
            record(mode,model,args.expert,args.difficulty,args.fps,ROOT/args.output/f"autopark_{mode}.gif")
    finally: pygame.quit()

if __name__=="__main__": main()
