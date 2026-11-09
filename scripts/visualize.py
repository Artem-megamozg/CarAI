"""Interactive Pygame viewer: PPO/expert, camera and sensor overlays, mode switching."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
import pygame
from stable_baselines3 import PPO
from autopark.environment import ParkingEnv,MODES
from autopark.simulation.renderer import ParkingRenderer


def main():
    p=argparse.ArgumentParser(description="AutoPark v0.4 visualization")
    p.add_argument("--model",default="models/best_model.zip")
    p.add_argument("--expert",action="store_true")
    p.add_argument("--mode",choices=("mixed",*MODES),default="mixed")
    p.add_argument("--difficulty",type=float,default=.55)
    p.add_argument("--fps",type=int,default=35)
    p.add_argument("--stochastic",action="store_true")
    args=p.parse_args()
    model=None
    if not args.expert:
        path=Path(args.model); path=path if path.is_absolute() else ROOT/path
        if not path.exists(): raise FileNotFoundError(f"Model missing: {path}")
        model=PPO.load(str(path),device="auto")
    env=ParkingEnv(mode=args.mode,difficulty=args.difficulty); renderer=ParkingRenderer(); clock=pygame.time.Clock()
    obs,_=env.reset(); episode=1; trajectory=[(env.car.x,env.car.y)]; total_reward=0.0; status="READY"; done=False; paused=False; terminal=0
    running=True; deterministic=not args.stochastic
    print("Pygame controls: SPACE pause | N next episode | R reset | 1 perpendicular | 2 reverse | 3 parallel | 4 tight | ESC quit")
    while running:
        request_reset=False
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False
            elif ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_ESCAPE: running=False
                elif ev.key==pygame.K_SPACE: paused=not paused
                elif ev.key in (pygame.K_n,pygame.K_r): request_reset=True
                elif ev.key in (pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4):
                    env.requested_mode=MODES[ev.key-pygame.K_1]; request_reset=True
        if request_reset:
            episode+=1; obs,_=env.reset(); trajectory=[(env.car.x,env.car.y)]; total_reward=0.0; status="NEW SCENARIO"; done=False; terminal=0
        if running and not paused and not done:
            action=env.expert_action() if args.expert else model.predict(obs,deterministic=deterministic)[0]
            obs,reward,terminated,truncated,info=env.step(action); total_reward+=float(reward)
            trajectory.append((env.car.x,env.car.y)); trajectory=trajectory[-1200:]
            if terminated or truncated:
                done=True; terminal=0
                status=(f"SUCCESS | {info['target_slot']} | {info['steps']} steps" if info["success"] else
                        "COLLISION" if info["collision"] else "OUTSIDE BOUNDS" if info["outside_bounds"] else "TIMEOUT")
                print(f"Episode {episode}: {env.mode} | {status} | reward={total_reward:.1f} | dist={info['distance']:.2f}m | angle={info['angle_error_deg']:.1f}°")
        elif done and not paused:
            terminal+=1
            if terminal>=args.fps*2:
                episode+=1; obs,_=env.reset(); trajectory=[(env.car.x,env.car.y)]; total_reward=0.0; status="NEW SCENARIO"; done=False; terminal=0
        renderer.draw(env,episode,total_reward,status,trajectory,paused,args.expert); clock.tick(max(10,args.fps))
    env.close(); pygame.quit()

if __name__=="__main__": main()
