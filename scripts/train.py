"""Behavior Cloning -> PPO training for all four AutoPark modes (PPO only)."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.monitor import Monitor
from autopark.environment import ParkingEnv,MODES
from autopark.imitation import (collect_expert_demonstrations,train_behavior_cloning,
                                initialize_ppo_actor,save_demonstrations)

class CurriculumCallback(BaseCallback):
    def __init__(self,env:ParkingEnv,total_timesteps:int):
        super().__init__(verbose=1); self.env=env; self.total=max(1,total_timesteps); self.last=-1.0
    def _on_step(self):
        progress=min(1.0,self.num_timesteps/self.total)
        # More empty space at the beginning; progressively denser lots later.
        difficulty=0.08+0.54*progress
        if abs(difficulty-self.last)>=0.02:
            self.env.set_difficulty(difficulty); self.last=difficulty
            print(f"\nCurriculum difficulty -> {difficulty:.2f}")
        return True

class SuccessAwareEvalCallback(BaseCallback):
    def __init__(self,eval_freq:int,episodes_per_mode:int,best_path:Path):
        super().__init__(verbose=1); self.eval_freq=max(1,eval_freq); self.n_ep=max(1,episodes_per_mode)
        self.envs={m:ParkingEnv(mode=m,difficulty=.62) for m in MODES}; self.best_path=best_path
        self.best_key=(-1.0,-1.0,0.0,-1.0); self.log_path=ROOT/"reports"/"training"/"success_evaluations.jsonl"
        metrics_file=ROOT/"reports"/"training"/"best_model_metrics.json"
        if metrics_file.exists():
            try:
                previous=json.loads(metrics_file.read_text(encoding="utf-8"))
                per=previous.get("per_mode",{})
                worst=min((float(v.get("success_rate",0.0)) for v in per.values()),default=float(previous.get("success_rate",0.0)))
                collision=float(previous.get("collision_rate",np.mean([v.get("collision_rate",0.0) for v in per.values()]) if per else 0.0))
                outside=float(previous.get("outside_bounds_rate",np.mean([v.get("outside_bounds_rate",0.0) for v in per.values()]) if per else 0.0))
                self.best_key=(float(previous.get("success_rate",0.0)),worst,-collision-outside,float(previous.get("mean_reward",0.0)))
            except (ValueError,TypeError,json.JSONDecodeError):
                pass
    def evaluate(self):
        per_mode={}; all_rewards=[]; all_steps=[]; total_success=total_collision=total_outside=total_timeout=0; count=0
        for mode,env in self.envs.items():
            success=collision=outside=timeout=0; rewards=[]; steps=[]; distances=[]; angles=[]
            for ep in range(self.n_ep):
                obs,_=env.reset(seed=200000+self.num_timesteps+MODE_SEED[mode]*1000+ep)
                terminated=truncated=False; total_reward=0.0; info={}
                while not (terminated or truncated):
                    action,_=self.model.predict(obs,deterministic=True)
                    obs,r,terminated,truncated,info=env.step(action); total_reward+=float(r)
                success+=int(info["success"]); collision+=int(info["collision"]); outside+=int(info["outside_bounds"]); timeout+=int(info["timeout"])
                rewards.append(total_reward); steps.append(info["steps"]); distances.append(info["distance"]); angles.append(info["angle_error_deg"])
            denom=self.n_ep
            per_mode[mode]={"episodes":denom,"success_rate":success/denom,"collision_rate":collision/denom,
                            "outside_bounds_rate":outside/denom,"timeout_rate":timeout/denom,
                            "mean_reward":float(np.mean(rewards)),"mean_steps":float(np.mean(steps)),
                            "mean_distance_m":float(np.mean(distances)),"mean_angle_error_deg":float(np.mean(angles))}
            total_success+=success; total_collision+=collision; total_outside+=outside; total_timeout+=timeout; count+=denom
            all_rewards+=rewards; all_steps+=steps
        result={"timesteps":int(self.num_timesteps),"episodes":count,
                "success_rate":total_success/count,"collision_rate":total_collision/count,
                "outside_bounds_rate":total_outside/count,"timeout_rate":total_timeout/count,
                "mean_reward":float(np.mean(all_rewards)),"mean_steps":float(np.mean(all_steps)),
                "worst_mode_success_rate":min(v["success_rate"] for v in per_mode.values()),"per_mode":per_mode}
        return result
    def _on_step(self):
        if self.n_calls%self.eval_freq: return True
        result=self.evaluate(); key=(result["success_rate"],result["worst_mode_success_rate"],
                                     -result["collision_rate"]-result["outside_bounds_rate"],result["mean_reward"])
        improved=key>self.best_key
        if improved:
            self.best_key=key; self.best_path.parent.mkdir(parents=True,exist_ok=True); self.model.save(str(self.best_path))
            (ROOT/"reports"/"training"/"best_model_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        with self.log_path.open("a",encoding="utf-8") as f: f.write(json.dumps(result,ensure_ascii=False)+"\n")
        print(f"\nEval @ {self.num_timesteps:,}: success={result['success_rate']:.1%}, worst-mode={result['worst_mode_success_rate']:.1%}, "
              f"collision={result['collision_rate']:.1%}, outside={result['outside_bounds_rate']:.1%}"+(" | NEW BEST" if improved else ""))
        for mode,metrics in result["per_mode"].items(): print(f"  {mode:14s}: {metrics['success_rate']:.1%} success / {metrics['collision_rate']:.1%} collision")
        return True
    def close(self):
        for env in self.envs.values(): env.close()

MODE_SEED={m:i+1 for i,m in enumerate(MODES)}

class CheckpointCallback(BaseCallback):
    def __init__(self,directory:Path,frequency:int):
        super().__init__(verbose=1); self.directory=directory; self.frequency=max(1,frequency)
    def _on_step(self):
        if self.n_calls%self.frequency==0:
            self.directory.mkdir(parents=True,exist_ok=True); path=self.directory/f"ppo_step_{self.num_timesteps}.zip"
            self.model.save(str(path)); print(f"\nCheckpoint saved: {path}")
        return True


def evaluate_initial_policy(model,episodes:int=12):
    results={}
    for mode in MODES:
        env=ParkingEnv(mode=mode,difficulty=.62); wins=coll=out=0
        for ep in range(episodes):
            obs,_=env.reset(seed=300000+MODE_SEED[mode]*100+ep); term=trunc=False; info={}
            while not(term or trunc):
                action,_=model.predict(obs,deterministic=True); obs,_,term,trunc,info=env.step(action)
            wins+=int(info["success"]); coll+=int(info["collision"]); out+=int(info["outside_bounds"])
        results[mode]={"success_rate":wins/episodes,"collision_rate":coll/episodes,"outside_bounds_rate":out/episodes}
        env.close()
    return results


def parse_args():
    p=argparse.ArgumentParser(description="AutoPark v0.4: multi-mode expert BC + PPO")
    p.add_argument("--timesteps",type=int,default=3_000_000)
    p.add_argument("--demo-episodes",type=int,default=1200)
    p.add_argument("--bc-epochs",type=int,default=30)
    p.add_argument("--eval-freq",type=int,default=50_000)
    p.add_argument("--eval-episodes",type=int,default=12,help="Evaluation episodes per parking mode")
    p.add_argument("--checkpoint-freq",type=int,default=250_000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=("cpu","cuda","auto"),default="cpu")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--skip-pretraining",action="store_true")
    return p.parse_args()


def main():
    args=parse_args()
    if min(args.timesteps,args.eval_freq,args.eval_episodes,args.checkpoint_freq)<1: raise SystemExit("All training counts must be positive")
    if args.demo_episodes<1 and not(args.skip_pretraining or args.resume): raise SystemExit("--demo-episodes must be positive")
    torch.set_num_threads(1); np.random.seed(args.seed); torch.manual_seed(args.seed)
    models=ROOT/"models"; reports=ROOT/"reports"/"training"
    models.mkdir(exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    best_path=models/"best_model.zip"; final_path=models/"final_model.zip"
    base_env=ParkingEnv(mode="mixed",difficulty=.08)
    train_env=Monitor(base_env)
    policy_kwargs={"net_arch":{"pi":[256,256,128],"vf":[256,256,128]},"activation_fn":torch.nn.Tanh}
    bc_stats=None; bc_metrics=None
    if args.resume:
        if not final_path.exists(): raise FileNotFoundError(f"Missing {final_path}")
        model=PPO.load(str(final_path),env=train_env,device=args.device)
        print(f"Resuming PPO from {final_path}")
    else:
        actor=None
        if not args.skip_pretraining:
            print("\n[1/4] Collecting balanced expert demonstrations for four modes...")
            demo_env=ParkingEnv(mode="mixed",difficulty=.25)
            obs,acts,bc_stats=collect_expert_demonstrations(demo_env,args.demo_episodes,args.seed)
            demo_env.close()
            save_demonstrations(reports/"expert_demonstrations.npz",obs,acts)
            (reports/"expert_demonstrations.json").write_text(json.dumps(bc_stats,indent=2),encoding="utf-8")
            print(f"Expert success: {bc_stats['expert_success_rate']:.1%}; transitions={len(obs):,}")
            weak=[m for m,v in bc_stats["per_mode"].items() if v["success_rate"]<.85]
            if bc_stats["expert_success_rate"]<.90 or weak:
                raise RuntimeError(f"Expert demonstrations are too weak; per-mode failures: {weak}. Refusing to start RL.")
            print("\n[2/4] Behavior Cloning pretraining...")
            actor,bc_metrics=train_behavior_cloning(obs,acts,args.bc_epochs,device=("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu"),seed=args.seed)
            torch.save(actor.state_dict(),models/"bc_actor.pt")
            (reports/"behavior_cloning.json").write_text(json.dumps({**bc_stats,**bc_metrics},indent=2),encoding="utf-8")
        print("\n[3/4] Initializing PPO policy...")
        model=PPO("MlpPolicy",train_env,learning_rate=1.0e-4,n_steps=2048,batch_size=512,n_epochs=6,
                  gamma=.995,gae_lambda=.95,clip_range=.15,ent_coef=.002,vf_coef=.5,max_grad_norm=.5,
                  policy_kwargs=policy_kwargs,tensorboard_log=str(ROOT/"reports"/"tensorboard"),verbose=1,
                  seed=args.seed,device=args.device)
        if actor is not None: initialize_ppo_actor(model,actor); print("BC actor weights copied to PPO policy.")
        print("\n[4/4] Measuring BC-initialized policy on all four modes...")
        baseline=evaluate_initial_policy(model,episodes=20)
        (reports/"bc_baseline_by_mode.json").write_text(json.dumps(baseline,indent=2),encoding="utf-8")
        print(json.dumps(baseline,indent=2))
        # Always preserve the initialized baseline as the first candidate.
        model.save(str(best_path))
        avg=float(np.mean([v["success_rate"] for v in baseline.values()]))
        (reports/"best_model_metrics.json").write_text(json.dumps({"timesteps":0,"success_rate":avg,"per_mode":baseline},indent=2),encoding="utf-8")
    eval_cb=SuccessAwareEvalCallback(args.eval_freq,args.eval_episodes,best_path)
    callbacks=CallbackList([eval_cb,CurriculumCallback(base_env,args.timesteps),CheckpointCallback(models/"checkpoints",args.checkpoint_freq)])
    metadata={"version":"0.4.0","algorithm":"PPO","observation_size":ParkingEnv.OBSERVATION_SIZE,
              "action_size":2,"modes":list(MODES),"sensor_suite":{"lidar_rays":16,"ultrasonic":4,"camera":"semantic RGB + bay occupancy segmentation"},
              "behavior_cloning":not args.skip_pretraining,"expert_demo_episodes":args.demo_episodes if not args.skip_pretraining else 0,
              "requested_rl_timesteps":args.timesteps,"seed":args.seed}
    (models/"model_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("\n"+"="*78); print("AUTOPARK RL v0.4 | BC + PPO | FOUR PARKING MODES")
    print(f"RL timesteps       : {args.timesteps:,}"); print(f"Expert demos       : {metadata['expert_demo_episodes']:,}")
    print(f"Modes              : {', '.join(MODES)}"); print("Sensors            : 16-ray LiDAR + 4 ultrasonic + RGB camera perception")
    print(f"Evaluation         : every {args.eval_freq:,} steps / {args.eval_episodes} episodes per mode")
    print("Model selection    : aggregate success -> worst-mode success -> safety -> reward")
    print(f"Device             : {args.device}"); print("="*78)
    try:
        model.learn(total_timesteps=args.timesteps,callback=callbacks,progress_bar=True,
                    reset_num_timesteps=not args.resume,tb_log_name="autopark_v04_ppo_multimode")
    except KeyboardInterrupt:
        print("\nTraining interrupted; saving the latest policy and checkpoints.")
    finally:
        model.save(str(final_path)); base_env.close(); eval_cb.close()
        print(f"\nFinal model: {final_path}\nBest model : {best_path}\nMetadata   : {models/'model_metadata.json'}")

if __name__=="__main__": main()
