"""Evaluate PPO or the expert baseline on each parking mode with reports."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from autopark.environment import ParkingEnv,MODES


def main():
    p=argparse.ArgumentParser(description="AutoPark v0.4 four-mode evaluation")
    p.add_argument("--model",default="models/best_model.zip")
    p.add_argument("--expert",action="store_true")
    p.add_argument("--episodes",type=int,default=200,help="Episodes per mode when --mode all")
    p.add_argument("--mode",choices=("all",*MODES),default="all")
    p.add_argument("--difficulty",type=float,default=.62)
    p.add_argument("--seed",type=int,default=2026)
    args=p.parse_args()
    if args.episodes<1: raise SystemExit("--episodes must be positive")
    model=None; model_label="expert_controller"
    if not args.expert:
        path=Path(args.model); path=path if path.is_absolute() else ROOT/path
        if not path.exists(): raise FileNotFoundError(f"Model not found: {path}. Train first or use --expert.")
        model=PPO.load(str(path),device="auto"); model_label=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    modes=MODES if args.mode=="all" else (args.mode,)
    per_mode={}; all_rows=[]
    for mode_idx,mode in enumerate(modes):
        env=ParkingEnv(mode=mode,difficulty=args.difficulty); rows=[]; slot_counts=Counter()
        for ep in range(args.episodes):
            obs,_=env.reset(seed=args.seed+mode_idx*100000+ep); terminated=truncated=False; total_reward=0.0; info={}
            while not(terminated or truncated):
                action=env.expert_action() if args.expert else model.predict(obs,deterministic=True)[0]
                obs,reward,terminated,truncated,info=env.step(action); total_reward+=float(reward)
            slot_counts[str(info["target_slot"])]+=1
            row={"mode":mode,"episode":ep+1,"target_slot":info["target_slot"],"success":info["success"],
                 "collision":info["collision"],"outside_bounds":info["outside_bounds"],"timeout":info["timeout"],
                 "steps":info["steps"],"distance_m":info["distance"],"angle_error_deg":info["angle_error_deg"],
                 "reward":total_reward,"camera_occupied_detected":info["camera_detected_occupied"],
                 "ground_truth_occupied":info["ground_truth_occupied"],"camera_occupancy_match":info["camera_occupancy_match"]}
            rows.append(row); all_rows.append(row)
            if (ep+1)%max(1,min(50,args.episodes//4))==0 or ep+1==args.episodes:
                print(f"{mode}: evaluated {ep+1}/{args.episodes}")
        # Mutually-exclusive outcomes; rates must sum to 100% (within FP tolerance).
        per_mode[mode]={"episodes":len(rows),"success_rate":float(np.mean([r["success"] for r in rows])),
                        "collision_rate":float(np.mean([r["collision"] for r in rows])),
                        "outside_bounds_rate":float(np.mean([r["outside_bounds"] for r in rows])),
                        "timeout_rate":float(np.mean([r["timeout"] for r in rows])),
                        "mean_steps":float(np.mean([r["steps"] for r in rows])),
                        "mean_final_distance_m":float(np.mean([r["distance_m"] for r in rows])),
                        "mean_final_angle_error_deg":float(np.mean([r["angle_error_deg"] for r in rows])),
                        "mean_reward":float(np.mean([r["reward"] for r in rows])),
                        "camera_occupancy_match_rate":float(np.mean([r["camera_occupancy_match"] for r in rows])),
                        "selected_bay_counts":dict(slot_counts)}
        env.close()
        print(f"\n{mode.upper()}: success={per_mode[mode]['success_rate']:.1%}; collision={per_mode[mode]['collision_rate']:.1%}; "
              f"outside={per_mode[mode]['outside_bounds_rate']:.1%}; timeout={per_mode[mode]['timeout_rate']:.1%}")
    total=len(all_rows)
    overall={key:float(np.mean([r[field] for r in all_rows])) for key,field in (
        ("success_rate","success"),("collision_rate","collision"),("outside_bounds_rate","outside_bounds"),("timeout_rate","timeout"),
        ("mean_steps","steps"),("mean_final_distance_m","distance_m"),("mean_final_angle_error_deg","angle_error_deg"),("mean_reward","reward"))}
    overall.update({"model":model_label,"mode":args.mode,"episodes":total,"episodes_per_mode":args.episodes if args.mode=="all" else None,
                    "difficulty":args.difficulty,"per_mode":per_mode})
    out=ROOT/"reports"/"evaluation"; fig_dir=ROOT/"reports"/"figures"; out.mkdir(parents=True,exist_ok=True); fig_dir.mkdir(parents=True,exist_ok=True)
    csv_path=out/("expert_episodes.csv" if args.expert else "episodes.csv")
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0].keys())); writer.writeheader(); writer.writerows(all_rows)
    (out/"summary_by_mode.json").write_text(json.dumps(overall,indent=2),encoding="utf-8")
    (out/("expert_summary.json" if args.expert else "summary.json")).write_text(json.dumps(overall,indent=2),encoding="utf-8")
    labels=list(per_mode)
    if labels:
        series=[("Success","success_rate"),("Collision","collision_rate"),("Outside","outside_bounds_rate"),("Timeout","timeout_rate")]
        fig,ax=plt.subplots(figsize=(10,5.5)); x=np.arange(len(labels)); width=.18
        for i,(name,key) in enumerate(series):
            vals=[per_mode[m][key]*100 for m in labels]
            bars=ax.bar(x+(i-1.5)*width,vals,width,label=name)
            for b,v in zip(bars,vals):
                if v>=3: ax.text(b.get_x()+b.get_width()/2,v+1,f"{v:.0f}%",ha="center",va="bottom",fontsize=8)
        ax.set_xticks(x,labels); ax.set_ylim(0,110); ax.set_ylabel("Episodes (%)")
        ax.set_title(f"AutoPark v0.4 — {model_label} — {total} evaluation episodes")
        ax.grid(axis="y",alpha=.25); ax.legend(ncols=4); fig.tight_layout()
        chart=fig_dir/("expert_evaluation_v04.png" if args.expert else "evaluation_v04.png"); fig.savefig(chart,dpi=180); plt.close(fig)
    else: chart=None
    print("\nOverall evaluation summary")
    print(json.dumps(overall,indent=2))
    print(f"\nCSV: {csv_path}\nSummary: {out/'summary_by_mode.json'}\nChart: {chart}")

if __name__=="__main__": main()
