"""Expert demonstration collection and behavior-cloning initialization."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from autopark.environment import ParkingEnv, MODES

HIDDEN_DIMS=(256,256,128)

class BehaviorCloningActor(nn.Module):
    def __init__(self,observation_size:int,action_size:int=2):
        super().__init__(); layers=[]; size=observation_size
        for width in HIDDEN_DIMS:
            layers.extend((nn.Linear(size,width),nn.Tanh())); size=width
        layers.append(nn.Linear(size,action_size)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


def collect_expert_demonstrations(env:ParkingEnv,episodes:int=1200,seed:int=42):
    if episodes<1: raise ValueError("episodes must be positive")
    observations=[]; actions=[]; successful=0; failed=0; steps=[]; mode_stats={m:{"success":0,"episodes":0} for m in MODES}
    for ep in range(episodes):
        # Balanced mode assignment prevents the largest scenario family dominating BC.
        mode=MODES[ep%len(MODES)]
        env.set_difficulty(float(np.random.default_rng(seed+ep).uniform(.08,.62)))
        obs,_=env.reset(seed=seed+ep,options={"mode":mode})
        ep_obs=[]; ep_actions=[]; terminated=truncated=False; info={}
        while not (terminated or truncated) and len(ep_actions)<env.max_steps:
            action=env.expert_action()
            ep_obs.append(np.asarray(obs,dtype=np.float32).copy())
            ep_actions.append(np.asarray(action,dtype=np.float32).copy())
            obs,_,terminated,truncated,info=env.step(action)
        mode_stats[mode]["episodes"]+=1
        if bool(info.get("success")):
            successful+=1; mode_stats[mode]["success"]+=1
            observations.extend(ep_obs); actions.extend(ep_actions); steps.append(len(ep_actions))
        else: failed+=1
        if (ep+1)%100==0 or ep+1==episodes:
            print(f"Expert demos {ep+1}/{episodes} | accepted={successful} ({successful/(ep+1):.1%}) | mode={mode}")
    if not observations: raise RuntimeError("Expert did not produce any successful demonstrations")
    obs_arr=np.asarray(observations,dtype=np.float32); act_arr=np.asarray(actions,dtype=np.float32)
    stats={"requested_episodes":episodes,"successful_episodes":successful,"failed_episodes":failed,
           "expert_success_rate":successful/episodes,"accepted_transitions":len(obs_arr),
           "mean_steps_successful":float(np.mean(steps)),
           "per_mode":{m:{"episodes":v["episodes"],"success_rate":v["success"]/max(1,v["episodes"])} for m,v in mode_stats.items()}}
    return obs_arr,act_arr,stats


def train_behavior_cloning(observations:np.ndarray,actions:np.ndarray,epochs:int=30,batch_size:int=512,
                            learning_rate:float=3e-4,device:str="cpu",seed:int=42):
    if len(observations)!=len(actions) or len(observations)<100: raise ValueError("Dataset too small or mismatched")
    torch.manual_seed(seed); rng=np.random.default_rng(seed); idx=rng.permutation(len(observations))
    split=max(1,int(.9*len(idx))); train_idx,val_idx=idx[:split],idx[split:]
    x=torch.as_tensor(observations,dtype=torch.float32); y=torch.as_tensor(actions,dtype=torch.float32)
    loader=DataLoader(TensorDataset(x[train_idx],y[train_idx]),batch_size=batch_size,shuffle=True)
    actor=BehaviorCloningActor(observations.shape[1]).to(device)
    opt=torch.optim.AdamW(actor.parameters(),lr=learning_rate,weight_decay=1e-5)
    loss_fn=nn.SmoothL1Loss(); best=float("inf"); best_state=None
    for epoch in range(epochs):
        actor.train(); train_losses=[]
        for bx,by in loader:
            bx,by=bx.to(device),by.to(device); pred=actor(bx); loss=loss_fn(pred,by)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(),1.0); opt.step()
            train_losses.append(float(loss.detach().cpu()))
        actor.eval()
        with torch.no_grad(): val=float(loss_fn(actor(x[val_idx].to(device)),y[val_idx].to(device)).cpu())
        if val<best:
            best=val; best_state={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        if epoch==0 or (epoch+1)%5==0 or epoch+1==epochs:
            print(f"BC epoch {epoch+1:02d}/{epochs} | train={np.mean(train_losses):.5f} | val={val:.5f}")
    if best_state is not None: actor.load_state_dict(best_state)
    return actor,{"best_validation_loss":best,"epochs":epochs}


def initialize_ppo_actor(model:Any,actor:BehaviorCloningActor)->None:
    """Copy BC MLP weights into Stable-Baselines3 PPO's policy mean network."""
    destination=[m for m in model.policy.mlp_extractor.policy_net.modules() if isinstance(m,nn.Linear)]
    destination += [m for m in model.policy.action_net.modules() if isinstance(m,nn.Linear)]
    source=[m for m in actor.net.modules() if isinstance(m,nn.Linear)]
    if len(destination)!=len(source): raise RuntimeError(f"Actor layer mismatch: {len(source)} != {len(destination)}")
    with torch.no_grad():
        for src,dst in zip(source,destination):
            if src.weight.shape!=dst.weight.shape: raise RuntimeError(f"BC/PPO shape mismatch {src.weight.shape} != {dst.weight.shape}")
            dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
        if hasattr(model.policy,"log_std"): model.policy.log_std.fill_(-1.4)


def save_demonstrations(path:str|Path,observations:np.ndarray,actions:np.ndarray):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,observations=observations,actions=actions)
