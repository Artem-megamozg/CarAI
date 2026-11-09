import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from autopark.environment import ParkingEnv,MODES


def run_expert(mode,seed):
    env=ParkingEnv(mode=mode,difficulty=.55)
    obs,_=env.reset(seed=seed)
    assert obs.shape==(ParkingEnv.OBSERVATION_SIZE,)
    terminated=truncated=False; info={}
    while not(terminated or truncated):
        obs,_,terminated,truncated,info=env.step(env.expert_action())
    env.close()
    return info


def test_all_modes_have_valid_observation_and_target():
    for i,mode in enumerate(MODES):
        env=ParkingEnv(mode=mode,difficulty=.55)
        obs,info=env.reset(seed=1234+i)
        assert obs.shape==(75,)
        assert np.all(np.isfinite(obs))
        assert info["mode"]==mode
        assert info["feasible_slots"]>=1
        assert any(r["selected"] and r["feasible"] and not r["occupied"] for r in env.slot_records)
        assert len(env.last_lidar)==16 and len(env.last_sonar)==4
        env.close()


def test_expert_can_complete_each_mode():
    for i,mode in enumerate(MODES):
        for ep in range(5):
            result=run_expert(mode,7000+i*100+ep)
            assert result["success"], (mode,result)
            assert not result["collision"] and not result["outside_bounds"]


def test_camera_occupancy_matches_simulated_parked_cars():
    for i,mode in enumerate(MODES):
        env=ParkingEnv(mode=mode,difficulty=.62)
        for ep in range(12):
            env.reset(seed=9000+i*100+ep)
            truth=np.zeros(12,dtype=np.float32)
            for j,record in enumerate(env.slot_records): truth[j]=float(record["occupied"])
            assert np.array_equal(truth,env.camera_occupancy)
        env.close()


def test_success_failure_categories_are_mutually_exclusive():
    for i,mode in enumerate(MODES):
        info=run_expert(mode,12000+i)
        assert sum(bool(info[k]) for k in ("success","collision","outside_bounds","timeout"))==1
