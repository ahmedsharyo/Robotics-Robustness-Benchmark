"""Budgeted random search; reports empirical worst observed samples only."""
import argparse, csv, json
from pathlib import Path
import numpy as np, yaml
from src.environments.arm_env import ArmReachEnv
from src.environments.perturbations import Perturbation
from src.controllers.classical_controller import PDController
from src.controllers.nominal_policy import load_policy
from src.evaluation.evaluate import run_controller

def main(path, budget=24):
    cfg=yaml.safe_load(open(path)); rng=np.random.default_rng(cfg["seed"]); env=ArmReachEnv(episode_steps=cfg.get("episode_steps",200)); ctrls=[("classical",PDController())]
    for n,p in (("nominal","results/checkpoints/nominal/policy.pt"),("randomized","results/checkpoints/randomized/policy.pt")):
        if Path(p).exists(): ctrls.append((n,load_policy(p)))
    rows=[]
    bounds={"mass":(.5,1.5),"friction":(.5,2.),"actuator_strength":(.7,1.3),"delay_steps":(0,5),"sensor_noise":(0.,.06),"disturbance_force":(0.,2.)}
    for i in range(budget):
        d={k:(int(rng.integers(a,b+1)) if k=="delay_steps" else float(rng.uniform(a,b))) for k,(a,b) in bounds.items()}; p=Perturbation(**d)
        for n,c in ctrls: rows.extend(run_controller(env,c,3,cfg["seed"]+i*10,p,n))
    Path("results").mkdir(exist_ok=True)
    with open("results/worst_case_search.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    worst={}
    for r in rows:
        if r["controller"] not in worst or r["episode_return"]<worst[r["controller"]]["episode_return"]: worst[r["controller"]]=r
    Path("results/worst_case_summary.json").write_text(json.dumps({"method":"seeded random search","budget_scenarios":budget,"episodes_per_scenario":3,"empirical_worst_observed":worst},indent=2)); env.close()
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--budget",type=int,default=24); a=p.parse_args(); main(a.config,a.budget)
