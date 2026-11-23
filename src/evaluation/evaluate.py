import argparse, csv, json
from pathlib import Path
import yaml, numpy as np
from src.environments.arm_env import ArmReachEnv
from src.environments.perturbations import Perturbation
from src.controllers.nominal_policy import load_policy
from src.controllers.classical_controller import PDController
from src.utils.metrics import summarize


def run_controller(env, controller, episodes, seed, perturbation, name):
    rows=[]
    for ep in range(episodes):
        obs,info=env.reset(seed=seed+ep, options={"perturbation":perturbation}); ret=0.; effort=0.; lim=0; steps=0
        for steps in range(1,env.episode_steps+1):
            action=controller.act(obs,deterministic=True) if hasattr(controller,"act") else controller(obs)
            obs,r,t,tr,inf=env.step(action); ret+=r; effort+=inf["control_effort"]; lim+=inf["joint_limit_violation"]
            if t or tr: break
        rows.append({"controller":name,"seed":seed+ep,"perturbation":json.dumps(perturbation.to_dict(),sort_keys=True),"episode_return":ret,"success":inf["success"],"endpoint_error":inf["endpoint_error"],"steps":steps,"control_effort":effort,"joint_limit_violations":lim})
    return rows


def evaluate(config_path, sweep=False):
    cfg=yaml.safe_load(open(config_path)); env=ArmReachEnv(episode_steps=cfg.get("episode_steps",200)); scenarios=[]
    if sweep:
        for key, vals in cfg["scenarios"].items():
            if key=="nominal":
                for d in vals: scenarios.append(("nominal",Perturbation(**d)))
            elif key in ("mass","friction","actuator_strength","delay_steps","sensor_noise","disturbance_force"):
                for v in vals:
                    p=Perturbation(); setattr(p,key,v); scenarios.append((key,Perturbation(**p.to_dict())))
        # Combinations withheld from randomized training: out-of-distribution test points.
        scenarios += [("combined_ood",Perturbation(mass=.55,friction=1.9,actuator_strength=.72,delay_steps=4,sensor_noise=.045,disturbance_force=1.6))]
    else: scenarios=[("nominal",Perturbation())]
    controllers=[("classical",PDController())]
    for nm,path in (("nominal",Path("results/checkpoints/nominal/policy.pt")),("randomized",Path("results/checkpoints/randomized/policy.pt"))):
        if path.exists(): controllers.append((nm,load_policy(path)))
    allrows=[]
    for category,p in scenarios:
        for name,controller in controllers:
            allrows += run_controller(env,controller,cfg.get("episodes",10),cfg["seed"],p,name)
    out=Path("results"); out.mkdir(exist_ok=True)
    with open(out/("robustness.csv" if sweep else "evaluation.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=allrows[0].keys()); w.writeheader(); w.writerows(allrows)
    summary={}
    for row in allrows: summary.setdefault((row["controller"],row["perturbation"]),[]).append(row)
    summ=[{"controller":k[0],"perturbation":json.loads(k[1]),**summarize(v)} for k,v in summary.items()]
    (out/("robustness_summary.json" if sweep else "evaluation_summary.json")).write_text(json.dumps(summ,indent=2))
    print(json.dumps(summ,indent=2)); env.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--sweep",action="store_true"); a=p.parse_args(); evaluate(a.config,a.sweep)
