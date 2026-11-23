"""Compact PPO trainer; update cadence and total interactions are config-controlled."""
import argparse, csv, json, os, platform, subprocess
from pathlib import Path
import yaml, numpy as np, torch
from src.environments.arm_env import ArmReachEnv
from src.controllers.nominal_policy import ActorCritic
from src.utils.seeds import seed_everything


def train(config_path):
    cfg = yaml.safe_load(open(config_path)); seed=cfg["seed"]; seed_everything(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    envcfg=cfg["environment"]
    env=ArmReachEnv(**envcfg); model=ActorCritic().to(dev); opt=torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    tc=cfg["training"]; total=tc["total_timesteps"]; n=tc["rollout_steps"]; batch=tc["batch_size"]; epochs=tc["epochs"]
    out=Path(tc["checkpoint_dir"]); out.mkdir(parents=True, exist_ok=True); obs,_=env.reset(seed=seed); step=0; logs=[]
    while step < total:
        ob=[]; ac=[]; lp=[]; rw=[]; dn=[]; val=[]
        for _ in range(min(n,total-step)):
            x=torch.as_tensor(obs,dtype=torch.float32,device=dev)
            with torch.no_grad():
                d=model.distribution(x); a=d.sample().clamp(-1,1); oldlp=d.log_prob(a).sum(); v=model.critic(x).squeeze()
            nxt,r,t,tr,info=env.step(a.cpu().numpy()); done=t or tr
            ob.append(obs); ac.append(a.cpu().numpy()); lp.append(float(oldlp)); rw.append(float(r)); dn.append(float(done)); val.append(float(v)); obs=nxt; step+=1
            if done: obs,_=env.reset()
            if step>=total: break
        with torch.no_grad(): last=float(model.critic(torch.as_tensor(obs,dtype=torch.float32,device=dev)).item())
        adv=np.zeros(len(rw),np.float32); gae=0.
        for i in reversed(range(len(rw))):
            nextv=last if i==len(rw)-1 else val[i+1]
            nonterminal=1.-dn[i]; delta=rw[i]+.99*nextv*nonterminal-val[i]; gae=delta+.99*.95*nonterminal*gae; adv[i]=gae
        ret=adv+np.asarray(val); adv=(adv-adv.mean())/(adv.std()+1e-8)
        ot=torch.tensor(np.asarray(ob),dtype=torch.float32,device=dev); at=torch.tensor(np.asarray(ac),dtype=torch.float32,device=dev); lpt=torch.tensor(lp,dtype=torch.float32,device=dev); adt=torch.tensor(adv,device=dev); rt=torch.tensor(ret,device=dev)
        inds=np.arange(len(ob))
        for _ in range(epochs):
            np.random.shuffle(inds)
            for st in range(0,len(inds),batch):
                ix=inds[st:st+batch]; d=model.distribution(ot[ix]); newlp=d.log_prob(at[ix]).sum(-1); ratio=(newlp-lpt[ix]).exp(); pg=-torch.minimum(ratio*adt[ix],ratio.clamp(.8,1.2)*adt[ix]).mean(); vloss=((model.critic(ot[ix]).squeeze(-1)-rt[ix])**2).mean(); ent=d.entropy().sum(-1).mean(); loss=pg+.5*vloss-.01*ent
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),.5); opt.step()
        mean_reward=float(np.mean(rw)); logs.append({"timesteps":step,"mean_rollout_reward":mean_reward})
        torch.save(model.cpu().state_dict(),out/"policy.pt"); model.to(dev)
        print(json.dumps(logs[-1]),flush=True)
    with open(out/"training.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=logs[0].keys()); w.writeheader(); w.writerows(logs)
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: commit="unavailable"
    (out/"metadata.json").write_text(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"mujoco":__import__('mujoco').__version__,"device":dev,"config":cfg,"seed":seed,"git_commit":commit},indent=2))
    env.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); train(p.parse_args().config)
