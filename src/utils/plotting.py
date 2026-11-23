from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def plot_sweep(csv_path="results/robustness.csv", output="results/figures"):
    out=Path(output); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(csv_path)
    df["params"]=df.perturbation.map(json.loads)
    for key,title in [("mass","Mass mismatch"),("friction","Joint friction"),("delay_steps","Actuator delay"),("sensor_noise","Sensor noise"),("disturbance_force","External force")]:
        df["x"]=df.params.map(lambda p:p.get(key,0))
        fig,ax=plt.subplots(figsize=(6,4))
        for name,g in df.groupby("controller"):
            s=g.groupby("x").success.agg(["mean","std","count"]); ax.errorbar(s.index,s["mean"],yerr=s["std"]/(s["count"]**.5),marker="o",label=name)
        ax.set(xlabel=key,ylabel="Success rate",title=title); ax.set_ylim(-.03,1.03); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/f"success_{key}.png",dpi=180); fig.savefig(out/f"success_{key}.pdf"); plt.close(fig)
    if Path("results/checkpoints/nominal/training.csv").exists() or Path("results/checkpoints/randomized/training.csv").exists():
        fig,ax=plt.subplots(figsize=(6,4))
        for n in ("nominal","randomized"):
            p=Path(f"results/checkpoints/{n}/training.csv")
            if p.exists():
                d=pd.read_csv(p); ax.plot(d.timesteps,d.mean_rollout_reward,label=n)
        ax.set(xlabel="Environment steps",ylabel="Mean rollout reward",title="Training curves"); ax.legend(); fig.tight_layout(); fig.savefig(out/"training_curves.png",dpi=180); plt.close(fig)

if __name__=="__main__": plot_sweep()
