import numpy as np


def summarize(rows):
    returns = np.asarray([r["episode_return"] for r in rows], float)
    success = np.asarray([r["success"] for r in rows], float)
    errs = np.asarray([r["endpoint_error"] for r in rows], float)
    return {"episodes": len(rows), "success_rate": float(success.mean()), "success_std": float(success.std(ddof=1)) if len(success)>1 else 0., "mean_endpoint_error": float(errs.mean()), "median_endpoint_error": float(np.median(errs)), "mean_return": float(returns.mean()), "return_std": float(returns.std(ddof=1)) if len(returns)>1 else 0., "return_p05": float(np.quantile(returns,.05)), "mean_completion_time": float(np.mean([r["steps"] for r in rows])), "mean_control_effort": float(np.mean([r["control_effort"] for r in rows])), "joint_limit_violations": int(sum(r["joint_limit_violations"] for r in rows))}
