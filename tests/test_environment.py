import numpy as np
import pytest
gym=pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")
from src.environments.arm_env import ArmReachEnv

def test_reset_step_spaces_and_reproducibility():
    env=ArmReachEnv(episode_steps=3); a,i=env.reset(seed=8); b,j=env.reset(seed=8)
    assert a.shape==(10,) and np.allclose(a,b) and env.observation_space.contains(a)
    o,r,t,tr,info=env.step(np.zeros(2)); assert env.observation_space.contains(o) and np.isfinite(r)
    for _ in range(3): o,r,t,tr,info=env.step(np.zeros(2))
    assert tr

def test_joint_limit_metric_present():
    env=ArmReachEnv(); env.reset(seed=1); *_,info=env.step(np.ones(2)); assert "joint_limit_violation" in info
