import numpy as np
import pytest
pytest.importorskip("mujoco"); pytest.importorskip("gymnasium")
from src.environments.arm_env import ArmReachEnv
from src.environments.perturbations import Perturbation

def test_mass_friction_and_actuator_change():
    env=ArmReachEnv(); env.reset(options={"perturbation":Perturbation(mass=1.4,friction=1.5,actuator_strength=.8)})
    assert np.allclose(env.model.body_mass,env.nominal["body_mass"]*1.4)
    assert np.allclose(env.model.dof_frictionloss,env.nominal["frictionloss"]*1.5)

def test_delay_queue_is_explicit():
    env=ArmReachEnv(); env.reset(options={"perturbation":Perturbation(delay_steps=2)})
    _,_,_,_,i=env.step(np.ones(2)); assert np.allclose(i["applied_action"],0)
    _,_,_,_,i=env.step(np.ones(2)); assert np.allclose(i["applied_action"],0)
    _,_,_,_,i=env.step(np.ones(2)); assert np.allclose(i["applied_action"],[5,3])

def test_sensor_noise_only_affects_observation():
    env=ArmReachEnv(); env.reset(seed=1, options={"perturbation":Perturbation(sensor_noise=.1)})
    state=env.data.qpos.copy(); obs=env._obs(); assert np.array_equal(env.data.qpos,state) and obs.shape==(10,)
