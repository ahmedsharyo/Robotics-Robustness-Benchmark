"""Two-link MuJoCo reaching task with explicit observation-side noise and delay."""
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from .perturbations import Perturbation, apply_perturbation


XML = r'''<mujoco model="two_link_arm"><option gravity="0 0 0" timestep="0.002"/><worldbody>
<body name="base" pos="0 0 0.1"><joint name="shoulder" type="hinge" axis="0 1 0" damping="0.02"/><geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.035" mass="0.8"/><body name="link2" pos="0.35 0 0"><joint name="elbow" type="hinge" axis="0 1 0" damping="0.02"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/><site name="tip" pos="0.3 0 0" size="0.015"/></body></body></worldbody>
<actuator><motor joint="shoulder" gear="1" ctrllimited="true" ctrlrange="-5 5"/><motor joint="elbow" gear="1" ctrllimited="true" ctrlrange="-3 3"/></actuator></mujoco>'''


class ArmReachEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, episode_steps=200, control_dt=0.04, success_threshold=0.08, randomization=False, randomization_ranges=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.nominal = {"body_mass": self.model.body_mass.copy(),
                        "body_inertia": self.model.body_inertia.copy(),
                        "frictionloss": self.model.dof_frictionloss.copy(), "actuator_gainprm": self.model.actuator_gainprm.copy()}
        self.episode_steps, self.control_dt = episode_steps, control_dt
        self.frame_skip = max(1, round(control_dt / self.model.opt.timestep))
        self.success_threshold, self.randomization = success_threshold, randomization
        self.ranges = randomization_ranges or {"mass": (.75, 1.25),
                                               "friction": (.5, 1.8),
                                               "actuator_strength": (.8, 1.2),
                                               "delay_steps": (0, 2), "sensor_noise": (0., .02), "disturbance_force": (0., .8)}
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.steps = 0
        self.p = Perturbation()
        self.queue = deque()
        self.target = np.array([.4, 0., .15])

    def _tip(self):
        return self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")].copy()

    def _obs(self):
        noise = self.rng.normal(0, self.p.sensor_noise, 4)
        q = self.data.qpos.copy() + noise[:2]
        dq = self.data.qvel.copy() + noise[2:]
        tip = self._tip()
        # Target is represented in the planar task coordinates x,z; third component retained in error metric.
        return np.asarray([*q, *dq, self.target[0], self.target[2], tip[0], tip[2]], dtype=np.float32)

    def set_perturbation(self, p):
        self.p = p
        apply_perturbation(self.model, self.nominal, p)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.rng = np.random.default_rng(seed)
        if options and "perturbation" in options:
            self.set_perturbation(options["perturbation"])
        elif self.randomization:
            from .perturbations import sample_perturbation
            self.set_perturbation(sample_perturbation(self.rng, self.ranges))
        else: self.set_perturbation(Perturbation())
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.rng.uniform(-.15, .15, 2)
        self.target = np.array([self.rng.uniform(.35, .58), 0., self.rng.uniform(.05, .32)])
        self.steps = 0
        self.queue = deque()
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {"target": self.target.copy(), "perturbation": self.p.to_dict()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(2).clip(-1, 1)
        command = action * np.array([5., 3.])
        if self.p.delay_steps:
            self.queue.append(command.copy())
            applied = self.queue.popleft() if len(self.queue) > self.p.delay_steps else np.zeros(2)
        else: applied = command
        self.data.ctrl[:] = applied
        for _ in range(self.frame_skip):
            if self.p.disturbance_force:
                self.data.qfrc_applied[0] = self.rng.uniform(-self.p.disturbance_force, self.p.disturbance_force)
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        pos = self._tip()
        err = float(np.linalg.norm(pos - self.target))
        effort = float(np.sum(np.square(action)))
        reward = -err - .002 * effort + (1.0 if err < self.success_threshold else 0.0)
        success = err < self.success_threshold
        terminated = bool(success)
        truncated = self.steps >= self.episode_steps
        info = {"endpoint_error": err, "success": success, "control_effort": effort, "joint_limit_violation": bool(np.any(np.abs(self.data.qpos) > 2.8)), "applied_action": applied.copy()}
        return self._obs(), reward, terminated, truncated, info
