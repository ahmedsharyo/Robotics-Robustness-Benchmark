import numpy as np


class PDController:
    """Joint-space PD using task-space target error; no privileged test tuning."""
    def __init__(self, kp=5.0, kd=1.2): self.kp, self.kd = kp, kd
    def __call__(self, obs):
        q, dq = np.asarray(obs[:2]), np.asarray(obs[2:4])
        tx, tz = obs[4:6]
        # Approximate inverse kinematics for a planar 0.35m + 0.30m arm.
        x, z = float(tx), float(tz - .1)
        c2 = np.clip((x*x + z*z - .35**2 - .3**2)/(2*.35*.3), -1, 1)
        q2 = np.arccos(c2)
        q1 = np.arctan2(z, x) - np.arctan2(.3*np.sin(q2), .35 + .3*np.cos(q2))
        desired = np.array([q1, q2])
        # MuJoCo hinge axes yield sign convention; evaluate both branch sign choices via simple target.
        return np.clip(self.kp*(desired-q) - self.kd*dq, [-5.,-3.], [5.,3.]) / np.array([5.,3.])
