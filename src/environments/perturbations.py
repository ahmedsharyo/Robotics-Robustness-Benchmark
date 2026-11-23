"""Episode-level parameter sampling and physically valid MuJoCo changes."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Perturbation:
    mass: float = 1.0
    friction: float = 1.0
    actuator_strength: float = 1.0
    delay_steps: int = 0
    sensor_noise: float = 0.0
    disturbance_force: float = 0.0

    def to_dict(self):
        return asdict(self)


def sample_perturbation(rng, ranges):
    v = {}
    for k, bounds in ranges.items():
        v[k] = int(rng.integers(bounds[0], bounds[1] + 1)) if k == "delay_steps" else float(rng.uniform(*bounds))
    return Perturbation(**v)


def apply_perturbation(model, nominal, p):
    """Scale masses and inertia together; preserve positive physically valid inertias."""
    model.body_mass[:] = nominal["body_mass"] * p.mass
    model.body_inertia[:] = nominal["body_inertia"] * p.mass
    model.dof_frictionloss[:] = nominal["frictionloss"] * p.friction
    model.actuator_gainprm[:] = nominal["actuator_gainprm"]
    model.actuator_gainprm[:, 0] *= p.actuator_strength
    model.body_mass[:] = np.maximum(model.body_mass, 1e-5)
    model.body_inertia[:] = np.maximum(model.body_inertia, 1e-8)
