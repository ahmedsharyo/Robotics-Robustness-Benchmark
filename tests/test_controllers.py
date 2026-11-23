import numpy as np
from src.controllers.classical_controller import PDController

def test_pd_output_valid():
    a=PDController()(np.zeros(10)); assert a.shape==(2,) and np.all(np.isfinite(a)) and np.max(np.abs(a))<=1
