# Robotics Robustness Benchmark

A reproducible CPU/MPS benchmark for planar two-link reaching under MuJoCo model mismatch. The repository compares a nominal PPO policy, the same PPO architecture trained with randomized dynamics, and a joint-space PD baseline. It measures simulation robustness only; no physical robot testing is provided.

## Install and run

Use Python 3.11+ (the current project setup targets this version). On an Apple Silicon Mac, PyTorch may use MPS; training falls back to CPU automatically.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m src.training.train --config configs/smoke_test.yaml
python -m src.training.train --config configs/nominal.yaml
python -m src.training.train --config configs/domain_randomization.yaml
python -m src.evaluation.evaluate --config configs/evaluation.yaml
python -m src.evaluation.robustness_sweep --config configs/evaluation.yaml
python -m src.evaluation.worst_case --config configs/evaluation.yaml --budget 24
python -m src.utils.plotting
```

PPO uses identical MLP, optimizer, reward, and configured environment-step budgets for nominal and randomized policies. The randomized policy samples physical factors per episode and sensor noise/disturbance per observation/control interval. Evaluation uses common episode seeds. The classical controller uses a fixed approximate inverse-kinematics plus PD law with no held-out tuning; it uses joint position/velocity and the target, while learned policies consume the same observation. None receives true perturbed parameters.

## Outputs

Training checkpoints and metadata go to `results/checkpoints/{nominal,randomized}/`. Evaluation writes raw episode records and summaries under `results/`; worst-case search records its seeded scenario budget and empirical minimum return. `src.utils.plotting` generates PNG/PDF figures for robustness dimensions and available learning curves. Evaluation condition records serialize controller, episode seed, sampled perturbation, and metrics.

## Protocol and interpretation

`configs/domain_randomization.yaml` defines training ranges. `configs/evaluation.yaml` holds isolated mismatch levels plus a combined held-out configuration. The finite random search in `worst_case.py` reports only the worst observed case, not a certified worst case. For full benchmark claims, use at least three independent training seeds and more evaluation episodes than the quick default; explicitly separate training-seed variability from episode variability. The supplied defaults are affordable starter settings, not a substitute for that statistical study.

`configs/smoke_test.yaml` is intended to validate the execution path without full training. A training invocation will create a checkpoint even at smoke scale. Checkpoint paths are intentionally distinct for the two learned controllers.

## Limitations

This is a minimal planar arm with simplified gravity-free dynamics. The target and end-effector geometry are planar; actuator delay is defined as a queue of exactly N control commands (N=0 applies the current command, N>0 initially applies zero until a command reaches the queue head). Link masses and inertias are scaled together to maintain valid positive inertial quantities. Disturbances are bounded random joint torque impulses refreshed within each physics substep, not calibrated environmental loads. This does not establish sim-to-real transfer.
