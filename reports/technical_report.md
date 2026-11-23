# Technical report: model mismatch and domain randomization

## Introduction

Robotic controllers depend on a model of dynamics and sensing. Differences in mass, friction, actuation, delay, sensing, and disturbances can degrade a controller trained or tuned in one simulator. This benchmark is designed to measure those effects in a compact simulated reaching task.

## Methodology

The primary task is a two-link MuJoCo planar arm with bounded joint torque, randomized initial joint positions, and randomly sampled planar targets. Observations contain joint position and velocity, target coordinates, and end-effector coordinates. The environment uses Gymnasium reset/step semantics. A two-hidden-layer 64-unit Gaussian actor and value network are trained with clipped PPO on CPU or MPS. Nominal and domain-randomized configurations share architecture and optimization settings. Randomization samples episode-level mass, friction, actuator strength, and delay, plus observation noise and joint torque disturbances. The classical comparison is a fixed approximate inverse-kinematics PD controller.

Evaluation scenarios are seeded and shared across available controllers. The robustness sweep tests isolated mismatch severities and a combined configuration outside the configured randomized training ranges. The worst-case script uses a seeded finite random search over declared bounds and reports an empirical observed minimum only.

## Results

**Pending execution.** No benchmark experiment has been run in this workspace. The current environment has Python 3.10 and lacks MuJoCo, Gymnasium, PyTorch, NumPy, SciPy, Matplotlib, Pandas, PyYAML, and pytest. Therefore there are no measured success rates, learning curves, robustness plots, or worst-case results to report. Run the commands in the README under Python 3.11+ after installing dependencies. Do not interpret planned configurations as findings.

## Analysis

Pending measured results. The scripts produce per-controller and per-condition summaries suitable for comparing nominal performance, in-range robustness, held-out combined generalization, failure rates, and lower-tail return. No claims about domain randomization benefits or classical-controller superiority are made before experiments are executed.

## Limitations

The robot model and reaching task are intentionally small and simplified. The configured randomization distributions are assumptions rather than universal physical bounds. The default evaluation episode count is for a quick start and cannot establish variation across three independent training seeds. The random search is finite and non-certified. No physical robot is available, so sim-to-real transfer is unverified.

## Conclusion

This repository provides an executable protocol intended to measure robustness under simulation mismatch. Empirical conclusions remain pending dependency installation and benchmark execution.
