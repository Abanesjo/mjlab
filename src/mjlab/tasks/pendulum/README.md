# Pendulum

Quadruped position-tracking while balancing a passive inverted pendulum rigidly
attached to the base. Port of IsaacLab's `Template-Go2-Pendulum-Direct-v0`.

Registered task: `Mjlab-Pendulum-Balance-Unitree-Go2`.

## Training

```sh
uv run train Mjlab-Pendulum-Balance-Unitree-Go2
```

Useful flags:

```sh
# Override env/run settings (tyro parses nested dataclasses).
uv run train Mjlab-Pendulum-Balance-Unitree-Go2 \
  --agent.run-name my_experiment \
  --agent.max-iterations 75000

# Multi-GPU.
uv run train Mjlab-Pendulum-Balance-Unitree-Go2 --gpu-ids "[0, 1]"

# Resume from a W&B checkpoint.
uv run train Mjlab-Pendulum-Balance-Unitree-Go2 \
  --wandb-run-path <entity>/<project>/<run_id>

# See all options.
uv run train Mjlab-Pendulum-Balance-Unitree-Go2 --help
```

Logs land in `logs/rsl_rl/go2_pendulum/<timestamp>[_<run_name>]/`.

## Playing

```sh
# Latest local checkpoint under logs/rsl_rl/go2_pendulum/.
uv run play Mjlab-Pendulum-Balance-Unitree-Go2 \
  --checkpoint-file logs/rsl_rl/go2_pendulum/<run>/model_<N>.pt

# Pull a checkpoint from W&B.
uv run play Mjlab-Pendulum-Balance-Unitree-Go2 \
  --wandb-run-path <entity>/<project>/<run_id>

# Dummy policies (useful for inspecting the env).
uv run play Mjlab-Pendulum-Balance-Unitree-Go2 --agent zero
uv run play Mjlab-Pendulum-Balance-Unitree-Go2 --agent random

# Record a video instead of launching a viewer.
uv run play Mjlab-Pendulum-Balance-Unitree-Go2 \
  --checkpoint-file <...> --video --video-length 600
```

`play` loads the env with `play=True` (corruption disabled, pushes removed,
`base_contact`/`position_goal_violation` terminations disabled, curriculum
frozen). The ONNX policy is exported next to the checkpoint automatically.

## Observation space

Actor observation is a single flat `float32` tensor of shape `(batch, 112)`,
assembled term-major from the entries in `actor_terms`
(`pendulum_env_cfg.py`). History terms are flattened oldest-to-newest with
joints interleaved inside each frame.

| Slice | Term | Dim | Semantics |
|---|---|---|---|
| `[0:3]` | `base_lin_vel` | 3 | IMU body-frame linear velocity [m/s] |
| `[3:6]` | `base_ang_vel` | 3 | IMU body-frame angular velocity [rad/s] |
| `[6:9]` | `projected_gravity` | 3 | gravity in body frame (~`[0, 0, -1]` upright) |
| `[9:12]` | `state_error` | 3 | body-frame goal error `[dx, dy, dyaw]` (m, m, rad in `[-pi, pi]`) |
| `[12:24]` | `leg_joint_pos` | 12 | `q_i - q_i_default` [rad] |
| `[24:36]` | `leg_joint_vel` | 12 | raw `qd` [rad/s] (default `qd` is zero) |
| `[36:66]` | `pendulum_joint_pos` | 30 | 15-frame history, 2 joints: `[p(t-14)_j1, p(t-14)_j2, ..., p(t0)_j1, p(t0)_j2]` |
| `[66:96]` | `pendulum_joint_vel` | 30 | 15-frame history, same layout [rad/s] |
| `[96:108]` | `actions` | 12 | last pre-scale network output |
| `[108:112]` | `clock_inputs` | 4 | `sin(2*pi*phase)` per foot, diagonal-trot offsets |

**Leg joint order** (also the order of the 12-dim action output):

```
[FL_hip, FL_thigh, FL_calf,
 FR_hip, FR_thigh, FR_calf,
 RL_hip, RL_thigh, RL_calf,
 RR_hip, RR_thigh, RR_calf]
```

**Pendulum joint order**: `[pendulum_joint1, pendulum_joint2]`.

The exported ONNX policy expects **raw values in the units above**. Running
mean/std normalization (`obs_normalization=True` in
`config/go2/rl_cfg.py`) is baked into the graph, so callers do not normalize
manually.

**Critic observation** (training-only) shares the term order but collapses
pendulum history to a single frame (`history_length=0` override on the critic
group), for a total of 56 dims. The exported policy is actor-only.

## Action space

Policy output: `float32` tensor of shape `(batch, 12)`. Each entry is a
pre-scale residual applied on top of the default joint pose:

```
target_joint_pos[i] = default_joint_pos[i] + GO2_ACTION_SCALE[i] * action[i]
```

`GO2_ACTION_SCALE[i] = 0.25 * effort_limit / stiffness = 0.235` for every leg
joint (`asset_zoo/robots/unitree_go2/go2_constants.py`). `target_joint_pos`
is fed to the 50 Hz PD loop (`decimation=4 * sim timestep=0.005s`).

The leg joint ordering above applies to both the action tensor and
`leg_joint_pos`/`leg_joint_vel` observations.
