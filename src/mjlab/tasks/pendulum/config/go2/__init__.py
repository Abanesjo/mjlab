from mjlab.tasks.pendulum.config.go2.env_cfgs import (
  unitree_go2_pendulum_env_cfg,
)
from mjlab.tasks.pendulum.config.go2.rl_cfg import (
  unitree_go2_pendulum_ppo_runner_cfg,
)
from mjlab.tasks.pendulum.rl import PendulumOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Pendulum-Balance-Unitree-Go2",
  env_cfg=unitree_go2_pendulum_env_cfg(),
  play_env_cfg=unitree_go2_pendulum_env_cfg(play=True),
  rl_cfg=unitree_go2_pendulum_ppo_runner_cfg(),
  runner_cls=PendulumOnPolicyRunner,
)
