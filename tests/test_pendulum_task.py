"""Tests specific to the Go2 pendulum task."""

import pytest

from mjlab.asset_zoo.robots import GO2_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.pendulum.mdp import PositionGoalCommandCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg


@pytest.fixture(scope="module")
def pendulum_task_ids() -> list[str]:
  return [t for t in list_tasks() if "Pendulum" in t]


def test_go2_pendulum_registered(pendulum_task_ids: list[str]) -> None:
  assert "Mjlab-Pendulum-Balance-Unitree-Go2" in pendulum_task_ids


def test_pendulum_task_has_position_goal_command(
  pendulum_task_ids: list[str],
) -> None:
  for task_id in pendulum_task_ids:
    cfg = load_env_cfg(task_id)
    assert "position_goal" in cfg.commands, (
      f"Task {task_id} missing 'position_goal' command"
    )
    assert isinstance(cfg.commands["position_goal"], PositionGoalCommandCfg)


def test_pendulum_task_has_pendulum_terminations(
  pendulum_task_ids: list[str],
) -> None:
  for task_id in pendulum_task_ids:
    cfg = load_env_cfg(task_id)
    for name in ("pendulum_fallen", "pendulum_contact"):
      assert name in cfg.terminations, f"Task {task_id} missing '{name}' termination"


def test_pendulum_task_has_required_rewards(
  pendulum_task_ids: list[str],
) -> None:
  expected = (
    "position_tracking",
    "progress",
    "yaw_alignment",
    "pendulum_upright",
    "pendulum_velocity",
    "balanced_movement",
    "termination_penalty",
  )
  for task_id in pendulum_task_ids:
    cfg = load_env_cfg(task_id)
    for name in expected:
      assert name in cfg.rewards, f"Task {task_id} missing reward '{name}'"


def test_go2_pendulum_has_correct_action_scale(
  pendulum_task_ids: list[str],
) -> None:
  for task_id in pendulum_task_ids:
    if "Go2" not in task_id:
      continue
    cfg = load_env_cfg(task_id)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    assert joint_pos_action.scale == GO2_ACTION_SCALE


def test_go2_pendulum_play_config_disables_corruption() -> None:
  cfg = load_env_cfg("Mjlab-Pendulum-Balance-Unitree-Go2", play=True)
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.curriculum == {}
  assert "push_robot" not in cfg.events


@pytest.mark.slow
def test_go2_pendulum_env_steps_smoke() -> None:
  """Instantiate the env on CPU, step a handful of zero-action times."""
  import io
  import warnings
  from contextlib import redirect_stderr, redirect_stdout

  import torch

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      cfg = load_env_cfg("Mjlab-Pendulum-Balance-Unitree-Go2")
      cfg.scene.num_envs = 2
      env = ManagerBasedRlEnv(cfg, device="cpu")
      obs_dict, _ = env.reset()
      actor = obs_dict["actor"]
      critic = obs_dict["critic"]
      assert isinstance(actor, torch.Tensor)
      assert isinstance(critic, torch.Tensor)
      assert actor.shape == (2, 112)
      assert critic.shape == (2, 56)
      assert not torch.isnan(actor).any()

      action = torch.zeros(
        env.num_envs, env.action_manager.total_action_dim, device=env.device
      )
      for _ in range(3):
        obs_dict, reward, terminated, truncated, info = env.step(action)
        actor = obs_dict["actor"]
        assert isinstance(actor, torch.Tensor)
        assert not torch.isnan(reward).any()
        assert not torch.isnan(actor).any()
      env.close()
