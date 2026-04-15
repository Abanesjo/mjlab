"""Base environment config factory for the Go2 pendulum task.

Produces a :class:`ManagerBasedRlEnvCfg` configured for position-goal
tracking while balancing a passive inverted pendulum. Robot-specific wiring
(Go2 entity, sensor frame names, reward body/joint selections, action
scale) lives in ``config/go2/env_cfgs.py``.

Reward weights, sigmas, and termination thresholds mirror Isaac Lab's
``Template-Go2-Pendulum-Direct-v0``. See the port plan for the one-to-one
mapping and called-out deviations.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.pendulum import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Joint selectors. The Go2 variant uses these as-is; they match joint names
# declared in the unitree_go2 MJCF.
_LEG_JOINT_REGEX = r".*_(hip|thigh|calf)_joint"
_PENDULUM_JOINT_NAMES = ("pendulum_joint1", "pendulum_joint2")


def make_pendulum_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create a base pendulum task configuration.

  Scene is a flat plane with a single (unnamed, set per-robot) entity slot
  and no sensors (contact sensors are added per-robot because they reference
  robot-specific geom/body names).
  """

  ##
  # Observations.
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "state_error": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "position_goal"},
    ),
    "leg_joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(_LEG_JOINT_REGEX,)),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "leg_joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(_LEG_JOINT_REGEX,)),
      },
      noise=Unoise(n_min=-1.0, n_max=1.0),
    ),
    "pendulum_joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_PENDULUM_JOINT_NAMES),
      },
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "pendulum_joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_PENDULUM_JOINT_NAMES),
      },
      noise=Unoise(n_min=-1.0, n_max=1.0),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "clock_inputs": ObservationTermCfg(
      func=mdp.clock_inputs,
      params={"period_s": 0.5, "offsets": (0.0, 0.5, 0.5, 0.0)},
    ),
  }

  # Critic sees the clean state (no corruption).
  critic_terms = {**actor_terms}

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions.
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),  # Matches only the 12 leg position actuators.
      scale=1.0,  # Set per-robot via GO2_ACTION_SCALE dict.
      use_default_offset=True,
    )
  }

  ##
  # Commands.
  ##

  commands: dict[str, CommandTermCfg] = {
    "position_goal": mdp.PositionGoalCommandCfg(
      entity_name="robot",
      resampling_time_range=(1e9, 1e9),  # Per-episode (resample only on reset).
      dist_range=(0.0, 0.0),  # Stage 1 default; curriculum widens.
      bearing_range=(0.0, 2.0 * math.pi),
      yaw_range=(0.0, 0.0),
      debug_vis=True,
    )
  }

  ##
  # Events.
  ##

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_leg_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(_LEG_JOINT_REGEX,)),
      },
    ),
    "reset_pendulum_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        # Symmetric uniform on [-a, +a] matches sign+magnitude sampling in
        # distribution. Start at 0 (upright) for stage 1; curriculum may widen.
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=_PENDULUM_JOINT_NAMES),
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.apply_body_impulse,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "force_range": (0.0, 0.0),  # Curriculum widens.
        "torque_range": (0.0, 0.0),
        "duration_s": (0.05, 0.15),
        "cooldown_s": (5.0, 10.0),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.5, 1.25),
        "shared_random": True,
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "scale",
        "ranges": (0.9, 1.55),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.03, 0.03),
          1: (-0.03, 0.03),
          2: (-0.02, 0.05),
        },
      },
    ),
    "motor_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(_LEG_JOINT_REGEX,)),
        "operation": "scale",
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-math.radians(1.0), math.radians(1.0)),
      },
    ),
  }

  ##
  # Rewards.
  ##

  rewards = {
    # Goal-tracking signals.
    "position_tracking": RewardTermCfg(
      func=mdp.position_tracking,
      weight=0.4,
      params={"command_name": "position_goal", "std": 0.3},
    ),
    "progress": RewardTermCfg(
      func=mdp.progress,
      weight=10.0,
      params={"command_name": "position_goal"},
    ),
    "yaw_alignment": RewardTermCfg(
      func=mdp.yaw_alignment,
      weight=0.3,
      params={"command_name": "position_goal", "std": 0.2},
    ),
    # Pendulum balance.
    "pendulum_upright": RewardTermCfg(
      func=mdp.pendulum_upright,
      weight=0.45,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_PENDULUM_JOINT_NAMES),
        "std": 0.15,
      },
    ),
    "pendulum_velocity": RewardTermCfg(
      func=mdp.pendulum_velocity_l2,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_PENDULUM_JOINT_NAMES),
      },
    ),
    "balanced_movement": RewardTermCfg(
      func=mdp.balanced_movement,
      weight=0.1,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_PENDULUM_JOINT_NAMES),
      },
    ),
    # Action regularization.
    "action_l2": RewardTermCfg(func=envs_mdp.action_l2, weight=-0.1),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    "action_acc_l2": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.01),
    "torque_l2": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-0.0001,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(_LEG_JOINT_REGEX,)),
      },
    ),
    # Base posture.
    "orient_l2": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-0.8),
    "base_height": RewardTermCfg(
      func=envs_mdp.base_height_l2,
      weight=0.2,
      params={"target_height": 0.33, "std": 0.1},
    ),
    "lin_vel_z": RewardTermCfg(func=envs_mdp.lin_vel_z_l2, weight=-2.0),
    "ang_vel_xy": RewardTermCfg(func=envs_mdp.ang_vel_xy_l2, weight=-0.01),
    "dof_vel": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.003,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(_LEG_JOINT_REGEX,)),
      },
    ),
    "dof_acc": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(_LEG_JOINT_REGEX,)),
      },
    ),
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-5.0),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_tilt": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.pi / 3},  # 60 deg.
    ),
    "base_contact": TerminationTermCfg(
      func=envs_mdp.sustained,
      params={
        "inner": {
          "func": envs_mdp.body_contact_force,
          "params": {"sensor_name": "base_contact", "threshold": 1.0},
        },
        "duration_s": 0.0,
        "grace_period_s": 0.5,
      },
    ),
    "pendulum_contact": TerminationTermCfg(
      func=envs_mdp.sustained,
      params={
        "inner": {
          "func": envs_mdp.body_contact_force,
          "params": {"sensor_name": "pendulum_contact", "threshold": 1.0},
        },
        "duration_s": 0.0,
        "grace_period_s": 0.1,
      },
    ),
    "pendulum_fallen": TerminationTermCfg(
      func=envs_mdp.sustained,
      params={
        "inner": {
          "func": mdp.pendulum_fallen,
          "params": {
            "asset_cfg": SceneEntityCfg("robot", joint_names=_PENDULUM_JOINT_NAMES),
            "angle_rad": math.radians(60.0),
          },
        },
        "duration_s": 0.5,
        "grace_period_s": 0.1,
      },
    ),
    "base_height_too_low": TerminationTermCfg(
      func=envs_mdp.sustained,
      params={
        "inner": {
          "func": envs_mdp.root_height_below_minimum,
          "params": {"minimum_height": 0.28},
        },
        "duration_s": 10.0,
        "grace_period_s": 0.1,
      },
    ),
    "position_goal_violation": TerminationTermCfg(
      func=envs_mdp.sustained,
      params={
        "inner": {
          "func": mdp.position_goal_violation,
          "params": {"command_name": "position_goal", "max_dist": 5.0},
        },
        "duration_s": 15.0,
        "grace_period_s": 0.1,
      },
    ),
  }

  ##
  # Curriculum.
  ##

  curriculum = {
    "pendulum_difficulty": CurriculumTermCfg(
      func=mdp.pendulum_difficulty,
      params={
        "command_name": "position_goal",
        "pendulum_termination_name": "pendulum_fallen",
        "position_termination_name": "position_goal_violation",
        "push_event_name": "push_robot",
      },
    ),
  }

  ##
  # Assemble.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
      ),
      sensors=(),  # Per-robot config adds contact + IMU sensors.
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
