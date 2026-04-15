"""Unitree Go2 constants (with pendulum attached to the base)."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_PENDULUM_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go2" / "xmls" / "go2_pendulum.xml"
)
assert GO2_PENDULUM_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2_PENDULUM_XML))


##
# Joint names.
##

GO2_LEG_JOINT_NAMES: tuple[str, ...] = (
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
)

GO2_PENDULUM_JOINT_NAMES: tuple[str, ...] = (
  "pendulum_joint1",
  "pendulum_joint2",
)

##
# Actuator config.
##

# Mirrors Isaac Lab's DCMotorCfg values for Go2 legs
# (stiffness=25, damping=0.6, effort=23.5, no armature/friction).
GO2_LEG_STIFFNESS = 25.0
GO2_LEG_DAMPING = 0.6
GO2_LEG_EFFORT_LIMIT = 23.5

GO2_LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_joint",
    ".*_thigh_joint",
    ".*_calf_joint",
  ),
  stiffness=GO2_LEG_STIFFNESS,
  damping=GO2_LEG_DAMPING,
  effort_limit=GO2_LEG_EFFORT_LIMIT,
  armature=0.0,
)

##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.4),
  joint_pos={
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[LR]_thigh_joint": 0.8,
    "R[LR]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
    "pendulum_joint1": 0.0,
    "pendulum_joint2": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1.0, 5e-3, 5e-4)},
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_LEG_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_pendulum_robot_cfg() -> EntityCfg:
  """Return a fresh Unitree Go2 + pendulum EntityCfg.

  A new EntityCfg is returned each call so the config is safe to share across
  task configs without mutation leaking between them.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )


# Per-joint action scale: 0.25 * effort / stiffness, mirroring the convention
# used by Go1 (go1_constants.py:148-156). With Isaac Lab's DCMotor values this
# is 0.25 * 23.5 / 25.0 = 0.235 per leg joint.
GO2_ACTION_SCALE: dict[str, float] = {}
for _name_expr in GO2_LEG_ACTUATOR_CFG.target_names_expr:
  GO2_ACTION_SCALE[_name_expr] = 0.25 * GO2_LEG_EFFORT_LIMIT / GO2_LEG_STIFFNESS


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_pendulum_robot_cfg())
  viewer.launch(robot.spec.compile())
