"""Task-specific events for the Go2 pendulum environment.

No pendulum-specific events today — the pendulum initial pose is set by
reusing :func:`mjlab.envs.mdp.events.reset_joints_by_offset` with an
``asset_cfg`` scoped to the two pendulum joints. Reserved as an extension
point for future DR tuning (e.g., sampling the pendulum start with the
sign+magnitude split Isaac Lab uses instead of symmetric uniform).
"""

from __future__ import annotations
