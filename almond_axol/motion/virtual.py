"""An in-memory robot, so simulated and real motion run the identical code path.

:class:`~almond_axol.robot.sim.Sim` couples "hold joint state" to "render it in
a viser server on a fixed port". That is exactly right for ``--sim`` playback
of an existing program, and exactly wrong for an interactive application that
owns its own viser scene: you end up with two servers, two ports, and two
copies of the robot.

:class:`VirtualRobot` is the state half on its own. It satisfies
:class:`~almond_axol.robot.base.RobotBase`, so
:class:`~almond_axol.motion.commander.MotionCommander` cannot tell it apart
from real hardware — the jog UI streams a trajectory into a ``VirtualRobot``
using the same call it will later make against an :class:`~almond_axol.robot.axol.Axol`,
and rendering is the application's business rather than the robot's.

It starts at the rest pose rather than at zeros. Zeros is the straight-down
singular configuration (see :mod:`almond_axol.motion.kinematics`), and seeding
a session there means the first IK solve has nowhere to go.
"""

from __future__ import annotations

import numpy as np

from ..constants import ARM_JOINTS
from ..robot.base import RobotBase

N_ARM_JOINTS = len(ARM_JOINTS)


class VirtualRobot(RobotBase):
    """A robot that only remembers what it was told.

    Reads return the last commanded positions, so a control loop closes exactly
    as it would on hardware with perfect tracking. That is the honest
    description of what this provides and what it does not: geometry, not
    dynamics. It will replay a millimetre-perfect insertion regardless of
    whether the real arm could manage one.

    Args:
        left:  Initial ``(8,)`` left-arm state — 7 joints then gripper.
            Defaults to the configured rest pose with the gripper open.
        right: Same for the right arm.

    Example::

        async with VirtualRobot() as robot:
            await robot.motion_control(left=q_left, right=q_right)
            left, right = await robot.get_positions()
    """

    def __init__(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        from ..teleop.config import VRTeleopConfig

        rest = VRTeleopConfig()
        self._left = (
            np.append(np.asarray(rest.rest_pose_left, dtype=np.float32), 1.0)
            if left is None
            else np.asarray(left, dtype=np.float32).copy()
        )
        self._right = (
            np.append(np.asarray(rest.rest_pose_right, dtype=np.float32), 1.0)
            if right is None
            else np.asarray(right, dtype=np.float32).copy()
        )
        self._enabled = False

    async def enable(self) -> None:
        """Mark the robot enabled. No hardware to bring up."""
        self._enabled = True

    async def disable(self) -> None:
        """Mark the robot disabled."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Whether :meth:`enable` has been called."""
        return self._enabled

    async def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the last commanded ``(8,)`` state of each arm."""
        return self._left.copy(), self._right.copy()

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Store the commanded positions. ``None`` leaves that arm alone."""
        if left is not None:
            self._left = np.asarray(left, dtype=np.float32).copy()
        if right is not None:
            self._right = np.asarray(right, dtype=np.float32).copy()

    def set_state(self, left: np.ndarray, right: np.ndarray) -> None:
        """Force both arms' state without going through the control path.

        For seeding a session from a solved configuration. Not something
        hardware can do, which is why it is not on :class:`RobotBase`.
        """
        self._left = np.asarray(left, dtype=np.float32).copy()
        self._right = np.asarray(right, dtype=np.float32).copy()
