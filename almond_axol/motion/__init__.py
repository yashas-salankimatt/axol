"""Cartesian motion: absolute poses, waypoint programs, and execution.

Where :mod:`almond_axol.kinematics` provides the raw solver and
:mod:`almond_axol.teleop` drives it from a headset in *relative* deltas, this
package is the absolute-coordinate layer: a pose you can type, jog, store in a
program, reorder, replay in the simulator, and then run unchanged on hardware.

The pieces, in dependency order:

- :mod:`.frames`     — :class:`~.frames.Pose`, the world frame, tools, user frames.
- :mod:`.kinematics` — :class:`~.kinematics.CartesianKinematics`, FK/IK that
  converges on absolute targets.
- :mod:`.program`    — :class:`~.program.Waypoint` / :class:`~.program.Program`,
  the editable, reorderable, on-disk motion program.
- :mod:`.commander`  — :class:`~.commander.MotionCommander`, planning and
  execution against any :class:`~almond_axol.robot.base.RobotBase`.
- :mod:`.virtual`    — :class:`~.virtual.VirtualRobot`, an in-memory robot so
  the simulator and the real arm run the identical code path.
"""

from .commander import ExecutionReport, MotionCommander, PlannedProgram
from .frames import (
    ARMS,
    FLANGE,
    GRIPPER_TIP,
    WORLD,
    Arm,
    Frame,
    Pose,
    Tool,
    default_tool,
)
from .kinematics import CartesianKinematics, IKSolution, UnreachableError
from .program import MoveType, Program, Waypoint
from .virtual import VirtualRobot

__all__ = [
    "ARMS",
    "FLANGE",
    "GRIPPER_TIP",
    "WORLD",
    "Arm",
    "CartesianKinematics",
    "ExecutionReport",
    "Frame",
    "IKSolution",
    "MotionCommander",
    "MoveType",
    "PlannedProgram",
    "Pose",
    "Program",
    "Tool",
    "UnreachableError",
    "VirtualRobot",
    "Waypoint",
    "default_tool",
]
