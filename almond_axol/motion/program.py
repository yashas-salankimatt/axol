"""The motion program: an ordered, editable list of Cartesian waypoints.

This is the file an operator builds, reorders, saves, replays in the simulator,
and then runs on the robot. It is the equivalent of a UR program tree or an
xArm Blockly sequence, reduced to the part that matters for moving an arm
somewhere precisely: a list of poses, how to travel between them, and what the
grippers do on arrival.

Pose is the intent; joints are the memory
----------------------------------------

Each :class:`Waypoint` stores **both** a Cartesian ``pose`` per arm and the
joint vector that reaches it. That redundancy is deliberate and it is the
single most important decision in this module.

- The **pose** is what the operator meant. It is what you edit, what you jog
  to, what survives being expressed relative to a fixture frame, and what makes
  a program readable and re-teachable.
- The **joints** are which of the infinitely many ways to reach that pose the
  arm actually used. Axol has 7 joints for a 6-DOF task, so a pose does not
  determine a configuration — the elbow can swing through a whole null space
  without the gripper moving at all.

Replaying from the stored joints is what makes a program *repeatable*. Solving
fresh each run invites the solver to settle into a different elbow branch, and
a different branch means different gear backlash, different deflection under
load, and a different absolute error at the tool — which is precisely the thing
a repeatability test is trying to measure. So joints are recorded when a
waypoint is authored and are used as the seed on replay.

Frames
------

A waypoint's poses are stored in a named :class:`~.frames.Frame`, defaulting to
the world. Teaching a fixture once and expressing waypoints relative to it means
nudging the fixture is a one-line re-teach instead of re-authoring the program —
the "feature" of a UR program, and the reason a machine-tending routine survives
contact with a real cell.

On-disk format
--------------

Version 2 JSON. Version 1 files (``axol waypoints``, joint angles only) are
read too: pass a :class:`~.kinematics.CartesianKinematics` to :meth:`Program.load`
and their poses are recovered by forward kinematics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..constants import ARM_JOINTS
from .frames import ARMS, GRIPPER_TIP, Arm, Frame, Pose, Tool

if TYPE_CHECKING:
    from .kinematics import CartesianKinematics

FORMAT_VERSION = 2
"""Version written by :meth:`Program.save`."""

LEGACY_VERSION = 1
"""Version written by ``axol waypoints`` — joint angles only, no poses."""

N_ARM_JOINTS = len(ARM_JOINTS)

GRIP_OPEN = 1.0
GRIP_CLOSED = 0.0


class MoveType(Enum):
    """How the arm travels *to* a waypoint.

    Mirrors the two motion primitives every industrial controller exposes.

    Attributes:
        JOINT:  Interpolate in joint space — the tool sweeps an arc, but the
            move is fast, always feasible if both ends are, and cannot run into
            a wrist singularity partway. Free-air repositioning. (``MoveJ``.)
        LINEAR: Hold the tool to a straight world-frame line while its
            orientation slerps. What you need for an approach, an insertion, or
            a withdrawal — anything where the *path* matters and not just the
            endpoint. Can fail if the line leaves the workspace. (``MoveL``.)
    """

    JOINT = "joint"
    LINEAR = "linear"

    @property
    def label(self) -> str:
        """Operator-facing name, matching teach-pendant vocabulary."""
        return {"joint": "MoveJ (joint)", "linear": "MoveL (linear)"}[self.value]


@dataclass
class ArmTarget:
    """What one arm does at one waypoint.

    Attributes:
        pose:   Target tool pose, in the waypoint's frame.
        joints: ``(7,)`` joint configuration that reaches ``pose``, or ``None``
            if this waypoint has never been resolved. See the module docstring
            for why this is stored rather than re-solved.
        grip:   Commanded gripper opening on arrival, ``[0, 1]``.
    """

    pose: Pose
    joints: np.ndarray | None = None
    grip: float = GRIP_OPEN

    def __post_init__(self) -> None:
        if self.joints is not None:
            # Copy, don't view: ``asarray`` aliases an already-float32 input,
            # so a caller mutating theirs would rewrite a stored waypoint.
            arr = np.array(self.joints, dtype=np.float32).reshape(-1)
            if arr.shape != (N_ARM_JOINTS,):
                raise ValueError(f"joints must be ({N_ARM_JOINTS},), got {arr.shape}")
            self.joints = arr
        self.grip = float(np.clip(self.grip, 0.0, 1.0))

    def to_json(self) -> dict:
        """JSON-serialisable form."""
        out: dict = {"pose": self.pose.to_json(), "grip": round(self.grip, 4)}
        if self.joints is not None:
            out["joints"] = [round(float(v), 6) for v in self.joints]
        return out

    @classmethod
    def from_json(cls, data: dict) -> ArmTarget:
        """Inverse of :meth:`to_json`."""
        joints = data.get("joints")
        return cls(
            pose=Pose.from_json(data["pose"]),
            joints=None if joints is None else np.asarray(joints, dtype=np.float32),
            grip=float(data.get("grip", GRIP_OPEN)),
        )

    def copy(self) -> ArmTarget:
        """A deep copy — the joint array is not shared."""
        return ArmTarget(
            pose=self.pose,
            joints=None if self.joints is None else self.joints.copy(),
            grip=self.grip,
        )


@dataclass
class Waypoint:
    """One pose of the whole robot, and how to get there.

    Attributes:
        targets: Per-arm :class:`ArmTarget`. Both arms are always present; an
            arm that should hold still simply repeats the previous waypoint's
            pose, and the planner pins it (see ``min_travel`` in
            :func:`~almond_axol.kinematics.path.plan_linear_segment`).
        move:    How to travel here from the previous waypoint.
        label:   Operator-facing name.
        speed:   Override for the program default, in m/s for
            :attr:`MoveType.LINEAR` and rad/s for :attr:`MoveType.JOINT`.
            ``None`` uses the program default.
        dwell:   Seconds to hold still on arrival, after the grippers move.
            ``None`` uses the program default.
        frame:   Name of the frame ``targets`` poses are expressed in.
    """

    targets: dict[Arm, ArmTarget]
    move: MoveType = MoveType.LINEAR
    label: str = ""
    speed: float | None = None
    dwell: float | None = None
    frame: str = "world"

    def __post_init__(self) -> None:
        missing = [arm for arm in ARMS if arm not in self.targets]
        if missing:
            raise ValueError(
                f"waypoint is missing targets for: {', '.join(a.value for a in missing)}"
            )

    # -- convenience -----------------------------------------------------

    def pose(self, arm: Arm) -> Pose:
        """Target tool pose for ``arm``."""
        return self.targets[arm].pose

    def grip(self, arm: Arm) -> float:
        """Commanded gripper opening for ``arm``."""
        return self.targets[arm].grip

    def joints(self, arm: Arm) -> np.ndarray | None:
        """Stored joint configuration for ``arm``, if this waypoint is resolved."""
        return self.targets[arm].joints

    @property
    def is_resolved(self) -> bool:
        """True once both arms carry a joint configuration."""
        return all(self.targets[arm].joints is not None for arm in ARMS)

    def full_q(self, kinematics: CartesianKinematics) -> np.ndarray | None:
        """Assemble both arms' joints into a full joint vector, or ``None``.

        ``None`` when the waypoint has not been resolved — the caller is
        expected to solve it (see :meth:`Program.resolve`).
        """
        if not self.is_resolved:
            return None
        q = np.zeros(kinematics.num_joints, dtype=np.float32)
        for arm in ARMS:
            q[kinematics.indices(arm)] = self.targets[arm].joints
        return q

    def with_pose(self, arm: Arm, pose: Pose) -> Waypoint:
        """A copy with ``arm``'s pose replaced and its stale joints dropped.

        Dropping the joints is the point: they described the *old* pose, and
        silently keeping them would replay the waypoint the operator just
        edited away from.
        """
        targets = {a: t.copy() for a, t in self.targets.items()}
        targets[arm] = ArmTarget(pose=pose, joints=None, grip=targets[arm].grip)
        return replace(self, targets=targets)

    def with_grip(self, arm: Arm, grip: float) -> Waypoint:
        """A copy with ``arm``'s commanded gripper opening replaced."""
        targets = {a: t.copy() for a, t in self.targets.items()}
        targets[arm] = replace(targets[arm], grip=float(np.clip(grip, 0.0, 1.0)))
        return replace(self, targets=targets)

    def resolved(self, q: np.ndarray, kinematics: CartesianKinematics) -> Waypoint:
        """A copy carrying the joint configuration from full vector ``q``."""
        targets = {a: t.copy() for a, t in self.targets.items()}
        for arm in ARMS:
            targets[arm] = replace(
                targets[arm],
                joints=np.asarray(q, dtype=np.float32)[kinematics.indices(arm)],
            )
        return replace(self, targets=targets)

    def copy(self) -> Waypoint:
        """A deep copy."""
        return replace(self, targets={a: t.copy() for a, t in self.targets.items()})

    # -- serialization ---------------------------------------------------

    def to_json(self) -> dict:
        """JSON-serialisable form."""
        out: dict = {
            "label": self.label,
            "move": self.move.value,
            "frame": self.frame,
        }
        if self.speed is not None:
            out["speed"] = round(float(self.speed), 6)
        if self.dwell is not None:
            out["dwell"] = round(float(self.dwell), 4)
        for arm in ARMS:
            out[arm.value] = self.targets[arm].to_json()
        return out

    @classmethod
    def from_json(cls, data: dict) -> Waypoint:
        """Inverse of :meth:`to_json`."""
        return cls(
            targets={arm: ArmTarget.from_json(data[arm.value]) for arm in ARMS},
            move=MoveType(data.get("move", MoveType.LINEAR.value)),
            label=str(data.get("label", "")),
            speed=data.get("speed"),
            dwell=data.get("dwell"),
            frame=str(data.get("frame", "world")),
        )


class ProgramError(ValueError):
    """A program file could not be read, or an edit was out of range."""


@dataclass
class Program:
    """An ordered, editable list of waypoints plus the settings they run under.

    Indexing, iteration and ``len`` behave like a list. Every mutation is a
    method rather than direct list access so the UI has one place to hook and
    so an out-of-range edit raises something an operator can read.

    Attributes:
        waypoints:     Ordered waypoints.
        tool:          Tool centre point every pose is expressed at.
        frames:        Named frames by name, always including ``world``.
        speed:         Default linear speed, m/s.
        ang_speed:     Default angular speed, rad/s.
        joint_speed:   Default joint speed for :attr:`MoveType.JOINT`, rad/s.
        dwell:         Default hold at each waypoint, seconds.
        grip_time:     Seconds spent working a gripper that changes state.
    """

    waypoints: list[Waypoint] = field(default_factory=list)
    tool: Tool = GRIPPER_TIP
    frames: dict[str, Frame] = field(default_factory=dict)
    speed: float = 0.10
    ang_speed: float = 0.8
    joint_speed: float = 0.6
    dwell: float = 0.3
    grip_time: float = 0.75

    def __post_init__(self) -> None:
        self.frames.setdefault("world", Frame("world"))

    # -- list protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.waypoints)

    def __iter__(self):
        return iter(self.waypoints)

    def __getitem__(self, index: int | slice) -> Waypoint | list[Waypoint]:
        if isinstance(index, slice):
            return self.waypoints[index]
        return self.waypoints[self._check(index)]

    # -- edit operations -------------------------------------------------

    def _check(self, index: int) -> int:
        """Validate an index, allowing Python-style negatives."""
        n = len(self.waypoints)
        if n == 0:
            raise ProgramError("the program has no waypoints")
        if not -n <= index < n:
            raise ProgramError(f"waypoint {index} is out of range (0..{n - 1})")
        return index % n

    def append(self, waypoint: Waypoint) -> int:
        """Add to the end. Returns the new index."""
        self.waypoints.append(waypoint)
        return len(self.waypoints) - 1

    def insert(self, index: int, waypoint: Waypoint) -> int:
        """Insert before ``index``, clamped into range. Returns the new index."""
        index = max(0, min(int(index), len(self.waypoints)))
        self.waypoints.insert(index, waypoint)
        return index

    def remove(self, index: int) -> Waypoint:
        """Delete and return the waypoint at ``index``."""
        return self.waypoints.pop(self._check(index))

    def replace(self, index: int, waypoint: Waypoint) -> None:
        """Overwrite the waypoint at ``index``."""
        self.waypoints[self._check(index)] = waypoint

    def duplicate(self, index: int) -> int:
        """Copy the waypoint at ``index`` in just after it. Returns the new index."""
        index = self._check(index)
        self.waypoints.insert(index + 1, self.waypoints[index].copy())
        return index + 1

    def move(self, index: int, to: int) -> int:
        """Move a waypoint to a new position. Returns where it ended up.

        ``to`` is clamped, so "move up" at the top is a no-op rather than a
        wrap-around — an operator holding a button should not see the first
        waypoint teleport to the end.
        """
        index = self._check(index)
        to = max(0, min(int(to), len(self.waypoints) - 1))
        waypoint = self.waypoints.pop(index)
        self.waypoints.insert(to, waypoint)
        return to

    def move_up(self, index: int) -> int:
        """Swap with the waypoint before it."""
        return self.move(index, self._check(index) - 1)

    def move_down(self, index: int) -> int:
        """Swap with the waypoint after it."""
        return self.move(index, self._check(index) + 1)

    def clear(self) -> None:
        """Drop every waypoint. Frames and settings are kept."""
        self.waypoints.clear()

    # -- frames ----------------------------------------------------------

    def frame(self, name: str) -> Frame:
        """Look up a named frame, defaulting to the world frame."""
        try:
            return self.frames[name]
        except KeyError as exc:
            raise ProgramError(
                f"waypoint references frame {name!r}, which this program does not define"
            ) from exc

    def add_frame(self, frame: Frame) -> None:
        """Define (or redefine) a named frame."""
        self.frames[frame.name] = frame

    def world_pose(self, index: int, arm: Arm) -> Pose:
        """A waypoint's pose for ``arm``, resolved into world coordinates."""
        waypoint = self[index]
        return self.frame(waypoint.frame).to_world(waypoint.pose(arm))

    # -- resolution ------------------------------------------------------

    def resolve(
        self,
        kinematics: CartesianKinematics,
        *,
        seed: np.ndarray | None = None,
        force: bool = False,
    ) -> list[int]:
        """Solve IK for any waypoint lacking a stored joint configuration.

        Each waypoint is seeded from the previous one's solution, so the whole
        program stays on one continuous branch of the null space instead of
        hopping between elbow configurations partway through.

        Args:
            kinematics: Solver to use.
            seed:       Seed for the first waypoint. Defaults to the rest pose.
            force:      Re-solve every waypoint, discarding stored joints.

        Returns:
            Indices that could **not** be solved — unreachable, or only
            reachable through the torso. They keep whatever they had.
        """
        q = kinematics.rest_q() if seed is None else np.asarray(seed, dtype=np.float32)
        failed: list[int] = []
        for i, waypoint in enumerate(self.waypoints):
            if waypoint.is_resolved and not force:
                q = waypoint.full_q(kinematics)
                continue
            targets = {arm: self.world_pose(i, arm) for arm in ARMS}
            solution = kinematics.solve(targets, seed=q, tool=self.tool)
            # ``usable``, not ``reached``: a pose the arm can only hit by
            # passing through the torso is not one to record or replay.
            if solution.usable:
                self.waypoints[i] = waypoint.resolved(solution.q, kinematics)
                q = solution.q
            else:
                failed.append(i)
        return failed

    # -- serialization ---------------------------------------------------

    def to_json(self) -> dict:
        """JSON-serialisable form."""
        return {
            "version": FORMAT_VERSION,
            "tool": {"name": self.tool.name, "offset": list(self.tool.offset)},
            "frames": [f.to_json() for f in self.frames.values()],
            "defaults": {
                "speed": self.speed,
                "ang_speed": self.ang_speed,
                "joint_speed": self.joint_speed,
                "dwell": self.dwell,
                "grip_time": self.grip_time,
            },
            # Labels are written verbatim. Substituting "waypoint 3" for an
            # empty one makes save/load non-idempotent and leaves a name that
            # lies the moment the program is reordered; the display fallback
            # belongs in the UI, not the file.
            "waypoints": [w.to_json() for w in self.waypoints],
        }

    def save(self, path: str | Path) -> None:
        """Write to ``path`` atomically, creating parent directories.

        Written to a temporary file and renamed, so an interrupted save cannot
        truncate a program that took real time to build.
        """
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        tmp.replace(p)

    @classmethod
    def from_json(
        cls, data: dict, kinematics: CartesianKinematics | None = None
    ) -> Program:
        """Build from parsed JSON, upgrading a version 1 file if needed."""
        version = int(data.get("version", FORMAT_VERSION))
        if version == LEGACY_VERSION:
            return cls._from_legacy(data, kinematics)
        if version != FORMAT_VERSION:
            raise ProgramError(
                f"this is a version {version} program file; this build reads "
                f"version {FORMAT_VERSION} (and upgrades version {LEGACY_VERSION})"
            )
        tool_data = data.get("tool") or {}
        tool = (
            Tool(str(tool_data["name"]), tuple(float(v) for v in tool_data["offset"]))
            if tool_data
            else GRIPPER_TIP
        )
        defaults = data.get("defaults") or {}
        program = cls(
            waypoints=[Waypoint.from_json(w) for w in data.get("waypoints", [])],
            tool=tool,
            frames={f["name"]: Frame.from_json(f) for f in data.get("frames", [])},
            speed=float(defaults.get("speed", 0.10)),
            ang_speed=float(defaults.get("ang_speed", 0.8)),
            joint_speed=float(defaults.get("joint_speed", 0.6)),
            dwell=float(defaults.get("dwell", 0.3)),
            grip_time=float(defaults.get("grip_time", 0.75)),
        )
        return program

    @classmethod
    def _from_legacy(
        cls, data: dict, kinematics: CartesianKinematics | None
    ) -> Program:
        """Upgrade an ``axol waypoints`` v1 file, recovering poses by FK."""
        if kinematics is None:
            raise ProgramError(
                "this is a version 1 waypoint file (joint angles only). Reading it "
                "needs a CartesianKinematics to recover the Cartesian poses — pass "
                "one to Program.load()."
            )
        # The caller's tool, not the default. On the gripperless SKU the tool
        # is the flange, and doing FK with GRIPPER_TIP instead would place every
        # upgraded waypoint 145 mm along the tool axis from where it belongs.
        program = cls(tool=kinematics.tool)
        for entry in data.get("waypoints", []):
            q = np.zeros(kinematics.num_joints, dtype=np.float32)
            grips = {}
            for arm in ARMS:
                values = np.asarray(entry[arm.value], dtype=np.float32)
                q[kinematics.indices(arm)] = values[:N_ARM_JOINTS]
                grips[arm] = float(values[N_ARM_JOINTS])
            poses = kinematics.fk(q, program.tool)
            program.append(
                Waypoint(
                    targets={
                        arm: ArmTarget(
                            pose=poses[arm],
                            joints=q[kinematics.indices(arm)],
                            grip=grips[arm],
                        )
                        for arm in ARMS
                    },
                    move=MoveType.LINEAR,
                    label=str(entry.get("label", "")),
                )
            )
        return program

    @classmethod
    def load(
        cls, path: str | Path, kinematics: CartesianKinematics | None = None
    ) -> Program:
        """Read a program file, returning an empty program if it does not exist."""
        p = Path(path).expanduser()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise ProgramError(f"{p} is not valid JSON: {exc}") from exc
        return cls.from_json(data, kinematics)
