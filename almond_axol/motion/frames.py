"""Poses, frames, and tool offsets — the coordinate layer everything else sits on.

Every Cartesian API in :mod:`almond_axol.motion` speaks :class:`Pose`, and every
:class:`Pose` is expressed in the **world frame** unless it is explicitly said to
be in a named :class:`Frame`. Getting this layer wrong is the single most
expensive mistake available here — a sign error on an axis is an arm swinging
into a fixture — so the frame is documented from *measurement*, not from CAD
intent, and the measurements are pinned by tests.

World frame
-----------

The world frame is the URDF root (link ``base``). Measured by forward
kinematics on the bundled URDF (see ``tests/test_frames.py``, which fails if
the URDF ever changes underneath these numbers):

- ``+X`` points toward the arm named **left**. The shoulders sit at
  ``(+0.13, 0, 0.86)`` and ``(-0.13, 0, 0.86)``.
- ``+Z`` points **up**. The arms hang to ``z ≈ 0`` at the zero configuration,
  and the shoulders are 0.86 m above the origin.
- ``+Y`` completes the right-handed frame, which — given ``+X`` is the left
  side and ``+Z`` is up — puts ``+Y`` **behind** the robot. Equivalently,
  :data:`FORWARD` is ``-Y``. Flexing ``shoulder_1`` or ``elbow`` positively
  swings the tip toward ``-Y``, which is the direction an arm reaches into its
  own workspace, and that corroborates the handedness argument.

.. warning::
   Docstrings elsewhere in this package describe the world frame as "FLU"
   (Forward-Left-Up). That is **wrong**: under FLU the two arms would be
   separated along ``Y``, and they are measurably separated along ``X``. It
   never caused a bug because nothing before this module commanded an absolute
   world pose — VR teleop works entirely in deltas rotated into the current FK
   frame (see :func:`almond_axol.teleop.worker._relative_target_np`), so the
   absolute convention cancels out.

.. note::
   Which physical side of the robot is "front" is a mounting convention that
   cannot be read off a URDF. ``-Y`` is where the arms flex to, which is the
   only defensible answer from geometry alone. Confirm against the physical
   robot before trusting :data:`FORWARD` for anything that could collide.

Tool frames
-----------

The URDF chain ends at the gripper *mount*; the fingers close 145 mm further
along that link's ``-Z`` (:data:`~almond_axol.constants.GRIPPER_TIP_OFFSET`).
Which of the two you mean matters: holding the mount on a straight line swings
the tip through an arc as the wrist reorients. :class:`Tool` names the choice,
and every Cartesian call takes one, so it is never implicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from ..constants import GRIPPER_TIP_OFFSET

# ---------------------------------------------------------------------------
# World frame
# ---------------------------------------------------------------------------

AXIS_LEFT: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32)
"""Unit vector toward the left arm (``+X``)."""

AXIS_RIGHT: np.ndarray = -AXIS_LEFT
"""Unit vector toward the right arm (``-X``)."""

AXIS_BACKWARD: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32)
"""Unit vector behind the robot (``+Y``). See the module docstring."""

AXIS_FORWARD: np.ndarray = -AXIS_BACKWARD
"""Unit vector out in front of the robot (``-Y``). See the module docstring."""

AXIS_UP: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
"""Unit vector up (``+Z``)."""

AXIS_DOWN: np.ndarray = -AXIS_UP
"""Unit vector down (``-Z``)."""

AXIS_LABELS: tuple[str, str, str] = ("X (left+)", "Y (back+)", "Z (up+)")
"""Human-facing axis names, for UI readouts that must not be ambiguous."""


class Arm(Enum):
    """Which arm a Cartesian command is aimed at."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def is_left(self) -> bool:
        """True for :attr:`LEFT` — matches the ``is_left`` keyword used across the SDK."""
        return self is Arm.LEFT

    @property
    def label(self) -> str:
        """Capitalised name for UI headings."""
        return self.value.capitalize()


ARMS: tuple[Arm, Arm] = (Arm.LEFT, Arm.RIGHT)
"""Both arms in the order every ``(left, right)`` tuple in the SDK uses."""


# ---------------------------------------------------------------------------
# Tool frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """A tool centre point, expressed in the gripper *mount* link frame.

    This is the ``TCP`` of an industrial controller. ``offset`` is where the
    working point sits relative to the mount; ``name`` is what the UI calls it.

    Attributes:
        name:   Operator-facing label.
        offset: ``(3,)`` translation in the gripper link frame, metres.
    """

    name: str
    offset: tuple[float, float, float]

    @property
    def vector(self) -> np.ndarray:
        """The offset as a ``(3,)`` float32 array."""
        return np.asarray(self.offset, dtype=np.float32)


FLANGE = Tool("flange", (0.0, 0.0, 0.0))
"""The gripper mount itself — what the IK solver natively targets."""

GRIPPER_TIP = Tool("gripper_tip", GRIPPER_TIP_OFFSET)
"""Where the fingers close. The default for a gripper-equipped arm."""


def default_tool(*, has_gripper: bool) -> Tool:
    """The tool to use when the caller has not chosen one.

    A gripperless SKU has nothing beyond the mount, so holding the mount to the
    path is both correct and what ``axol waypoints`` already does.
    """
    return GRIPPER_TIP if has_gripper else FLANGE


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _normalize_quat(wxyz: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(wxyz))
    if n < 1e-12:
        raise ValueError("quaternion has zero norm")
    return (wxyz / n).astype(np.float32)


def quat_to_matrix(wxyz: ArrayLike) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a ``(3, 3)`` rotation matrix."""
    w, x, y, z = _normalize_quat(np.asarray(wxyz, dtype=np.float64))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def matrix_to_quat(rotation: ArrayLike) -> np.ndarray:
    """Convert a ``(3, 3)`` rotation matrix to a ``(w, x, y, z)`` quaternion.

    Shepperd's method: pick the branch whose divisor is largest so the result
    stays conditioned near each of the four degenerate cases.
    """
    m = np.asarray(rotation, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        r = math.sqrt(trace + 1.0)
        s = 0.5 / r
        q = (
            0.5 * r,
            (m[2, 1] - m[1, 2]) * s,
            (m[0, 2] - m[2, 0]) * s,
            (m[1, 0] - m[0, 1]) * s,
        )
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        r = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        s = 0.5 / r
        q = (
            (m[2, 1] - m[1, 2]) * s,
            0.5 * r,
            (m[0, 1] + m[1, 0]) * s,
            (m[0, 2] + m[2, 0]) * s,
        )
    elif m[1, 1] >= m[2, 2]:
        r = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        s = 0.5 / r
        q = (
            (m[0, 2] - m[2, 0]) * s,
            (m[0, 1] + m[1, 0]) * s,
            0.5 * r,
            (m[1, 2] + m[2, 1]) * s,
        )
    else:
        r = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        s = 0.5 / r
        q = (
            (m[1, 0] - m[0, 1]) * s,
            (m[0, 2] + m[2, 0]) * s,
            (m[1, 2] + m[2, 1]) * s,
            0.5 * r,
        )
    out = np.asarray(q, dtype=np.float64)
    # Canonical sign: w >= 0, so equal rotations compare equal elementwise.
    if out[0] < 0.0:
        out = -out
    return _normalize_quat(out)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert intrinsic X-Y-Z Euler angles (rad) to a rotation matrix.

    The composition is ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, the same
    convention URDF ``rpy`` attributes and industrial teach pendants use, so a
    number typed into the UI means what an operator expects.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def matrix_to_rpy(rotation: ArrayLike) -> tuple[float, float, float]:
    """Inverse of :func:`rpy_to_matrix`, returning ``(roll, pitch, yaw)`` in rad.

    At gimbal lock (``|pitch| = π/2``) roll and yaw are not separable; roll is
    pinned to zero and the whole rotation is reported as yaw, which is the
    conventional choice and keeps the round trip exact.
    """
    m = np.asarray(rotation, dtype=np.float64)
    sp = -m[2, 0]
    if sp > 1.0 - 1e-9:
        return (0.0, math.pi / 2, math.atan2(-m[0, 1], m[1, 1]))
    if sp < -1.0 + 1e-9:
        return (0.0, -math.pi / 2, math.atan2(-m[0, 1], m[1, 1]))
    return (
        math.atan2(m[2, 1], m[2, 2]),
        math.asin(max(-1.0, min(1.0, sp))),
        math.atan2(m[1, 0], m[0, 0]),
    )


def rotation_angle(a: ArrayLike, b: ArrayLike) -> float:
    """Angle (rad) between two rotation matrices."""
    m = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    return float(np.arccos(max(-1.0, min(1.0, (float(np.trace(m)) - 1.0) * 0.5))))


def orthonormalize(rotation: ArrayLike) -> np.ndarray:
    """Snap a nearly-orthonormal matrix back onto SO(3).

    Rotations that have been round-tripped through a UI, a JSON file, or a
    float32 gizmo drift off the manifold by parts in 1e-7. Left alone that
    drift accumulates into a matrix the IK cost treats as a scale as well as a
    rotation. The nearest rotation in the Frobenius sense is ``U @ Vᵀ`` from
    the SVD, with the determinant forced positive so a reflection can never
    come back.
    """
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    m = u @ vt
    if np.linalg.det(m) < 0.0:
        u[:, -1] *= -1.0
        m = u @ vt
    return m.astype(np.float32)


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """A rigid transform: where a frame sits, and how it is turned.

    Immutable on purpose — poses are passed into planners, stored in programs,
    and held by UI widgets, and an in-place rotation edit that silently changed
    a stored waypoint would be very hard to see.

    Attributes:
        position: ``(3,)`` metres.
        rotation: ``(3, 3)`` rotation matrix.
    """

    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        # ``np.asarray`` is a no-op view when the input is already float32, so
        # a Pose built from a caller's array would alias it: mutating theirs
        # afterwards would silently rewrite a stored waypoint. Copy, then mark
        # read-only so the immutability in the docstring is enforced rather
        # than merely intended — including for views handed out by
        # ``rotation.T`` in :meth:`inverse`.
        pos = np.array(self.position, dtype=np.float32).reshape(-1)
        rot = np.array(self.rotation, dtype=np.float32)
        if pos.shape != (3,):
            raise ValueError(f"position must be (3,), got {pos.shape}")
        if rot.shape != (3, 3):
            raise ValueError(f"rotation must be (3, 3), got {rot.shape}")
        pos.flags.writeable = False
        rot.flags.writeable = False
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "rotation", rot)

    # -- constructors ----------------------------------------------------

    @classmethod
    def identity(cls) -> Pose:
        """The world origin, unrotated."""
        return cls(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))

    @classmethod
    def from_quat(cls, position: ArrayLike, wxyz: ArrayLike) -> Pose:
        """Build from a position and a ``(w, x, y, z)`` quaternion."""
        return cls(np.asarray(position, dtype=np.float32), quat_to_matrix(wxyz))

    @classmethod
    def from_rpy(
        cls, position: ArrayLike, roll: float, pitch: float, yaw: float
    ) -> Pose:
        """Build from a position and intrinsic X-Y-Z Euler angles (rad)."""
        return cls(
            np.asarray(position, dtype=np.float32), rpy_to_matrix(roll, pitch, yaw)
        )

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> Pose:
        """Build from a ``(4, 4)`` homogeneous transform."""
        m = np.asarray(matrix, dtype=np.float32)
        if m.shape != (4, 4):
            raise ValueError(f"matrix must be (4, 4), got {m.shape}")
        return cls(m[:3, 3], m[:3, :3])

    # -- accessors -------------------------------------------------------

    @property
    def quat(self) -> np.ndarray:
        """Orientation as a ``(w, x, y, z)`` quaternion."""
        return matrix_to_quat(self.rotation)

    @property
    def rpy(self) -> tuple[float, float, float]:
        """Orientation as intrinsic X-Y-Z Euler angles (rad)."""
        return matrix_to_rpy(self.rotation)

    @property
    def matrix(self) -> np.ndarray:
        """The ``(4, 4)`` homogeneous transform."""
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = self.rotation
        out[:3, 3] = self.position
        return out

    # -- algebra ---------------------------------------------------------

    def __mul__(self, other: Pose) -> Pose:
        """Compose: ``self`` then ``other``, i.e. ``other`` expressed in ``self``."""
        return Pose(
            self.position + self.rotation @ other.position,
            orthonormalize(self.rotation @ other.rotation),
        )

    def inverse(self) -> Pose:
        """The transform that undoes this one."""
        rt = self.rotation.T
        return Pose(-(rt @ self.position), rt)

    def translated(self, delta: ArrayLike) -> Pose:
        """Copy shifted by ``delta`` in the **world** frame."""
        return replace(
            self, position=self.position + np.asarray(delta, dtype=np.float32)
        )

    def translated_local(self, delta: ArrayLike) -> Pose:
        """Copy shifted by ``delta`` expressed in **this pose's own** frame.

        This is the "tool frame" jog of a teach pendant: pressing +Z moves
        along the tool's approach direction wherever the wrist happens to be
        pointing, not along world up.
        """
        return replace(
            self,
            position=self.position
            + self.rotation @ np.asarray(delta, dtype=np.float32),
        )

    def rotated(self, roll: float, pitch: float, yaw: float) -> Pose:
        """Copy turned by an X-Y-Z Euler delta applied in the **world** frame."""
        return replace(
            self,
            rotation=orthonormalize(rpy_to_matrix(roll, pitch, yaw) @ self.rotation),
        )

    def rotated_local(self, roll: float, pitch: float, yaw: float) -> Pose:
        """Copy turned by an X-Y-Z Euler delta applied in **this pose's** frame."""
        return replace(
            self,
            rotation=orthonormalize(self.rotation @ rpy_to_matrix(roll, pitch, yaw)),
        )

    def apply_tool(self, tool: Tool) -> Pose:
        """Where ``tool``'s working point sits, given this is the mount pose."""
        return replace(self, position=self.position + self.rotation @ tool.vector)

    def remove_tool(self, tool: Tool) -> Pose:
        """The mount pose that puts ``tool``'s working point here.

        The exact inverse of :meth:`apply_tool` — this is what converts an
        operator's "put the fingertips there" into the frame the IK solver
        actually targets.
        """
        return replace(self, position=self.position - self.rotation @ tool.vector)

    # -- comparison ------------------------------------------------------

    def distance_to(self, other: Pose) -> float:
        """Straight-line distance (m) between the two positions."""
        return float(np.linalg.norm(self.position - other.position))

    def angle_to(self, other: Pose) -> float:
        """Angle (rad) between the two orientations."""
        return rotation_angle(self.rotation, other.rotation)

    def is_close(
        self, other: Pose, *, pos_tol: float = 1e-4, ori_tol: float = 1e-3
    ) -> bool:
        """True if both position and orientation are within tolerance."""
        return self.distance_to(other) <= pos_tol and self.angle_to(other) <= ori_tol

    # -- serialization ---------------------------------------------------

    def to_json(self) -> dict:
        """JSON-serialisable form: position plus quaternion.

        Stored as a quaternion rather than nine matrix entries because a
        hand-edited file should not be able to express a non-rotation, and
        because a quaternion survives a round trip without drifting off SO(3).
        """
        return {
            "position": [round(float(v), 6) for v in self.position],
            "quaternion": [round(float(v), 8) for v in self.quat],
        }

    @classmethod
    def from_json(cls, data: dict) -> Pose:
        """Inverse of :meth:`to_json`."""
        return cls.from_quat(data["position"], data["quaternion"])

    def __repr__(self) -> str:
        x, y, z = (float(v) for v in self.position)
        roll, pitch, yaw = (math.degrees(v) for v in self.rpy)
        return (
            f"Pose(xyz=({x:+.4f}, {y:+.4f}, {z:+.4f}) m, "
            f"rpy=({roll:+.1f}, {pitch:+.1f}, {yaw:+.1f})°)"
        )


# ---------------------------------------------------------------------------
# Named user frames
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """A named coordinate system, expressed in the world frame.

    The "feature" of a UR program or the "user coordinate system" of an xArm:
    teach a fixture once, express every waypoint relative to it, and re-teaching
    the fixture moves the whole program. Nothing forces you to use one — the
    default :data:`WORLD` is the identity — but a machine-tending program that
    hard-codes world coordinates has to be retaught from scratch when the CNC
    gets nudged, and one written against a fixture frame does not.
    """

    name: str
    pose: Pose = field(default_factory=Pose.identity)

    def to_world(self, local: Pose) -> Pose:
        """Convert a pose expressed in this frame into the world frame."""
        return self.pose * local

    def from_world(self, world: Pose) -> Pose:
        """Convert a world-frame pose into this frame."""
        return self.pose.inverse() * world

    def to_json(self) -> dict:
        """JSON-serialisable form."""
        return {"name": self.name, "pose": self.pose.to_json()}

    @classmethod
    def from_json(cls, data: dict) -> Frame:
        """Inverse of :meth:`to_json`."""
        return cls(name=str(data["name"]), pose=Pose.from_json(data["pose"]))


WORLD = Frame("world")
"""The identity frame — the URDF root. See the module docstring for its axes."""
