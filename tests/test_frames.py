"""Pose algebra, conversions, and the measured world-frame convention.

The world-frame tests are the important ones. They encode measurements taken
from the bundled URDF, and they exist so that a URDF change which silently
moves or reorients the robot fails here instead of on a real arm.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from almond_axol.motion.frames import (
    AXIS_FORWARD,
    AXIS_LEFT,
    AXIS_UP,
    FLANGE,
    GRIPPER_TIP,
    Arm,
    Frame,
    Pose,
    Tool,
    default_tool,
    matrix_to_quat,
    matrix_to_rpy,
    orthonormalize,
    quat_to_matrix,
    rotation_angle,
    rpy_to_matrix,
)

RNG = np.random.default_rng(20260730)

# Poses are stored in float32, and ``rotation_angle`` takes an arccos of a value
# near 1, where the derivative is infinite: an elementwise error of eps gives an
# angle error of about sqrt(2*eps). At float32 that floor is ~5e-4 rad (0.03°),
# so a "round trip is exact" assertion has to be written against that and not
# against float64 intuition.
ROT_TOL = 1e-3
POS_TOL = 1e-5


def random_rotation() -> np.ndarray:
    """A uniformly random rotation matrix, via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(RNG.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def random_pose() -> Pose:
    return Pose(RNG.uniform(-1.0, 1.0, 3).astype(np.float32), random_rotation())


# ---------------------------------------------------------------------------
# Rotation conversions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("_", range(25))
def test_quat_matrix_round_trip(_):
    rot = random_rotation()
    assert rotation_angle(rot, quat_to_matrix(matrix_to_quat(rot))) < ROT_TOL


@pytest.mark.parametrize("_", range(25))
def test_rpy_matrix_round_trip(_):
    rot = random_rotation()
    assert rotation_angle(rot, rpy_to_matrix(*matrix_to_rpy(rot))) < ROT_TOL


def test_rpy_matches_urdf_convention():
    """``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` — what URDF and teach pendants use."""
    roll, pitch, yaw = 0.3, -0.4, 1.1

    def rx(a):
        return np.array(
            [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
        )

    def ry(a):
        return np.array(
            [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        )

    def rz(a):
        return np.array(
            [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        )

    assert (
        rotation_angle(rpy_to_matrix(roll, pitch, yaw), rz(yaw) @ ry(pitch) @ rx(roll))
        < ROT_TOL
    )


def test_matrix_to_rpy_at_gimbal_lock():
    """Pitch = ±90° must still round-trip as a rotation, even though rpy is not unique."""
    for pitch in (math.pi / 2, -math.pi / 2):
        rot = rpy_to_matrix(0.0, pitch, 0.7)
        assert rotation_angle(rot, rpy_to_matrix(*matrix_to_rpy(rot))) < ROT_TOL


def test_matrix_to_quat_near_pi():
    """The Shepperd branch for a 180° rotation must not lose the axis."""
    for axis in (AXIS_LEFT, np.array([0.0, 1.0, 0.0]), AXIS_UP):
        rot = _axis_angle(axis, math.pi - 1e-7)
        assert rotation_angle(rot, quat_to_matrix(matrix_to_quat(rot))) < 1e-3


def _axis_angle(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return (np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)).astype(
        np.float32
    )


def test_quat_canonical_sign():
    """``q`` and ``-q`` are the same rotation; the canonical form has ``w >= 0``."""
    for _ in range(10):
        assert matrix_to_quat(random_rotation())[0] >= 0.0


def test_orthonormalize_repairs_drift():
    rot = random_rotation().astype(np.float64) + RNG.normal(scale=1e-3, size=(3, 3))
    fixed = orthonormalize(rot)
    assert np.allclose(fixed.T @ fixed, np.eye(3), atol=1e-5)
    assert np.linalg.det(fixed) > 0


def test_orthonormalize_rejects_reflection():
    """A reflection must come back as a rotation, not a flipped frame."""
    reflection = np.diag([1.0, 1.0, -1.0])
    assert np.linalg.det(orthonormalize(reflection)) > 0


# ---------------------------------------------------------------------------
# Pose algebra
# ---------------------------------------------------------------------------


def test_pose_validates_shapes():
    with pytest.raises(ValueError):
        Pose(np.zeros(2), np.eye(3))
    with pytest.raises(ValueError):
        Pose(np.zeros(3), np.eye(4))


def test_inverse_round_trip():
    for _ in range(10):
        pose = random_pose()
        identity = pose * pose.inverse()
        assert identity.distance_to(Pose.identity()) < 1e-4
        assert identity.angle_to(Pose.identity()) < ROT_TOL


def test_compose_is_associative():
    a, b, c = random_pose(), random_pose(), random_pose()
    left, right = (a * b) * c, a * (b * c)
    assert left.distance_to(right) < 1e-4
    assert left.angle_to(right) < ROT_TOL


def test_matrix_round_trip():
    pose = random_pose()
    assert Pose.from_matrix(pose.matrix).is_close(
        pose, pos_tol=POS_TOL, ori_tol=ROT_TOL
    )


def test_translated_is_world_frame():
    """World jog moves along world axes regardless of how the tool is turned."""
    pose = Pose(np.zeros(3, dtype=np.float32), rpy_to_matrix(0.0, 0.0, math.pi / 2))
    moved = pose.translated((0.1, 0.0, 0.0))
    assert np.allclose(moved.position, [0.1, 0.0, 0.0], atol=1e-6)


def test_translated_local_is_tool_frame():
    """Tool jog follows the tool's own axes — yawed 90°, +X becomes world +Y."""
    pose = Pose(np.zeros(3, dtype=np.float32), rpy_to_matrix(0.0, 0.0, math.pi / 2))
    moved = pose.translated_local((0.1, 0.0, 0.0))
    assert np.allclose(moved.position, [0.0, 0.1, 0.0], atol=1e-6)


def test_world_rotation_turns_about_the_world_axis():
    """Pins the frame, not the magnitude.

    A magnitude-only assertion is satisfied by ``rotated`` and ``rotated_local``
    being swapped — the operator's "Jog frame" selector would then do the exact
    opposite of what it says, next to a fixture. ``Rz @ M`` leaves each tool
    axis's *world-Z component* untouched, which is a statement only the world
    form satisfies.
    """
    pose = random_pose()
    turned = pose.rotated(0.0, 0.0, 0.4)
    assert np.allclose(turned.rotation[2, :], pose.rotation[2, :], atol=1e-5)


def test_tool_rotation_turns_about_the_tool_axis():
    """``M @ Rz`` leaves the tool's own Z axis fixed in world coordinates."""
    pose = random_pose()
    turned = pose.rotated_local(0.0, 0.0, 0.4)
    assert np.allclose(turned.rotation[:, 2], pose.rotation[:, 2], atol=1e-5)


def test_world_and_tool_rotation_are_not_interchangeable():
    """Guards directly against the two implementations being swapped."""
    pose = random_pose()
    assert not np.allclose(
        pose.rotated(0.0, 0.0, 0.4).rotation,
        pose.rotated_local(0.0, 0.0, 0.4).rotation,
        atol=1e-3,
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_world_rotation_uses_the_named_axis_and_sign(axis, sign):
    """The delta rotation must be about +/- that world axis, not another."""
    pose = random_pose()
    rpy = [0.0, 0.0, 0.0]
    rpy[axis] = sign * 0.25
    delta = (
        pose.rotated(*rpy).rotation.astype(np.float64)
        @ pose.rotation.astype(np.float64).T
    )
    expected = np.zeros(3)
    expected[axis] = sign * 0.25
    assert np.allclose(_log(delta), expected, atol=2e-3)


def _log(rotation):
    """Axis-angle vector of a rotation matrix."""
    cos = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    theta = math.acos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    return (
        np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
        * theta
        / (2.0 * math.sin(theta))
    )


def test_pose_arrays_are_not_aliased_to_the_caller():
    """``Pose`` is documented immutable; that has to hold against the caller too."""
    position = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    rotation = np.eye(3, dtype=np.float32)
    pose = Pose(position, rotation)
    position[0] = 99.0
    rotation[0, 0] = 5.0
    assert pose.position[0] == pytest.approx(0.1)
    assert pose.rotation[0, 0] == pytest.approx(1.0)


def test_pose_arrays_are_read_only():
    pose = random_pose()
    with pytest.raises(ValueError):
        pose.position[0] = 1.0
    with pytest.raises(ValueError):
        pose.rotation[0, 0] = 1.0


def test_inverse_does_not_share_a_buffer_with_the_original():
    pose = random_pose()
    assert not np.shares_memory(pose.inverse().rotation, pose.rotation) or (
        not pose.inverse().rotation.flags.writeable
    )


def test_from_matrix_does_not_alias_the_matrix():
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = (0.1, 0.2, 0.3)
    pose = Pose.from_matrix(matrix)
    matrix[:3, 3] = (9.0, 9.0, 9.0)
    assert np.allclose(pose.position, [0.1, 0.2, 0.3])


def test_local_and_world_rotation_differ_and_both_preserve_position():
    pose = random_pose()
    world = pose.rotated(0.2, 0.0, 0.0)
    local = pose.rotated_local(0.2, 0.0, 0.0)
    assert np.allclose(world.position, pose.position)
    assert np.allclose(local.position, pose.position)
    assert world.angle_to(local) > 1e-3
    # Both turn by the same amount, just about different axes.
    assert abs(world.angle_to(pose) - local.angle_to(pose)) < 1e-5


def test_repeated_local_rotation_stays_on_so3():
    """Drift must not accumulate — this is what jogging does hundreds of times."""
    pose = random_pose()
    for _ in range(500):
        pose = pose.rotated_local(0.01, 0.02, -0.015)
    rot = pose.rotation.astype(np.float64)
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_tool_apply_remove_is_identity():
    pose = random_pose()
    assert (
        pose.apply_tool(GRIPPER_TIP)
        .remove_tool(GRIPPER_TIP)
        .is_close(pose, pos_tol=POS_TOL, ori_tol=ROT_TOL)
    )


def test_flange_tool_is_a_no_op():
    pose = random_pose()
    assert pose.apply_tool(FLANGE).is_close(pose, pos_tol=POS_TOL, ori_tol=ROT_TOL)


def test_tool_offset_is_along_the_tool_axis():
    """The tip sits 145 mm along the gripper link's -Z, wherever the wrist points."""
    pose = Pose(np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))
    assert np.allclose(pose.apply_tool(GRIPPER_TIP).position, [0, 0, -0.145], atol=1e-6)
    turned = Pose(np.zeros(3, dtype=np.float32), rpy_to_matrix(math.pi / 2, 0.0, 0.0))
    # Rolled 90° about X, the tool's -Z points along world +Y.
    assert np.allclose(
        turned.apply_tool(GRIPPER_TIP).position, [0, 0.145, 0], atol=1e-6
    )


def test_default_tool_follows_the_sku():
    assert default_tool(has_gripper=True) is GRIPPER_TIP
    assert default_tool(has_gripper=False) is FLANGE


# ---------------------------------------------------------------------------
# Named frames
# ---------------------------------------------------------------------------


def test_frame_round_trip():
    fixture = Frame("fixture", random_pose())
    local = random_pose()
    assert fixture.from_world(fixture.to_world(local)).is_close(
        local, pos_tol=1e-4, ori_tol=ROT_TOL
    )


def test_frame_moves_everything_with_it():
    """Re-teaching a fixture must move its waypoints, which is the whole point."""
    local = Pose(
        np.array([0.1, 0.0, 0.0], dtype=np.float32), np.eye(3, dtype=np.float32)
    )
    before = Frame("f", Pose.identity()).to_world(local)
    after = Frame(
        "f",
        Pose(np.array([0.0, 0.0, 0.5], dtype=np.float32), np.eye(3, dtype=np.float32)),
    ).to_world(local)
    assert np.allclose(after.position - before.position, [0, 0, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_pose_json_round_trip():
    for _ in range(10):
        pose = random_pose()
        assert Pose.from_json(pose.to_json()).is_close(
            pose, pos_tol=POS_TOL, ori_tol=ROT_TOL
        )


def test_frame_json_round_trip():
    frame = Frame("cnc_vise", random_pose())
    restored = Frame.from_json(frame.to_json())
    assert restored.name == "cnc_vise"
    assert restored.pose.is_close(frame.pose, pos_tol=POS_TOL, ori_tol=ROT_TOL)


def test_tool_json_shape():
    tool = Tool("probe", (0.0, 0.0, -0.2))
    assert tool.vector.tolist() == pytest.approx([0.0, 0.0, -0.2])


# ---------------------------------------------------------------------------
# Arm enum
# ---------------------------------------------------------------------------


def test_arm_is_left_matches_sdk_keyword():
    assert Arm.LEFT.is_left is True
    assert Arm.RIGHT.is_left is False


# ---------------------------------------------------------------------------
# The measured world frame — regression guards on the URDF itself
# ---------------------------------------------------------------------------


def test_axis_constants_are_right_handed():
    """X × Y = Z, so the documented directions actually form a frame."""
    y = np.cross(AXIS_UP, AXIS_LEFT)
    assert np.allclose(np.cross(AXIS_LEFT, y), AXIS_UP, atol=1e-6)
    assert np.allclose(AXIS_FORWARD, -y, atol=1e-6)


def test_shoulders_are_separated_along_x_not_y(kin):
    """The load-bearing measurement: the frame is X-lateral, **not** FLU.

    Under FLU the arms would differ in Y. They differ in X, by ±0.13 m, and sit
    0.86 m up. If this ever fails, every Cartesian pose in every saved program
    is being interpreted in the wrong frame.
    """
    left = kin.shoulder_position(Arm.LEFT)
    right = kin.shoulder_position(Arm.RIGHT)
    assert left[0] == pytest.approx(0.13, abs=1e-3)
    assert right[0] == pytest.approx(-0.13, abs=1e-3)
    assert abs(left[1] - right[1]) < 1e-6, "arms must not be separated along Y"
    assert left[2] == pytest.approx(0.86, abs=1e-3)
    assert right[2] == pytest.approx(0.86, abs=1e-3)


def test_left_arm_is_on_the_positive_x_side(kin, q_rest):
    poses = kin.fk(q_rest)
    assert poses[Arm.LEFT].position[0] > 0.0
    assert poses[Arm.RIGHT].position[0] < 0.0


def test_positive_shoulder_flex_reaches_forward(kin, q_rest):
    """+shoulder_1 swings the tool toward -Y, which is why -Y is documented as forward."""
    q = q_rest.copy()
    q[kin.indices(Arm.LEFT)[0]] += 0.8
    before = kin.fk(q_rest)[Arm.LEFT].position
    after = kin.fk(q)[Arm.LEFT].position
    assert after[1] < before[1], "shoulder flex should move the tool toward -Y"


def test_arms_hang_down_at_zero(kin):
    """At the zero configuration the tools sit near the floor, not up at the shoulder."""
    poses = kin.fk(np.zeros(kin.num_joints, dtype=np.float32))
    for arm in (Arm.LEFT, Arm.RIGHT):
        assert abs(poses[arm].position[2]) < 0.05
