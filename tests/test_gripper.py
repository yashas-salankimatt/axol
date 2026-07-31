"""Jaw geometry measured from Almond's published gripper CAD.

These pin the numbers taken out of ``gripper.step`` so a later edit to the
constants — or an articulated URDF built from the same measurements — cannot
silently disagree with the CAD or with the drawing.
"""

from __future__ import annotations

import numpy as np
import pytest

from almond_axol.constants import GRIPPER_TIP_OFFSET
from almond_axol.motion.frames import Arm, Pose

# float32 storage, and ``angle_to`` takes an arccos near 1 where the error is
# amplified to about sqrt(2*eps) ~ 5e-4 rad. Same floor as tests/test_frames.py.
POS_TOL = 1e-7
ROT_TOL = 1e-3

from almond_axol.motion.gripper import (
    CLOSED_GAP,
    JAW_AXIS,
    JAW_DEPTH,
    JAW_HEIGHT,
    JAW_THICKNESS,
    MAX_TRAVEL,
    OPEN_GAP,
    gap_for,
    jaws,
)


def test_matches_the_published_drawing():
    """Overall envelope has to come back to the 119.8 x 68 mm on the drawing."""
    overall = OPEN_GAP + 2 * JAW_THICKNESS
    assert overall == pytest.approx(0.1198, abs=5e-5)
    assert JAW_DEPTH == pytest.approx(0.068, abs=5e-5)


def test_travel_is_half_the_opening():
    assert MAX_TRAVEL == pytest.approx((OPEN_GAP - CLOSED_GAP) / 2)
    assert MAX_TRAVEL == pytest.approx(0.0362, abs=1e-4)


def test_gap_is_linear_and_clamped():
    assert gap_for(0.0) == pytest.approx(CLOSED_GAP)
    assert gap_for(1.0) == pytest.approx(OPEN_GAP)
    assert gap_for(0.5) == pytest.approx((OPEN_GAP + CLOSED_GAP) / 2)
    assert gap_for(-3.0) == pytest.approx(CLOSED_GAP)
    assert gap_for(9.0) == pytest.approx(OPEN_GAP)


@pytest.mark.parametrize("opening", [0.0, 0.25, 0.5, 1.0])
def test_jaws_are_symmetric_about_the_tool_axis(opening):
    left, right = jaws(Pose.identity(), opening)
    assert left.pose.position[JAW_AXIS] == pytest.approx(
        -right.pose.position[JAW_AXIS], abs=POS_TOL
    )
    for axis in range(3):
        if axis != JAW_AXIS:
            assert left.pose.position[axis] == pytest.approx(
                right.pose.position[axis], abs=POS_TOL
            )


@pytest.mark.parametrize("opening", [0.0, 0.4, 1.0])
def test_inner_faces_are_exactly_the_commanded_gap_apart(opening):
    left, right = jaws(Pose.identity(), opening)
    inner_left = left.pose.position[JAW_AXIS] + JAW_THICKNESS / 2
    inner_right = right.pose.position[JAW_AXIS] - JAW_THICKNESS / 2
    assert inner_right - inner_left == pytest.approx(gap_for(opening), abs=POS_TOL)


def test_jaws_reach_the_fingertip_plane():
    """The jaw tips must land on GRIPPER_TIP_OFFSET, which is what IK targets."""
    for jaw in jaws(Pose.identity(), 1.0):
        tip = jaw.pose.position[2] - JAW_HEIGHT / 2
        assert tip == pytest.approx(GRIPPER_TIP_OFFSET[2], abs=POS_TOL)


def test_jaws_follow_the_gripper_link_pose():
    """Jaws are placed in the link frame, so they ride the wrist."""
    moved = Pose.from_rpy((0.3, -0.2, 0.5), 0.4, -0.3, 1.2)
    at_origin = jaws(Pose.identity(), 0.6)
    at_pose = jaws(moved, 0.6)
    for a, b in zip(at_origin, at_pose):
        expected = moved * Pose(a.pose.position, np.eye(3, dtype=np.float32))
        assert b.pose.position == pytest.approx(expected.position, abs=1e-5)
        assert b.pose.angle_to(moved) < ROT_TOL


def test_closing_moves_both_jaws_inward():
    open_l, open_r = jaws(Pose.identity(), 1.0)
    shut_l, shut_r = jaws(Pose.identity(), 0.0)
    assert abs(shut_l.pose.position[JAW_AXIS]) < abs(open_l.pose.position[JAW_AXIS])
    assert abs(shut_r.pose.position[JAW_AXIS]) < abs(open_r.pose.position[JAW_AXIS])
    travel = abs(open_l.pose.position[JAW_AXIS] - shut_l.pose.position[JAW_AXIS])
    assert travel == pytest.approx(MAX_TRAVEL, abs=POS_TOL)


def test_dimensions_are_positive_and_axis_aligned():
    for jaw in jaws(Pose.identity(), 0.5):
        assert all(d > 0 for d in jaw.dimensions)
        assert jaw.dimensions[JAW_AXIS] == pytest.approx(JAW_THICKNESS)
        assert jaw.dimensions[2] == pytest.approx(JAW_HEIGHT)


def test_arm_argument_is_accepted():
    """Both arms carry the same gripper today; the argument is a seam."""
    assert len(jaws(Pose.identity(), 0.5, Arm.LEFT)) == 2
    assert len(jaws(Pose.identity(), 0.5, Arm.RIGHT)) == 2
