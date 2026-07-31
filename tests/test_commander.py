"""Planning and execution against a virtual robot."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from almond_axol.motion import (
    ARMS,
    Arm,
    MotionCommander,
    MoveType,
    Program,
    VirtualRobot,
)
from almond_axol.motion.commander import PlannedLeg, ease_in_offsets
from almond_axol.motion.program import ArmTarget, Waypoint

# The suite streams whole trajectories; a fast rate keeps that to milliseconds
# without changing any of the logic under test.
TEST_RATE = 50.0


@pytest.fixture
def robot():
    return VirtualRobot()


@pytest.fixture
def commander(kin, robot):
    return MotionCommander(kin, robot, rate_hz=TEST_RATE)


def waypoint_at(kin, q, offset=(0.0, 0.0, 0.0), grip=1.0):
    poses = kin.fk(q)
    return Waypoint(
        targets={
            arm: ArmTarget(
                pose=poses[arm].translated(offset),
                joints=None,
                grip=grip,
            )
            for arm in ARMS
        }
    )


# ---------------------------------------------------------------------------
# VirtualRobot
# ---------------------------------------------------------------------------


async def test_virtual_robot_echoes_commands(robot):
    left = np.arange(8, dtype=np.float32)
    right = np.arange(8, dtype=np.float32) * 2
    await robot.motion_control(left=left, right=right)
    got_left, got_right = await robot.get_positions()
    assert np.allclose(got_left, left)
    assert np.allclose(got_right, right)


async def test_virtual_robot_starts_at_rest_not_zero(kin, robot):
    """Starting at zeros would seat the session in a singularity."""
    left, right = await robot.get_positions()
    assert np.max(np.abs(left[:7])) > 0.0 or np.max(np.abs(right[:7])) > 0.0


async def test_virtual_robot_none_skips_an_arm(robot):
    before_left, _ = await robot.get_positions()
    await robot.motion_control(right=np.ones(8, dtype=np.float32))
    after_left, after_right = await robot.get_positions()
    assert np.allclose(after_left, before_left)
    assert np.allclose(after_right, np.ones(8))


async def test_virtual_robot_copies_inputs(robot):
    """A caller mutating its buffer must not retroactively change robot state."""
    command = np.zeros(8, dtype=np.float32)
    await robot.motion_control(left=command)
    command[0] = 42.0
    left, _ = await robot.get_positions()
    assert left[0] == 0.0


async def test_context_manager_enables_and_disables(robot):
    async with robot:
        assert robot.is_enabled
    assert not robot.is_enabled


# ---------------------------------------------------------------------------
# State plumbing
# ---------------------------------------------------------------------------


async def test_current_state_round_trips_through_the_robot(commander, kin, q_rest):
    """A fresh virtual robot reports the park pose, and the commander reads it back."""
    q, grips = await commander.current_state()
    assert q.shape == (kin.num_joints,)
    assert grips == (1.0, 1.0)
    assert np.allclose(q, q_rest, atol=1e-6)


def test_arm_command_splits_and_carries_grips(commander, kin, q_ready):
    left, right = commander.arm_command(q_ready, (0.25, 0.75))
    assert left.shape == right.shape == (8,)
    assert np.allclose(left[:7], kin.arm_q(q_ready, Arm.LEFT))
    assert np.allclose(right[:7], kin.arm_q(q_ready, Arm.RIGHT))
    assert left[7] == pytest.approx(0.25)
    assert right[7] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def test_plan_joint_reaches_the_target_configuration(commander, kin, q_ready):
    q_to = q_ready.copy()
    q_to[kin.indices(Arm.LEFT)[0]] += 0.3
    trajectory = commander.plan_joint(q_ready, q_to, speed=0.6)
    assert len(trajectory) >= 2
    assert np.max(np.abs(trajectory[-1] - q_to)) < 0.05


def test_plan_linear_holds_a_straight_line(commander, kin, q_ready):
    """The defining property of MoveL: the tool stays on the chord."""
    target = kin.fk(q_ready)[Arm.LEFT].translated((0.0, -0.12, 0.0))
    solution = kin.solve({Arm.LEFT: target}, seed=q_ready)
    assert solution.reached
    trajectory = commander.plan_linear(q_ready, solution.q, speed=0.05, ang_speed=0.5)

    start = kin.fk(trajectory[0])[Arm.LEFT].position
    end = kin.fk(trajectory[-1])[Arm.LEFT].position
    direction = end - start
    length = np.linalg.norm(direction)
    direction = direction / length
    worst = 0.0
    for q in trajectory:
        point = kin.fk(q)[Arm.LEFT].position - start
        worst = max(
            worst, float(np.linalg.norm(point - np.dot(point, direction) * direction))
        )
    assert worst < 3e-3, f"tool bowed {worst * 1e3:.1f} mm off the straight line"


def test_plan_to_poses_produces_matching_lengths(commander, kin, q_ready):
    target = kin.fk(q_ready)[Arm.LEFT].translated((0.0, -0.06, 0.03))
    leg = commander.plan_to_poses({Arm.LEFT: target}, q_ready, move=MoveType.JOINT)
    assert len(leg.q) == len(leg.grips) > 0


def test_planned_leg_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="gripper samples"):
        PlannedLeg(label="x", move=MoveType.JOINT, q=[np.zeros(14)], grips=[])


def test_plan_program_is_planned_before_anything_moves(commander, kin, q_ready):
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.02)))
    program.append(waypoint_at(kin, q_ready, (0.0, -0.10, 0.05), grip=0.0))
    plan = commander.plan_program(program, q_ready)
    assert len(plan.legs) == 2
    assert plan.ticks > 0
    assert plan.duration == pytest.approx(plan.ticks / TEST_RATE)


def test_first_leg_is_always_a_joint_move(commander, kin, q_ready):
    """The approach starts from an arbitrary pose, so it must not be linear."""
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.02)))
    program[0].move = MoveType.LINEAR
    plan = commander.plan_program(program, q_ready)
    assert plan.legs[0].move is MoveType.JOINT


def test_empty_program_is_rejected(commander, q_ready):
    with pytest.raises(ValueError, match="no waypoints"):
        commander.plan_program(Program(), q_ready)


def test_loops_repeat_the_body_but_not_the_approach(commander, kin, q_ready):
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0)))
    program.append(waypoint_at(kin, q_ready, (0.0, -0.09, 0.03)))
    once = commander.plan_program(program, q_ready, loops=1)
    twice = commander.plan_program(program, q_ready, loops=2)
    assert len(twice.legs) > len(once.legs)
    # The approach is flown once, at the start.
    assert sum(1 for leg in twice.legs if leg.label.startswith("approach")) == 1


def seams(plan):
    """Joint-space discontinuity at each leg boundary, in radians."""
    return [
        float(np.max(np.abs(plan.legs[i + 1].q[0] - plan.legs[i].q[-1])))
        for i in range(len(plan.legs) - 1)
    ]


@pytest.mark.parametrize("loops", [2, 3, 0])
def test_repeats_close_the_cycle_instead_of_jumping(commander, kin, q_ready, loops):
    """Every leg boundary must be continuous, including the pass seam.

    Without a closing leg, pass two starts from waypoint 1's configuration
    while the arm is at the last waypoint's. ``execute`` then absorbs the whole
    difference into its 0.4 s ease-in — measured at 0.217 rad against a
    within-leg peak of 0.002, a several-fold rate spike at the seam.
    """
    program = Program()
    for offset in ((0.0, -0.05, 0.02), (0.0, -0.09, 0.05), (0.0, -0.03, 0.08)):
        program.append(waypoint_at(kin, q_ready, offset))
    plan = commander.plan_program(program, q_ready, loops=loops)
    assert max(seams(plan)) < 0.05, seams(plan)


def test_repeated_legs_are_not_shared_objects(commander, kin, q_ready):
    """Editing one repeat must not edit every other copy of it."""
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0)))
    program.append(waypoint_at(kin, q_ready, (0.0, -0.09, 0.03)))
    plan = commander.plan_program(program, q_ready, loops=3)
    assert len({id(leg) for leg in plan.legs}) == len(plan.legs)


def test_negative_loops_is_rejected(commander, kin, q_ready):
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0)))
    with pytest.raises(ValueError, match="loops"):
        commander.plan_program(program, q_ready, loops=-1)


async def test_stop_reports_the_ticks_actually_sent(commander, kin, q_ready):
    """A stop mid-leg must not report zero ticks for motion that happened."""
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.10, 0.05)))
    plan = commander.plan_program(program, q_ready)

    stop = threading.Event()
    sent = 0
    original = commander.robot.motion_control

    async def counting(left=None, right=None):
        nonlocal sent
        sent += 1
        if sent == 12:
            stop.set()
        await original(left=left, right=right)

    commander.robot.motion_control = counting
    report = await commander.execute(plan, stop_event=stop)
    assert report.stopped
    assert report.ticks == sent, f"reported {report.ticks}, actually sent {sent}"


def test_gripper_change_adds_ticks_at_the_waypoint(commander, kin, q_ready):
    """A grasp must happen with the arm stationary, not during the leg."""
    same = Program()
    same.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0), grip=1.0))
    changed = Program()
    changed.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0), grip=0.0))
    plan_same = commander.plan_program(same, q_ready)
    plan_changed = commander.plan_program(changed, q_ready)
    assert plan_changed.ticks > plan_same.ticks
    # The extra ticks hold one configuration while the grip sweeps.
    tail = plan_changed.legs[0]
    assert np.allclose(tail.q[-1], tail.q[-2])
    assert tail.grips[-1][0] == pytest.approx(0.0)


def test_unreachable_waypoint_is_reported_at_plan_time(commander, kin, q_ready):
    from almond_axol.kinematics.path import PathPlanningError
    from almond_axol.motion.frames import Pose

    program = Program()
    program.append(
        Waypoint(
            targets={
                arm: ArmTarget(
                    pose=Pose(
                        np.array([0.0, -2.0, 1.0], dtype=np.float32),
                        np.eye(3, dtype=np.float32),
                    )
                )
                for arm in ARMS
            }
        )
    )
    with pytest.raises(PathPlanningError, match="[Oo]ut of reach|cannot fly"):
        commander.plan_program(program, q_ready)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def test_execute_moves_the_robot_to_the_end_of_the_plan(
    commander, kin, q_ready, robot
):
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.08, 0.04)))
    plan = commander.plan_program(program, q_ready)
    report = await commander.execute(plan)
    assert not report.stopped
    assert report.legs_completed == len(plan.legs)

    q_final, _ = await commander.current_state()
    assert np.allclose(q_final, plan.legs[-1].q[-1], atol=1e-5)
    reached = kin.fk(q_final)[Arm.LEFT]
    assert reached.distance_to(program[0].pose(Arm.LEFT)) < 5e-3


async def test_execute_applies_the_commanded_gripper(commander, kin, q_ready, robot):
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.06, 0.0), grip=0.0))
    await commander.execute(commander.plan_program(program, q_ready))
    left, right = await robot.get_positions()
    assert left[7] == pytest.approx(0.0, abs=1e-3)
    assert right[7] == pytest.approx(0.0, abs=1e-3)


async def test_stop_event_halts_execution(commander, kin, q_ready):
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.10, 0.05)))
    program.append(waypoint_at(kin, q_ready, (0.0, -0.02, 0.0)))
    plan = commander.plan_program(program, q_ready)

    stop = threading.Event()
    stop.set()
    report = await commander.execute(plan, stop_event=stop)
    assert report.stopped
    assert report.legs_completed == 0
    assert "Stopped" in report.message


async def test_progress_callback_reports_every_leg(commander, kin, q_ready):
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0)))
    program.append(waypoint_at(kin, q_ready, (0.0, -0.09, 0.02)))
    plan = commander.plan_program(program, q_ready)

    seen: list[int] = []
    await commander.execute(plan, on_progress=lambda i, leg, frac: seen.append(i))
    assert set(seen) == set(range(len(plan.legs)))


async def test_move_to_poses_is_a_one_call_move(commander, kin, q_ready):
    target = kin.fk(q_ready)[Arm.LEFT].translated((0.0, -0.07, 0.03))
    report = await commander.move_to_poses({Arm.LEFT: target})
    assert report.legs_completed == 1
    q_final, _ = await commander.current_state()
    assert kin.fk(q_final)[Arm.LEFT].distance_to(target) < 5e-3


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------


def test_ease_in_offsets_start_at_the_gap_and_end_at_zero():
    q_now = np.array([0.5, 0.0], dtype=np.float32)
    q_first = np.array([0.0, 0.0], dtype=np.float32)
    offsets = ease_in_offsets(q_now, q_first, 0.4, 50.0)
    assert len(offsets) == 20
    assert offsets[0][0] == pytest.approx(0.5, abs=1e-6)
    assert abs(offsets[-1][0]) < 0.05
    # Monotonically decaying, so the arm never doubles back.
    assert np.all(np.diff(offsets[:, 0]) <= 1e-6)


def test_ease_in_offsets_empty_when_already_aligned():
    q = np.zeros(3, dtype=np.float32)
    assert ease_in_offsets(q, q, 0.4, 50.0).size == 0
    assert ease_in_offsets(None, q, 0.4, 50.0).size == 0
