"""The hard gate: soft costs steer the solve, a check refuses what they missed.

Self-collision is an L2 cost inside the solver, so it can always be outvoted
and is never a guarantee. These cover the layer that *is* a guarantee — the
explicit check that runs before anything is commanded.
"""

from __future__ import annotations

import numpy as np
import pytest

from almond_axol.kinematics.path import PathPlanningError
from almond_axol.motion import ARMS, Arm, MotionCommander, Program, VirtualRobot
from almond_axol.motion.program import ArmTarget, Waypoint

TEST_RATE = 50.0


@pytest.fixture
def commander(kin):
    q = kin.ready_q()
    robot = VirtualRobot(
        np.append(kin.arm_q(q, Arm.LEFT), 1.0), np.append(kin.arm_q(q, Arm.RIGHT), 1.0)
    )
    return MotionCommander(kin, robot, rate_hz=TEST_RATE)


def waypoint_at(kin, q, offset=(0.0, 0.0, 0.0)):
    poses = kin.fk(q)
    return Waypoint(
        targets={arm: ArmTarget(pose=poses[arm].translated(offset)) for arm in ARMS}
    )


# ---------------------------------------------------------------------------
# Clearance measurement
# ---------------------------------------------------------------------------


def test_clearance_along_finds_the_worst_tick(kin, q_ready):
    """A trajectory is only as clear as its worst sample."""
    good = kin.ready_q()
    bad = kin.rest_q()  # folded against the torso, ~10 mm clearance
    clear_good, _ = kin.clearance_along([good, good, good])
    clear_mixed, index = kin.clearance_along([good, good, bad, good])
    assert clear_good > clear_mixed
    assert index == 2
    assert clear_mixed == pytest.approx(kin.self_collision_clearance(bad), abs=1e-6)


def test_clearance_along_matches_the_single_pose_check(kin, q_ready):
    single = kin.self_collision_clearance(q_ready)
    batched, _ = kin.clearance_along([q_ready])
    assert batched == pytest.approx(single, abs=1e-6)


def test_clearance_along_handles_an_empty_trajectory(kin):
    clearance, index = kin.clearance_along([])
    assert clearance == float("inf")
    assert index == -1


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_a_clear_program_plans(commander, kin, q_ready):
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.06, 0.03)))
    program.append(waypoint_at(kin, q_ready, (0.0, -0.10, 0.06)))
    plan = commander.plan_program(program, q_ready)
    assert len(plan.legs) == 2
    for leg in plan.legs:
        assert kin.clearance_along(leg.q)[0] >= 0.0


def test_the_gate_rejects_a_trajectory_that_passes_through_the_torso(
    commander, kin, q_ready, q_penetrating, monkeypatch
):
    """The module's stated guarantee, exercised directly.

    Both waypoints here are clear and individually solvable; the *trajectory*
    between them is what dips into the column. The planner is stubbed to return
    exactly such a path, because the guarantee under test is that
    ``plan_program`` inspects what it is handed — not that the interpolator
    happens to produce something bad.
    """
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0)))
    program.append(waypoint_at(kin, q_ready, (0.0, -0.09, 0.03)))

    def through_the_torso(q_from, q_to, **kwargs):
        return [np.asarray(q_from), q_penetrating, np.asarray(q_to)]

    monkeypatch.setattr(commander, "plan_joint", through_the_torso)
    monkeypatch.setattr(commander, "plan_linear", through_the_torso)

    with pytest.raises(PathPlanningError, match="through the torso"):
        commander.plan_program(program, q_ready)


def test_the_gate_reports_which_leg_and_how_deep(
    commander, kin, q_ready, q_penetrating, monkeypatch
):
    """An operator needs the leg and the depth, not just a refusal."""
    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0)))
    program.append(waypoint_at(kin, q_ready, (0.0, -0.09, 0.03)))
    program[1].label = "into the column"

    def through_the_torso(q_from, q_to, **kwargs):
        return [np.asarray(q_from), q_penetrating, np.asarray(q_to)]

    monkeypatch.setattr(commander, "plan_joint", through_the_torso)
    monkeypatch.setattr(commander, "plan_linear", through_the_torso)

    with pytest.raises(PathPlanningError) as info:
        commander.plan_program(program, q_ready)
    message = str(info.value)
    assert "mm" in message
    depth = kin.self_collision_clearance(q_penetrating)
    assert f"{abs(depth) * 1e3:.0f} mm" in message


def test_a_clear_program_is_not_rejected_by_the_gate(commander, kin, q_ready):
    """The gate must not be so eager that ordinary programs stop planning."""
    program = Program()
    for offset in ((0.0, -0.04, 0.02), (0.0, -0.08, 0.05), (0.05, -0.06, 0.03)):
        program.append(waypoint_at(kin, q_ready, offset))
    plan = commander.plan_program(program, q_ready, loops=2)
    assert len(plan.legs) > 3


def test_resolve_refuses_a_waypoint_that_only_reaches_through_the_torso(
    kin, q_ready, monkeypatch
):
    """The ``usable`` vs ``reached`` distinction, which nothing else pins.

    A solution can hit its target exactly and still put the arm inside the
    column. Recording or replaying that is the failure this guards.
    """
    from almond_axol.motion.kinematics import IKSolution

    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0)))

    def reached_but_colliding(targets, seed=None, **kwargs):
        return IKSolution(
            q=np.asarray(q_ready, dtype=np.float32),
            reached=True,
            position_error=dict.fromkeys(targets, 0.0),
            orientation_error=dict.fromkeys(targets, 0.0),
            clearance=-0.05,
        )

    monkeypatch.setattr(kin, "solve", reached_but_colliding)
    assert program.resolve(kin, seed=q_ready) == [0], (
        "a reached-but-colliding solution must not be accepted"
    )


def test_resolve_accepts_a_solution_that_is_both_reached_and_clear(
    kin, q_ready, monkeypatch
):
    """The mirror of the above, so the refusal is not simply always-on."""
    from almond_axol.motion.kinematics import IKSolution

    program = Program()
    program.append(waypoint_at(kin, q_ready, (0.0, -0.05, 0.0)))

    def reached_and_clear(targets, seed=None, **kwargs):
        return IKSolution(
            q=np.asarray(q_ready, dtype=np.float32),
            reached=True,
            position_error=dict.fromkeys(targets, 0.0),
            orientation_error=dict.fromkeys(targets, 0.0),
            clearance=+0.20,
        )

    monkeypatch.setattr(kin, "solve", reached_and_clear)
    assert program.resolve(kin, seed=q_ready) == []


def test_ik_solution_usable_requires_both_properties():
    from almond_axol.motion.kinematics import IKSolution

    q = np.zeros(14, dtype=np.float32)
    assert IKSolution(q=q, reached=True, clearance=0.2).usable
    assert not IKSolution(q=q, reached=True, clearance=-0.01).usable
    assert not IKSolution(q=q, reached=False, clearance=0.2).usable
    assert IKSolution(q=q, reached=True, clearance=-0.01).collides
