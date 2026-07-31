"""Absolute-pose IK: convergence, conditioning, and the failure modes it fixes.

Several of these tests pin numbers that were *measured* while building this
module — the steady-state offset of the untuned solver, the singularity margin
at the zero pose, the whole-solve freeze a rank-deficient held arm causes. They
are regression guards on behaviour that was genuinely surprising, and each one
would otherwise be rediscovered the expensive way.
"""

from __future__ import annotations

import numpy as np
import pytest

from almond_axol.motion.frames import ARMS, Arm, Pose
from almond_axol.motion.kinematics import SINGULARITY_THRESHOLD, UnreachableError


@pytest.fixture(scope="module")
def q_zero(kin):
    return np.zeros(kin.num_joints, dtype=np.float32)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_arm_indices_are_disjoint_and_complete(kin):
    left, right = kin.indices(Arm.LEFT), kin.indices(Arm.RIGHT)
    assert len(left) == len(right) == 7
    assert set(left).isdisjoint(right)
    assert sorted(left + right) == list(range(kin.num_joints))


def test_rest_pose_is_not_the_zero_pose(kin, q_rest, q_zero):
    """The default seed must not be the singular configuration."""
    assert np.max(np.abs(q_rest - q_zero)) > 0.1


def test_fk_round_trips_through_ik(kin, q_ready):
    """FK of one configuration, solved from a *different* seed, returns to it.

    Seeding with the answer makes this pass at iteration zero — it would still
    pass with the solver removed entirely — so the seed has to be somewhere
    else.
    """
    q_target = q_ready.copy()
    q_target[kin.indices(Arm.LEFT)[3]] -= 0.25
    q_target[kin.indices(Arm.RIGHT)[3]] += 0.25
    poses = kin.fk(q_target)

    solution = kin.solve(poses, seed=q_ready)
    assert solution.reached, solution.message
    assert not np.allclose(solution.q, q_ready, atol=1e-3), "the solver did not move"
    for arm in ARMS:
        assert kin.fk(solution.q)[arm].is_close(poses[arm], pos_tol=1e-3, ori_tol=1e-2)


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def test_zero_pose_is_singular(kin, q_zero):
    """Straight down, shoulder_3 and wrist_1 are roll joints along the link axis.

    They produce no tool motion at all, so the Jacobian loses rank. This is why
    a solve seeded at zeros used to return its seed unchanged forever.
    """
    for arm in ARMS:
        assert kin.singularity_margin(q_zero, arm) < 1e-3
        assert kin.is_singular(q_zero, arm)


def test_ready_pose_is_far_better_conditioned_than_rest(kin, q_rest, q_ready):
    """The reason :meth:`ready_q` exists.

    Rest is 88% extended; Cartesian work should not start there. This guards
    the gap rather than the absolute numbers, so retuning the rest pose stays
    possible without silently losing the property.
    """
    for arm in ARMS:
        assert kin.singularity_margin(q_ready, arm) > 3.0 * kin.singularity_margin(
            q_rest, arm
        )


def test_ready_pose_clears_the_torso(kin, q_ready, q_rest):
    """The reason ``shoulder_2`` is in the ready pose.

    An earlier version left it at zero and cleared the centre column by 9.7 mm
    — close enough that rotating the tool in place drove the gripper into it.
    A starting pose handed to an operator has to have room around it.
    """
    assert kin.self_collision_clearance(q_ready) > 0.10
    assert kin.self_collision_clearance(q_ready) > 10 * kin.self_collision_clearance(
        q_rest
    )


def test_rotating_in_place_at_the_ready_pose_stays_clear(kin, q_ready):
    """Turning the tool where it stands must not push it into the torso."""
    from almond_axol.motion.frames import Arm as _Arm

    target = kin.fk(q_ready)[_Arm.LEFT].rotated_local(0.4, -0.3, 0.2)
    solution = kin.solve({_Arm.LEFT: target}, seed=q_ready)
    assert solution.usable, solution.message


def test_ready_pose_leaves_room_to_jog(kin, q_ready):
    """Every axis must have travel left before the reach clamp bites."""
    for arm in ARMS:
        pose = kin.fk(q_ready)[arm]
        for delta in (
            (0.08, 0, 0),
            (-0.08, 0, 0),
            (0, -0.1, 0),
            (0, 0.1, 0),
            (0, 0, 0.1),
        ):
            assert kin.reach_excess(pose.translated(delta), arm) < 0.0


def test_ready_pose_reaches_forward_and_up(kin, q_ready):
    """Both tools point into the workspace, mirrored — not one of them backwards."""
    poses = kin.fk(q_ready)
    assert poses[Arm.LEFT].position[1] < 0.0
    assert poses[Arm.RIGHT].position[1] < 0.0
    assert poses[Arm.LEFT].position[1] == pytest.approx(
        poses[Arm.RIGHT].position[1], abs=1e-4
    )


def test_rest_pose_is_well_conditioned(kin, q_rest):
    for arm in ARMS:
        assert kin.singularity_margin(q_rest, arm) > SINGULARITY_THRESHOLD
        assert not kin.is_singular(q_rest, arm)


def test_singularity_threshold_separates_the_two(kin, q_rest, q_zero):
    """The threshold has to sit strictly between the known-bad and known-good poses."""
    worst_rest = min(kin.singularity_margin(q_rest, a) for a in ARMS)
    worst_zero = max(kin.singularity_margin(q_zero, a) for a in ARMS)
    assert worst_zero < SINGULARITY_THRESHOLD < worst_rest


def test_jacobian_shape_and_rank(kin, q_rest, q_zero):
    jac = kin.jacobian(q_rest, Arm.LEFT)
    assert jac.shape == (6, 7)
    assert np.linalg.matrix_rank(jac, tol=1e-3) == 6
    # Two roll joints contribute no tool motion at the zero pose.
    assert np.linalg.matrix_rank(kin.jacobian(q_zero, Arm.LEFT), tol=1e-3) < 6


def test_desingularize_clears_a_zero_seed(kin, q_zero):
    fixed = kin.desingularize(q_zero)
    for arm in ARMS:
        assert kin.singularity_margin(fixed, arm) >= 2.0 * SINGULARITY_THRESHOLD


def test_desingularize_leaves_a_good_seed_alone(kin, q_rest):
    assert np.allclose(kin.desingularize(q_rest), q_rest)


# ---------------------------------------------------------------------------
# Convergence — the headline fix
# ---------------------------------------------------------------------------


def test_absolute_target_converges_to_sub_millimetre(kin, q_ready):
    """The behaviour this module exists to provide."""
    poses = kin.fk(q_ready)
    targets = {
        Arm.LEFT: poses[Arm.LEFT].translated((0.0, -0.12, 0.08)),
        Arm.RIGHT: poses[Arm.RIGHT].translated((0.0, -0.12, 0.08)),
    }
    solution = kin.solve(targets, seed=q_ready)
    assert solution.reached
    assert solution.worst_position_error < 1e-3
    assert solution.worst_orientation_error < 1e-2


def test_untuned_weights_do_not_converge(kin, q_ready):
    """Guards the reason the weight scaling exists.

    With the teleop-tuned weights the rest and posture costs balance the pose
    cost and the solve plateaus centimetres away. If this ever starts passing,
    the upstream defaults changed and :data:`DEFAULT_WEIGHT_SCALE` should be
    revisited rather than silently carried.
    """
    poses = kin.fk(q_ready)
    targets = {Arm.LEFT: poses[Arm.LEFT].translated((0.0, -0.12, 0.08))}
    naive = kin.solve(targets, seed=q_ready, weight_scale=1.0, max_iterations=200)
    tuned = kin.solve(targets, seed=q_ready)
    assert not naive.reached
    assert naive.worst_position_error > 10.0 * tuned.worst_position_error


@pytest.mark.parametrize(
    "delta", [(0.05, 0.0, 0.0), (0.0, -0.15, 0.0), (0.0, 0.0, 0.12)]
)
def test_converges_along_each_axis(kin, q_ready, delta):
    target = kin.fk(q_ready)[Arm.LEFT].translated(delta)
    solution = kin.solve({Arm.LEFT: target}, seed=q_ready)
    assert solution.reached, solution.message


def test_converges_with_rotation(kin, q_ready):
    target = kin.fk(q_ready)[Arm.LEFT].rotated_local(0.4, -0.3, 0.2)
    solution = kin.solve({Arm.LEFT: target}, seed=q_ready)
    assert solution.reached, solution.message
    assert kin.fk(solution.q)[Arm.LEFT].angle_to(target) < 1e-2


def test_untargeted_arm_holds_still(kin, q_ready):
    """An arm with no target must not drift — and the targeted one must move.

    Without the second assertion, "nothing moved at all" satisfies this.
    """
    target = kin.fk(q_ready)[Arm.LEFT].translated((0.0, -0.10, 0.05))
    solution = kin.solve({Arm.LEFT: target}, seed=q_ready)
    assert solution.reached, solution.message
    assert np.allclose(
        kin.arm_q(solution.q, Arm.RIGHT), kin.arm_q(q_ready, Arm.RIGHT), atol=1e-6
    )
    assert kin.fk(solution.q)[Arm.LEFT].distance_to(target) < 1e-3


def test_a_well_conditioned_untargeted_arm_is_never_nudged(kin, q_ready):
    """``desingularize`` must leave a non-singular arm exactly where it is.

    It used to clear every arm to twice the singularity threshold, so an arm
    that was merely mediocre — but perfectly usable, and given no target — got
    dragged toward the rest pose and that displacement was returned as the
    answer.
    """
    for scale in np.linspace(0.0, 1.0, 25):
        q = kin.rest_q() * (1 - scale) + kin.ready_q() * scale
        margin = kin.singularity_margin(q, Arm.RIGHT)
        if margin < SINGULARITY_THRESHOLD:
            continue
        nudged = kin.desingularize(q, ARMS)
        assert np.allclose(
            kin.arm_q(nudged, Arm.RIGHT), kin.arm_q(q, Arm.RIGHT), atol=1e-9
        ), f"a non-singular arm (margin {margin:.4f}) was moved"


def test_singular_held_arm_does_not_freeze_the_solve(kin, q_zero, q_ready):
    """A rank-deficient *held* arm used to stop every joint from moving.

    The solver poses one coupled least-squares problem over all 14 joints, so a
    degenerate block anywhere makes the trust-region step get rejected. Measured
    before the fix: zero joint motion across 25 consecutive calls.
    """
    target = kin.fk(q_ready)[Arm.LEFT].translated((0.0, -0.10, 0.05))
    solution = kin.solve({Arm.LEFT: target}, seed=q_zero)
    assert solution.reached, solution.message
    assert np.max(np.abs(solution.q - q_zero)) > 0.05


def test_solution_is_the_closest_approach_not_the_last(kin, q_rest):
    """A failed solve still reports a usable configuration and a real error."""
    poses = kin.fk(q_rest)
    far = poses[Arm.LEFT].translated((0.0, -1.5, 0.0))
    solution = kin.solve({Arm.LEFT: far}, seed=q_rest)
    assert not solution.reached
    assert solution.q.shape == (kin.num_joints,)
    assert np.all(np.isfinite(solution.q))
    measured = kin.fk(solution.q)[Arm.LEFT].distance_to(far)
    assert solution.position_error[Arm.LEFT] == pytest.approx(measured, abs=1e-6)


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------


def test_out_of_reach_is_named_as_such(kin, q_rest):
    far = Pose(
        np.array([0.0, -2.0, 1.0], dtype=np.float32), np.eye(3, dtype=np.float32)
    )
    solution = kin.solve({Arm.LEFT: far}, seed=q_rest)
    assert not solution.reached
    assert "reach" in solution.message.lower()


def test_reach_excess_sign(kin, q_rest):
    here = kin.fk(q_rest)[Arm.LEFT]
    assert kin.reach_excess(here, Arm.LEFT) < 0.0
    far = Pose(
        np.array([0.0, -2.0, 1.0], dtype=np.float32), np.eye(3, dtype=np.float32)
    )
    assert kin.reach_excess(far, Arm.LEFT) > 0.0


def test_raise_on_failure(kin, q_rest):
    far = Pose(
        np.array([0.0, -2.0, 1.0], dtype=np.float32), np.eye(3, dtype=np.float32)
    )
    with pytest.raises(UnreachableError) as info:
        kin.solve({Arm.LEFT: far}, seed=q_rest, raise_on_failure=True)
    assert info.value.solution.q.shape == (kin.num_joints,)


def test_empty_targets_rejected(kin):
    with pytest.raises(ValueError):
        kin.solve({})


def test_solver_config_is_restored(kin, q_rest):
    """Weights are shared mutable state; a solve must leave them untouched."""
    before = (
        kin.solver.config.pos_weight,
        kin.solver.config.ori_weight,
        kin.solver.config.max_joint_delta,
    )
    kin.solve(
        {Arm.LEFT: kin.fk(q_rest)[Arm.LEFT].translated((0.0, -0.05, 0.0))}, seed=q_rest
    )
    assert (
        kin.solver.config.pos_weight,
        kin.solver.config.ori_weight,
        kin.solver.config.max_joint_delta,
    ) == before


def test_config_restored_even_when_the_solve_raises(kin, q_rest):
    far = Pose(
        np.array([0.0, -2.0, 1.0], dtype=np.float32), np.eye(3, dtype=np.float32)
    )
    before = kin.solver.config.pos_weight
    with pytest.raises(UnreachableError):
        kin.solve({Arm.LEFT: far}, seed=q_rest, raise_on_failure=True)
    assert kin.solver.config.pos_weight == before


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_tool_offset_changes_where_the_arm_goes(kin, q_ready):
    """Solving for the tip and for the flange must not produce the same joints."""
    from almond_axol.motion.frames import FLANGE, GRIPPER_TIP

    target = kin.fk(q_ready, GRIPPER_TIP)[Arm.LEFT].translated((0.0, -0.10, 0.05))
    tip = kin.solve({Arm.LEFT: target}, seed=q_ready, tool=GRIPPER_TIP)
    flange = kin.solve({Arm.LEFT: target}, seed=q_ready, tool=FLANGE)
    assert tip.reached and flange.reached
    assert np.max(np.abs(tip.q - flange.q)) > 1e-3
    # And each lands its own frame on the target.
    assert kin.fk(tip.q, GRIPPER_TIP)[Arm.LEFT].distance_to(target) < 1e-3
    assert kin.fk(flange.q, FLANGE)[Arm.LEFT].distance_to(target) < 1e-3


def test_concurrent_solves_do_not_corrupt_the_shared_config(kin, q_ready):
    """``solver.config`` is process-global and solve() scales it in place.

    Without a lock, two overlapping solves each capture the other's *scaled*
    values as their baseline and restore those — permanently, for the life of
    the process. ``max_joint_delta`` is one of them, and live VR teleop reads
    it from the same solver.
    """
    import threading

    from almond_axol.motion.frames import Arm as _Arm

    before = (
        kin.solver.config.pos_weight,
        kin.solver.config.ori_weight,
        kin.solver.config.max_joint_delta,
        kin.solver.config.limit_weight,
        kin.solver.config.self_collision_weight,
        kin.solver.config.self_collision_margin,
    )
    target = kin.fk(q_ready)[_Arm.LEFT].translated((0.0, -0.04, 0.02))

    def work():
        for _ in range(4):
            kin.solve({_Arm.LEFT: target}, seed=q_ready)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    after = (
        kin.solver.config.pos_weight,
        kin.solver.config.ori_weight,
        kin.solver.config.max_joint_delta,
        kin.solver.config.limit_weight,
        kin.solver.config.self_collision_weight,
        kin.solver.config.self_collision_margin,
    )
    assert after == before, f"shared config corrupted: {before} -> {after}"
