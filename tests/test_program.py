"""The editable motion program: edit operations, serialization, resolution."""

from __future__ import annotations

import json

import numpy as np
import pytest

from almond_axol.motion.frames import ARMS, GRIPPER_TIP, Arm, Frame, Pose, Tool
from almond_axol.motion.program import (
    FORMAT_VERSION,
    ArmTarget,
    MoveType,
    Program,
    ProgramError,
    Waypoint,
)

RNG = np.random.default_rng(7)


def make_waypoint(
    x: float = 0.0, label: str = "", move: MoveType = MoveType.LINEAR
) -> Waypoint:
    """A waypoint with distinguishable poses and a known joint vector."""
    return Waypoint(
        targets={
            arm: ArmTarget(
                pose=Pose(
                    np.array(
                        [x + (0.2 if arm is Arm.LEFT else -0.2), 0.0, 0.3], np.float32
                    ),
                    np.eye(3, dtype=np.float32),
                ),
                joints=np.full(7, x, dtype=np.float32),
                grip=1.0,
            )
            for arm in ARMS
        },
        move=move,
        label=label,
    )


def make_program(n: int = 4) -> Program:
    program = Program()
    for i in range(n):
        program.append(make_waypoint(x=float(i), label=f"wp{i}"))
    return program


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_waypoint_requires_both_arms():
    with pytest.raises(ValueError, match="missing targets"):
        Waypoint(targets={Arm.LEFT: ArmTarget(pose=Pose.identity())})


def test_arm_target_validates_joint_shape():
    with pytest.raises(ValueError, match=r"\(7,\)"):
        ArmTarget(pose=Pose.identity(), joints=np.zeros(8))


def test_arm_target_clamps_grip():
    assert ArmTarget(pose=Pose.identity(), grip=5.0).grip == 1.0
    assert ArmTarget(pose=Pose.identity(), grip=-2.0).grip == 0.0


def test_program_always_has_a_world_frame():
    assert "world" in Program().frames


# ---------------------------------------------------------------------------
# Edit operations
# ---------------------------------------------------------------------------


def test_append_and_index():
    program = make_program(3)
    assert len(program) == 3
    assert program[0].label == "wp0"
    assert program[-1].label == "wp2"


def test_insert_clamps_out_of_range():
    program = make_program(2)
    assert program.insert(99, make_waypoint(label="end")) == 2
    assert program.insert(-99, make_waypoint(label="start")) == 0
    assert [w.label for w in program] == ["start", "wp0", "wp1", "end"]


def test_remove_returns_the_waypoint():
    program = make_program(3)
    removed = program.remove(1)
    assert removed.label == "wp1"
    assert [w.label for w in program] == ["wp0", "wp2"]


def test_move_up_and_down():
    program = make_program(4)
    assert program.move_down(0) == 1
    assert [w.label for w in program] == ["wp1", "wp0", "wp2", "wp3"]
    assert program.move_up(1) == 0
    assert [w.label for w in program] == ["wp0", "wp1", "wp2", "wp3"]


def test_move_up_at_top_is_a_no_op_not_a_wrap():
    """Holding "up" on the first waypoint must not teleport it to the end."""
    program = make_program(3)
    assert program.move_up(0) == 0
    assert [w.label for w in program] == ["wp0", "wp1", "wp2"]


def test_move_down_at_bottom_is_a_no_op():
    program = make_program(3)
    assert program.move_down(2) == 2
    assert [w.label for w in program] == ["wp0", "wp1", "wp2"]


def test_move_to_arbitrary_position():
    program = make_program(5)
    assert program.move(0, 3) == 3
    assert [w.label for w in program] == ["wp1", "wp2", "wp3", "wp0", "wp4"]


def test_duplicate_inserts_after_and_is_a_deep_copy():
    program = make_program(2)
    index = program.duplicate(0)
    assert index == 1
    assert [w.label for w in program] == ["wp0", "wp0", "wp1"]
    program[1].targets[Arm.LEFT].joints[0] = 99.0
    assert program[0].targets[Arm.LEFT].joints[0] == 0.0


def test_edits_on_empty_program_raise():
    program = Program()
    for op in (program.remove, program.duplicate, program.move_up, program.move_down):
        with pytest.raises(ProgramError, match="no waypoints"):
            op(0)


def test_index_out_of_range_raises():
    program = make_program(2)
    with pytest.raises(ProgramError, match="out of range"):
        program.remove(5)


def test_negative_index_is_supported():
    program = make_program(3)
    assert program.remove(-1).label == "wp2"


def test_clear_keeps_settings_and_frames():
    program = make_program(3)
    program.add_frame(Frame("vise"))
    program.speed = 0.42
    program.clear()
    assert len(program) == 0
    assert program.speed == 0.42
    assert "vise" in program.frames


# ---------------------------------------------------------------------------
# Editing a waypoint
# ---------------------------------------------------------------------------


def test_with_pose_drops_stale_joints():
    """The stored joints described the old pose; keeping them would replay it."""
    waypoint = make_waypoint(x=1.0)
    assert waypoint.is_resolved
    moved = waypoint.with_pose(Arm.LEFT, Pose.identity())
    assert moved.joints(Arm.LEFT) is None
    assert not moved.is_resolved
    # The untouched arm keeps its solution.
    assert moved.joints(Arm.RIGHT) is not None


def test_with_pose_does_not_mutate_the_original():
    waypoint = make_waypoint(x=1.0)
    waypoint.with_pose(Arm.LEFT, Pose.identity())
    assert waypoint.joints(Arm.LEFT) is not None


def test_with_grip_preserves_joints():
    waypoint = make_waypoint(x=1.0).with_grip(Arm.LEFT, 0.0)
    assert waypoint.grip(Arm.LEFT) == 0.0
    assert waypoint.grip(Arm.RIGHT) == 1.0
    assert waypoint.is_resolved


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_json_round_trip(tmp_path):
    program = make_program(3)
    program.speed = 0.07
    program.dwell = 1.25
    program[1].move = MoveType.JOINT
    program[2].speed = 0.03
    path = tmp_path / "p.json"
    program.save(path)

    loaded = Program.load(path)
    assert len(loaded) == 3
    assert loaded.speed == pytest.approx(0.07)
    assert loaded.dwell == pytest.approx(1.25)
    assert loaded[1].move is MoveType.JOINT
    assert loaded[2].speed == pytest.approx(0.03)
    for i in range(3):
        for arm in ARMS:
            assert (
                loaded[i]
                .pose(arm)
                .is_close(program[i].pose(arm), pos_tol=1e-4, ori_tol=1e-3)
            )
            assert np.allclose(loaded[i].joints(arm), program[i].joints(arm), atol=1e-5)


def test_save_writes_version_and_is_valid_json(tmp_path):
    path = tmp_path / "p.json"
    make_program(2).save(path)
    data = json.loads(path.read_text())
    assert data["version"] == FORMAT_VERSION
    assert len(data["waypoints"]) == 2


def test_saving_does_not_invent_labels(tmp_path):
    """An unlabelled waypoint stays unlabelled on disk.

    Substituting its index makes save/load non-idempotent, and the name goes
    stale the moment the program is reordered — "waypoint 3" sitting second.
    The display fallback belongs in the UI.
    """
    program = Program()
    program.append(make_waypoint())
    path = tmp_path / "p.json"
    program.save(path)
    assert json.loads(path.read_text())["waypoints"][0]["label"] == ""


def test_save_load_is_idempotent(tmp_path):
    program = make_program(3)
    program[1].label = ""
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    program.save(first)
    Program.load(first).save(second)
    assert json.loads(first.read_text()) == json.loads(second.read_text())


def test_reordering_does_not_leave_stale_names(tmp_path):
    program = Program()
    for _ in range(3):
        program.append(make_waypoint())
    program.move_up(2)
    path = tmp_path / "p.json"
    program.save(path)
    labels = [w["label"] for w in json.loads(path.read_text())["waypoints"]]
    assert labels == ["", "", ""]


def test_load_missing_file_returns_empty(tmp_path):
    assert len(Program.load(tmp_path / "nope.json")) == 0


def test_load_rejects_bad_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ProgramError, match="not valid JSON"):
        Program.load(path)


def test_load_rejects_unknown_version(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"version": 99, "waypoints": []}))
    with pytest.raises(ProgramError, match="version 99"):
        Program.load(path)


def test_save_is_atomic_leaving_no_temp_file(tmp_path):
    path = tmp_path / "nested" / "p.json"
    make_program(1).save(path)
    assert path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_custom_tool_survives_a_round_trip(tmp_path):
    program = make_program(1)
    program.tool = Tool("probe", (0.0, 0.01, -0.2))
    path = tmp_path / "p.json"
    program.save(path)
    loaded = Program.load(path)
    assert loaded.tool.name == "probe"
    assert loaded.tool.offset == (0.0, 0.01, -0.2)


def test_frames_survive_a_round_trip(tmp_path):
    program = make_program(1)
    program.add_frame(
        Frame(
            "vise",
            Pose(
                np.array([0.1, -0.2, 0.3], dtype=np.float32),
                np.eye(3, dtype=np.float32),
            ),
        )
    )
    path = tmp_path / "p.json"
    program.save(path)
    loaded = Program.load(path)
    assert np.allclose(loaded.frame("vise").pose.position, [0.1, -0.2, 0.3], atol=1e-5)


def test_unknown_frame_reference_is_reported():
    program = make_program(1)
    program[0].frame = "missing"
    with pytest.raises(ProgramError, match="does not define"):
        program.world_pose(0, Arm.LEFT)


def test_world_pose_applies_the_frame():
    program = make_program(1)
    program.add_frame(
        Frame(
            "shifted",
            Pose(
                np.array([0.0, 0.0, 0.5], dtype=np.float32), np.eye(3, dtype=np.float32)
            ),
        )
    )
    program[0].frame = "shifted"
    local = program[0].pose(Arm.LEFT)
    assert program.world_pose(0, Arm.LEFT).position[2] == pytest.approx(
        local.position[2] + 0.5, abs=1e-5
    )


# ---------------------------------------------------------------------------
# Version 1 upgrade
# ---------------------------------------------------------------------------


def legacy_file(tmp_path, kin):
    q = kin.rest_q()
    entry = {
        "label": "taught",
        "left": [*kin.arm_q(q, Arm.LEFT).tolist(), 1.0],
        "right": [*kin.arm_q(q, Arm.RIGHT).tolist(), 0.0],
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"version": 1, "waypoints": [entry]}))
    return path


def test_legacy_file_needs_kinematics(tmp_path, kin):
    path = legacy_file(tmp_path, kin)
    with pytest.raises(ProgramError, match="version 1"):
        Program.load(path)


def test_legacy_file_upgrades_with_poses_from_fk(tmp_path, kin):
    """A hand-taught program must open here without being re-taught."""
    path = legacy_file(tmp_path, kin)
    program = Program.load(path, kin)
    assert len(program) == 1
    assert program[0].label == "taught"
    assert program[0].is_resolved
    assert program[0].grip(Arm.LEFT) == 1.0
    assert program[0].grip(Arm.RIGHT) == 0.0
    expected = kin.fk(kin.rest_q(), GRIPPER_TIP)
    for arm in ARMS:
        assert program[0].pose(arm).is_close(expected[arm], pos_tol=1e-4, ori_tol=1e-3)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolve_fills_in_missing_joints(kin, q_rest):
    poses = kin.fk(q_rest)
    program = Program()
    program.append(
        Waypoint(
            targets={
                arm: ArmTarget(pose=poses[arm].translated((0.0, -0.05, 0.02)))
                for arm in ARMS
            }
        )
    )
    assert not program[0].is_resolved
    assert program.resolve(kin, seed=q_rest) == []
    assert program[0].is_resolved
    for arm in ARMS:
        assert (
            kin.fk(program[0].full_q(kin))[arm].distance_to(program[0].pose(arm)) < 1e-3
        )


def test_resolve_reports_unreachable_waypoints(kin, q_rest):
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
    assert program.resolve(kin, seed=q_rest) == [0]
    assert not program[0].is_resolved


def test_resolve_skips_already_resolved_unless_forced(kin, q_rest):
    poses = kin.fk(q_rest)
    program = Program()
    program.append(
        Waypoint(
            targets={
                arm: ArmTarget(
                    pose=poses[arm], joints=np.full(7, 0.123, dtype=np.float32)
                )
                for arm in ARMS
            }
        )
    )
    program.resolve(kin, seed=q_rest)
    assert program[0].joints(Arm.LEFT)[0] == pytest.approx(0.123)
    program.resolve(kin, seed=q_rest, force=True)
    assert program[0].joints(Arm.LEFT)[0] != pytest.approx(0.123)


def test_full_q_is_none_until_resolved():
    class FakeKin:
        num_joints = 14

        def indices(self, arm):
            return list(range(7) if arm is Arm.LEFT else range(7, 14))

    waypoint = Waypoint(targets={arm: ArmTarget(pose=Pose.identity()) for arm in ARMS})
    assert waypoint.full_q(FakeKin()) is None
