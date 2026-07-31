"""End-to-end tests of the jog pendant's wiring.

These drive :class:`~almond_axol.cli.jog.JogApp` through the *same* handlers the
browser buttons are bound to, so they cover the parts a unit test of the motion
layer cannot: the callback-to-queue hand-off, the repaint guard that stops a
programmatic field write from bouncing back as an operator edit, and the path
from a button press to a joint command actually reaching the robot.

A real viser server is started on a scratch port. No browser is involved — the
server runs headless and every assertion is made against application state.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from almond_axol.cli.jog import JogApp, JogCmdConfig
from almond_axol.motion import ARMS, Arm, MoveType, Pose, Program, VirtualRobot

# Away from 8010 (a developer's own session) and 8002 (Sim).
TEST_PORT = 8123


def event(value=None):
    """A stand-in for the viser event object, whose ``.target`` is the handle."""
    return SimpleNamespace(target=SimpleNamespace(value=value))


@pytest.fixture
def app(kin, tmp_path):
    """A live JogApp against a virtual robot seeded at the ready pose."""
    cfg = JogCmdConfig(file=str(tmp_path / "program.json"), sim=True, port=TEST_PORT)
    q_ready = kin.ready_q()
    robot = VirtualRobot(
        np.append(kin.arm_q(q_ready, Arm.LEFT), 1.0),
        np.append(kin.arm_q(q_ready, Arm.RIGHT), 1.0),
    )
    application = JogApp(cfg, kin, robot)
    application._to_send = None
    application._playback_request = None
    try:
        yield application
    finally:
        application.server.stop()


def pump(app, solve: bool = True, ticks: int = 200) -> None:
    """Run the app loop's work: drain the queue, solve, then let motion settle.

    ``ticks`` advances the jog rate limiter, which walks the commanded
    configuration toward the solved one rather than jumping to it. Tests that
    care about where the arm ends up need it run to convergence.
    """
    for fn in app.pending.drain():
        fn()
    if solve and app._target_dirty:
        app._target_dirty = False
        app._solve_targets()
    # Mirror the real loop exactly: it only commands when something moved.
    # Pushing unconditionally makes "did the button reach the robot?" tests
    # pass even when the handler does nothing at all.
    for _ in range(ticks):
        moved = app._advance_commanded(1.0 / 30.0)
        if moved or app._to_send is not None:
            app._push_to_robot()
        if not moved:
            break


def tool_pose(app, arm: Arm = Arm.LEFT):
    return app.kin.fk_arm(app.q, arm, app.tool)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def test_starts_at_a_well_conditioned_pose(app):
    for arm in ARMS:
        assert not app.kin.is_singular(app.q, arm)


def test_pose_fields_match_the_arm(app):
    pose = tool_pose(app)
    assert app.ui["x"].value == pytest.approx(float(pose.position[0]), abs=1e-3)
    assert app.ui["y"].value == pytest.approx(float(pose.position[1]), abs=1e-3)
    assert app.ui["z"].value == pytest.approx(float(pose.position[2]), abs=1e-3)


# ---------------------------------------------------------------------------
# Cartesian jog
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis,sign", [(0, "+"), (0, "−"), (1, "+"), (1, "−"), (2, "+"), (2, "−")]
)
def test_translate_jog_moves_the_tool_along_that_world_axis(app, axis, sign):
    """Each button moves the tool along exactly one world axis, by the step size."""
    step = 0.02
    app.ui["step_mm"].value = step * 1e3
    before = tool_pose(app).position.copy()

    app._on_jog_translate(axis)(event(sign))
    pump(app)

    after = tool_pose(app).position
    delta = after - before
    expected = step if sign == "+" else -step
    assert delta[axis] == pytest.approx(expected, abs=2e-3)
    for other in range(3):
        if other != axis:
            assert abs(delta[other]) < 2e-3


def rotation_log(rotation):
    """Axis-angle vector of a rotation matrix."""
    import math

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


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", ["+", "−"])
def test_rotate_jog_turns_about_the_named_axis_with_the_right_sign(app, axis, sign):
    """Pins the axis and the direction, not just the amount.

    Asserting only ``angle_to == 10 deg`` passes if the button rotates about a
    different axis, or the opposite way — the exact class of bug that shipped
    once already in the jaw axes.
    """
    app.ui["frame_mode"].value = "World"
    app._sync_gizmos()
    app.ui["step_deg"].value = 10.0
    before = tool_pose(app)

    app._on_jog_rotate(axis)(event(sign))
    pump(app)

    after = tool_pose(app)
    delta = after.rotation.astype(np.float64) @ before.rotation.astype(np.float64).T
    expected = np.zeros(3)
    expected[axis] = (1.0 if sign == "+" else -1.0) * np.radians(10.0)
    assert np.allclose(rotation_log(delta), expected, atol=0.02), rotation_log(delta)
    assert after.distance_to(before) < 3e-3


def test_a_dead_jog_handler_is_detected(app):
    """Guards the test harness itself.

    ``pump`` used to command the robot unconditionally, so a jog handler that
    did nothing still left a plausible-looking command behind.
    """
    app._to_send = None
    app.ui["step_mm"].value = 20.0
    pump(app)  # no button pressed
    assert app._to_send is None


# ---------------------------------------------------------------------------
# The drag marker
#
# Every test here covers a bug that shipped: after a button jog the marker and
# the numeric fields stayed where they were, and the marker's arrows pointed
# along the tool axes even with the jog frame set to World — so the two ways of
# moving the arm disagreed about which direction "X" was.
# ---------------------------------------------------------------------------


def gizmo_pose(app, arm: Arm = Arm.LEFT):
    """The pose the on-screen marker is currently showing."""
    handle = app.gizmos[arm]
    return Pose.from_quat(
        np.asarray(handle.position, dtype=np.float32),
        np.asarray(handle.wxyz, dtype=np.float32),
    )


def drag_gizmo(app, pose: Pose, arm: Arm = Arm.LEFT):
    """Simulate the operator dragging a marker to ``pose``."""
    handle = SimpleNamespace(position=pose.position, wxyz=pose.quat)
    app._on_gizmo(arm)(SimpleNamespace(target=handle))
    pump(app)


def test_marker_follows_a_button_jog(app):
    """The marker has to track the arm, or the next drag teleports it back."""
    app.ui["step_mm"].value = 40.0
    app._on_jog_translate(2)(event("+"))
    pump(app)
    assert gizmo_pose(app).distance_to(tool_pose(app)) < 1e-4


def test_pose_fields_follow_a_button_jog(app):
    """The readout has to track the arm too."""
    before = float(app.ui["z"].value)
    app.ui["step_mm"].value = 30.0
    app._on_jog_translate(2)(event("+"))
    pump(app)
    assert float(app.ui["z"].value) == pytest.approx(before + 0.03, abs=3e-3)
    assert float(app.ui["z"].value) == pytest.approx(
        float(tool_pose(app).position[2]), abs=1e-3
    )


def test_marker_shows_world_axes_in_world_mode(app):
    """World jog frame means world-aligned arrows, whatever the wrist is doing."""
    app.ui["frame_mode"].value = "World"
    app._sync_gizmos()
    assert np.allclose(gizmo_pose(app).rotation, np.eye(3), atol=1e-5)
    # And the tool itself is *not* world-aligned, so this is a real distinction.
    assert tool_pose(app).angle_to(Pose.identity()) > 0.1


def test_marker_shows_tool_axes_in_tool_mode(app):
    app.ui["frame_mode"].value = "Tool"
    app._sync_gizmos()
    assert gizmo_pose(app).is_close(tool_pose(app), pos_tol=1e-4, ori_tol=1e-3)


def test_switching_jog_frame_reseats_the_marker(app):
    app.ui["frame_mode"].value = "Tool"
    app._sync_gizmos()
    assert not np.allclose(gizmo_pose(app).rotation, np.eye(3), atol=1e-3)
    app.ui["frame_mode"].value = "World"
    app._sync_gizmos()
    assert np.allclose(gizmo_pose(app).rotation, np.eye(3), atol=1e-5)


def test_marker_position_always_sits_on_the_tool(app):
    for mode in ("World", "Tool"):
        app.ui["frame_mode"].value = mode
        app._sync_gizmos()
        assert gizmo_pose(app).distance_to(tool_pose(app)) < 1e-4


def test_dragging_in_world_mode_translates_the_tool(app):
    """A drag must move the tool by the drag delta, not snap its orientation."""
    app.ui["frame_mode"].value = "World"
    app._sync_gizmos()
    before = tool_pose(app)
    drag_gizmo(app, gizmo_pose(app).translated((0.0, -0.05, 0.03)))
    after = tool_pose(app)
    assert np.allclose(after.position - before.position, [0.0, -0.05, 0.03], atol=3e-3)
    # Orientation must survive: the marker showed identity, the tool did not.
    assert after.angle_to(before) < 0.05


def test_dragging_in_world_mode_rotates_about_world_axes(app):
    app.ui["frame_mode"].value = "World"
    app._sync_gizmos()
    before = tool_pose(app)
    drag_gizmo(app, gizmo_pose(app).rotated(0.0, 0.0, 0.25))
    after = tool_pose(app)
    assert after.angle_to(before) == pytest.approx(0.25, abs=0.05)
    assert after.distance_to(before) < 5e-3


def test_dragging_in_tool_mode_still_works(app):
    app.ui["frame_mode"].value = "Tool"
    app._sync_gizmos()
    before = tool_pose(app)
    drag_gizmo(app, gizmo_pose(app).translated((0.0, -0.04, 0.0)))
    assert np.allclose(
        tool_pose(app).position - before.position, [0.0, -0.04, 0.0], atol=3e-3
    )


def test_marker_is_left_alone_while_it_is_being_dragged(app):
    """Writing to a handle under the cursor makes it stutter."""
    app._dragging.add(Arm.LEFT)
    stale = gizmo_pose(app)
    app.ui["step_mm"].value = 50.0
    app._on_jog_translate(2)(event("+"))
    pump(app)
    assert gizmo_pose(app).is_close(stale, pos_tol=1e-6, ori_tol=1e-5)
    # The other arm's marker is still maintained.
    app._dragging.discard(Arm.LEFT)
    app._sync_gizmos()
    assert gizmo_pose(app).distance_to(tool_pose(app)) < 1e-4


def test_marker_reseats_after_a_drag_ends(app):
    app.ui["frame_mode"].value = "World"
    app._sync_gizmos()
    app._on_drag(Arm.LEFT, True)(None)
    drag_gizmo(app, gizmo_pose(app).rotated(0.0, 0.0, 0.3))
    app._on_drag(Arm.LEFT, False)(None)
    pump(app)
    # Back to showing world axes, with the tool keeping the new orientation.
    assert np.allclose(gizmo_pose(app).rotation, np.eye(3), atol=1e-5)
    assert gizmo_pose(app).distance_to(tool_pose(app)) < 1e-4


def test_a_late_field_echo_cannot_undo_a_drag(app):
    """The race that made drags silently revert.

    viser delivers callbacks from a thread pool, so a repaint's ``on_update``
    can land *after* a drag has already moved the target. Comparing the fields
    against the current target made that echo look like a fresh edit and put
    the arm back. It has to be compared against what was last written instead.
    """
    app.ui["frame_mode"].value = "World"
    app._sync_gizmos()
    before = tool_pose(app)

    # A repaint handler that has not been delivered yet.
    app.pending.push(app._apply_pose_fields)
    # ...and now the operator drags, before it lands.
    app._on_gizmo(Arm.LEFT)(
        SimpleNamespace(
            target=SimpleNamespace(
                position=gizmo_pose(app).translated((0.0, -0.05, 0.0)).position,
                wxyz=gizmo_pose(app).quat,
            )
        )
    )
    pump(app)  # drains the stale handler first, exactly as the app loop does

    assert tool_pose(app).position[1] - before.position[1] == pytest.approx(
        -0.05, abs=3e-3
    ), "a stale field echo reverted the drag"


def test_a_late_field_echo_cannot_undo_a_button_jog(app):
    before = float(tool_pose(app).position[2])
    app.pending.push(app._apply_pose_fields)
    app.ui["step_mm"].value = 40.0
    app._on_jog_translate(2)(event("+"))
    pump(app)
    assert float(tool_pose(app).position[2]) - before == pytest.approx(0.04, abs=3e-3)


def test_typed_edit_still_wins_over_the_echo_guard(app):
    """The guard must not swallow a genuine edit."""
    target_z = float(app.ui["z"].value) + 0.05
    app.ui["z"].value = target_z
    pump(app)
    assert tool_pose(app).position[2] == pytest.approx(target_z, abs=3e-3)


def test_marker_follows_goto_and_playback(app):
    app.ui["step_mm"].value = 40.0
    add_waypoint(app, "home")
    app._on_jog_translate(2)(event("+"))
    pump(app)
    app.selected = 0
    app._goto_waypoint()
    pump(app)
    assert gizmo_pose(app).distance_to(tool_pose(app)) < 1e-4


def test_jog_reaches_the_robot_not_just_the_app(app):
    """The button has to end in a motion command, not only an internal pose."""
    app.ui["step_mm"].value = 20.0
    app._on_jog_translate(2)(event("+"))
    pump(app)
    assert app._to_send is not None
    left, right = app._to_send
    assert np.allclose(left[:7], app.kin.arm_q(app.q, Arm.LEFT), atol=1e-5)
    assert np.allclose(right[:7], app.kin.arm_q(app.q, Arm.RIGHT), atol=1e-5)


# ---------------------------------------------------------------------------
# Jog rate limiting
#
# IK says where to go, not how to get there. Commanding a solution outright is
# a setpoint step the robot chases as hard as its gains allow, and one large
# enough trips ``max_step_rad`` — which the robot handles by *silently dropping
# the command*, leaving the app believing the arm moved.
# ---------------------------------------------------------------------------


def test_a_big_jog_never_commands_a_step_over_the_robot_limit(app):
    """The single most important hardware property of the jog path."""
    from almond_axol.robot.config import AxolConfig

    limit = AxolConfig().max_step_rad
    app.ui["step_mm"].value = 100.0
    for _ in range(4):
        app._on_jog_translate(1)(event("-"))
    for fn in app.pending.drain():
        fn()
    app._target_dirty = False
    app._solve_targets()

    previous = app._commanded.copy()
    steps = 0
    while app._advance_commanded(1.0 / 30.0):
        assert np.max(np.abs(app._commanded - previous)) <= limit, (
            "command would be dropped"
        )
        previous = app._commanded.copy()
        steps += 1
        assert steps < 2000
    assert steps > 1, "a 400 mm jog should not arrive in a single tick"


def test_the_commanded_pose_converges_on_the_solved_one(app):
    app.ui["step_mm"].value = 60.0
    app._on_jog_translate(2)(event("+"))
    pump(app)
    assert np.allclose(app._commanded, app.q, atol=1e-6)


def test_a_small_jog_arrives_promptly(app):
    """Rate limiting must not make ordinary nudges feel sluggish."""
    app.ui["step_mm"].value = 2.0
    app._on_jog_translate(2)(event("+"))
    for fn in app.pending.drain():
        fn()
    app._target_dirty = False
    app._solve_targets()
    ticks = 0
    while app._advance_commanded(1.0 / 30.0):
        ticks += 1
    assert ticks <= 3, f"a 2 mm jog took {ticks} ticks"


def test_goto_waypoint_ramps_rather_than_teleporting(app):
    add_waypoint(app, "home")
    app.ui["step_mm"].value = 80.0
    for _ in range(3):
        app._on_jog_translate(1)(event("-"))
    pump(app)

    app.selected = 0
    app._goto_waypoint()
    assert not np.allclose(app._commanded, app.q), "goto should leave a gap to ramp"
    while app._advance_commanded(1.0 / 30.0):
        pass
    assert np.allclose(app._commanded, app.q, atol=1e-6)


async def test_playback_leaves_the_commanded_pose_consistent(app):
    """After a plan streams, jogging must resume from where it left the arm."""
    app.cfg.rate_hz = 50.0
    app.commander.rate_hz = 50.0
    app.ui["step_mm"].value = 40.0
    add_waypoint(app, "a")
    app._on_jog_translate(2)(event("+"))
    pump(app)
    add_waypoint(app, "b")

    await app._run_playback(loops=1)
    assert np.allclose(app._commanded, app.q, atol=1e-6)


def test_tool_frame_jog_differs_from_world_frame_jog(app):
    """Switching the jog frame must actually change the direction of travel.

    Jogged along Z, not X: the ready pose is rolled about X, which leaves the
    tool's X axis parallel to the world's, so an X jog is identical in both
    frames and would pass this test without proving anything.
    """
    app.ui["step_mm"].value = 30.0
    start_q = app.q.copy()
    start_targets = dict(app.targets)

    app.ui["frame_mode"].value = "World"
    app._on_jog_translate(2)(event("+"))
    pump(app)
    world_delta = (
        tool_pose(app).position - app.kin.fk_arm(start_q, Arm.LEFT, app.tool).position
    )

    app.q = start_q.copy()
    app.targets = start_targets
    app.ui["frame_mode"].value = "Tool"
    app._on_jog_translate(2)(event("+"))
    pump(app)
    tool_delta = (
        tool_pose(app).position - app.kin.fk_arm(start_q, Arm.LEFT, app.tool).position
    )

    assert np.linalg.norm(world_delta - tool_delta) > 5e-3


def test_sync_both_moves_both_arms(app):
    app.ui["step_mm"].value = 25.0
    app.ui["sync_both"].value = True
    before = {arm: tool_pose(app, arm).position.copy() for arm in ARMS}

    app._on_jog_translate(2)(event("+"))
    pump(app)

    for arm in ARMS:
        assert tool_pose(app, arm).position[2] - before[arm][2] == pytest.approx(
            0.025, abs=3e-3
        )


def test_single_arm_jog_leaves_the_other_alone(app):
    app.ui["step_mm"].value = 25.0
    app.ui["sync_both"].value = False
    app.ui["arm"].value = Arm.LEFT.label
    before_right = tool_pose(app, Arm.RIGHT).position.copy()

    app._on_jog_translate(2)(event("+"))
    pump(app)

    assert np.allclose(tool_pose(app, Arm.RIGHT).position, before_right, atol=1e-4)


def test_typed_pose_is_applied(app):
    """The numeric fields are an input, not just a readout."""
    target_z = float(app.ui["z"].value) + 0.04
    app.ui["z"].value = target_z
    pump(app)
    assert tool_pose(app).position[2] == pytest.approx(target_z, abs=2e-3)


def test_repaint_does_not_bounce_back_as_an_edit(app):
    """Repainting the fields must not re-enter as an operator edit.

    viser runs plain callbacks in a thread pool, so a repaint's ``on_update``
    can be delivered long after the repaint finished — a timing flag cannot
    make this safe. What makes it safe is that the actions are idempotent, so
    the property to assert is that running whatever got queued changes nothing,
    not that nothing got queued.
    """
    app.pending.drain()
    before_target = app.targets[Arm.LEFT]
    before_q = app.q.copy()

    app._sync_pose_fields()
    app._refresh_waypoints()
    pump(app)

    assert app.targets[Arm.LEFT].is_close(before_target, pos_tol=1e-6, ori_tol=1e-5)
    assert np.allclose(app.q, before_q, atol=1e-9)
    assert app._target_dirty is False


def test_repeated_repaints_do_not_drift(app):
    """A hundred repaint cycles must not walk the target anywhere."""
    start = app.targets[Arm.LEFT]
    for _ in range(100):
        app._sync_pose_fields()
        pump(app)
    assert app.targets[Arm.LEFT].is_close(start, pos_tol=1e-6, ori_tol=1e-5)


def test_reset_returns_to_the_ready_pose(app):
    app.ui["step_mm"].value = 50.0
    app._on_jog_translate(2)(event("+"))
    pump(app)
    app._go_rest()
    assert np.allclose(app.q, app.kin.ready_q(), atol=1e-6)


# ---------------------------------------------------------------------------
# Waypoints
# ---------------------------------------------------------------------------


def add_waypoint(app, label: str = "") -> None:
    app._add_waypoint()
    if label:
        app.ui["label"].value = label
        app._set_label()


def test_add_waypoint_captures_the_current_pose(app):
    pose = tool_pose(app)
    add_waypoint(app)
    assert len(app.program) == 1
    assert app.program[0].pose(Arm.LEFT).is_close(pose, pos_tol=1e-4, ori_tol=1e-3)
    # Recorded resolved, so replay reproduces this exact configuration.
    assert app.program[0].is_resolved


def test_added_waypoints_are_written_to_disk_immediately(app):
    add_waypoint(app)
    saved = Program.load(app.cfg.file, app.kin)
    assert len(saved) == 1


def test_waypoints_are_added_after_the_selection(app):
    app.ui["step_mm"].value = 30.0
    for i in range(3):
        add_waypoint(app, f"w{i}")
        app._on_jog_translate(2)(event("+"))
        pump(app)
    assert [w.label for w in app.program] == ["w0", "w1", "w2"]
    assert app.selected == 2


def test_reorder_moves_the_selection_with_it(app):
    app.ui["step_mm"].value = 30.0
    for i in range(3):
        add_waypoint(app, f"w{i}")
        app._on_jog_translate(2)(event("+"))
        pump(app)

    app.selected = 2
    app._on_reorder(event("▲ Up"))
    pump(app, solve=False)
    assert [w.label for w in app.program] == ["w0", "w2", "w1"]
    assert app.selected == 1

    app._on_reorder(event("▼ Down"))
    pump(app, solve=False)
    assert [w.label for w in app.program] == ["w0", "w1", "w2"]
    assert app.selected == 2


def test_duplicate_and_delete(app):
    add_waypoint(app, "only")
    app._on_edit(event("Duplicate"))
    pump(app, solve=False)
    assert len(app.program) == 2
    app._on_edit(event("Delete"))
    pump(app, solve=False)
    assert len(app.program) == 1


def test_delete_keeps_the_selection_in_range(app):
    app.ui["step_mm"].value = 30.0
    for i in range(2):
        add_waypoint(app, f"w{i}")
        app._on_jog_translate(2)(event("+"))
        pump(app)
    app.selected = 1
    app._on_edit(event("Delete"))
    pump(app, solve=False)
    assert app.selected == 0
    assert 0 <= app.selected < len(app.program)


def test_move_type_is_editable_per_waypoint(app):
    add_waypoint(app)
    app.ui["move_type"].value = MoveType.JOINT.label
    app._set_move_type()
    assert app.program[0].move is MoveType.JOINT


def test_goto_selected_moves_the_arm_there(app):
    """Editing then returning must land on the stored pose, not near it."""
    add_waypoint(app, "home")
    home = tool_pose(app)

    app.ui["step_mm"].value = 60.0
    app._on_jog_translate(2)(event("+"))
    pump(app)
    assert tool_pose(app).distance_to(home) > 0.05

    app.selected = 0
    app._goto_waypoint()
    assert tool_pose(app).is_close(home, pos_tol=1e-4, ori_tol=1e-3)


def test_clear_empties_the_program_and_the_selection(app):
    add_waypoint(app)
    app._clear_program()
    assert len(app.program) == 0
    assert app.selected == -1


def test_editing_with_no_selection_is_reported_not_raised(app):
    app.selected = -1
    app._update_waypoint()
    assert "Select a waypoint" in app._status


def test_waypoint_markers_are_drawn_and_cleaned_up(app):
    app.ui["step_mm"].value = 30.0
    for _ in range(2):
        add_waypoint(app)
        app._on_jog_translate(2)(event("+"))
        pump(app)
    assert len(app._waypoint_nodes) > 0
    app._clear_program()
    assert app._waypoint_nodes == []


# ---------------------------------------------------------------------------
# Persistence and playback
# ---------------------------------------------------------------------------


def test_save_and_reload_round_trip(app):
    app.ui["step_mm"].value = 30.0
    for i in range(3):
        add_waypoint(app, f"w{i}")
        app._on_jog_translate(2)(event("+"))
        pump(app)
    poses = [w.pose(Arm.LEFT).position.copy() for w in app.program]

    app._save()
    app._reload()

    assert [w.label for w in app.program] == ["w0", "w1", "w2"]
    for before, waypoint in zip(poses, app.program):
        assert np.allclose(waypoint.pose(Arm.LEFT).position, before, atol=1e-4)


async def test_playback_visits_every_waypoint(app):
    """The whole point: build a program by jogging, then fly it."""
    app.cfg.rate_hz = 50.0
    app.commander.rate_hz = 50.0
    app.ui["step_mm"].value = 40.0
    for i in range(3):
        add_waypoint(app, f"w{i}")
        app._on_jog_translate(2)(event("+"))
        pump(app)

    last = app.program[-1]
    await app._run_playback(loops=1)

    assert not app._playing
    reached = app.kin.fk(app.q, app.tool)
    for arm in ARMS:
        assert reached[arm].distance_to(last.pose(arm)) < 5e-3


async def test_playback_can_be_stopped(app):
    app.cfg.rate_hz = 50.0
    app.commander.rate_hz = 50.0
    app.ui["step_mm"].value = 40.0
    for i in range(2):
        add_waypoint(app, f"w{i}")
        app._on_jog_translate(2)(event("+"))
        pump(app)

    # Stop *during* the stream, not before it: setting the flag up front only
    # ever exercises the very first tick of the very first leg.
    original = app.robot.motion_control
    sent = 0

    async def stop_partway(left=None, right=None):
        nonlocal sent
        sent += 1
        if sent == 15:
            app.stop_playback.set()
        await original(left=left, right=right)

    app.robot.motion_control = stop_partway
    await app._run_playback(loops=1)
    assert sent >= 15
    assert "Stopped" in app._status


async def test_playback_failure_is_reported_not_raised(app):
    """An unreachable program must leave the UI usable."""
    from almond_axol.motion.frames import Pose
    from almond_axol.motion.program import ArmTarget, Waypoint

    app.program.append(
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
    await app._run_playback(loops=1)
    assert not app._playing
    assert "Cannot play" in app._status or "error" in app._status.lower()


async def test_jog_still_works_after_playback(app):
    """Playback adopts wherever it left the arm, so jogging resumes from there."""
    app.cfg.rate_hz = 50.0
    app.commander.rate_hz = 50.0
    app.ui["step_mm"].value = 40.0
    add_waypoint(app, "a")
    app._on_jog_translate(2)(event("+"))
    pump(app)
    add_waypoint(app, "b")

    await app._run_playback(loops=1)

    before = tool_pose(app).position.copy()
    app.ui["step_mm"].value = 20.0
    app._on_jog_translate(0)(event("+"))
    pump(app)
    assert tool_pose(app).position[0] - before[0] == pytest.approx(0.02, abs=3e-3)


async def test_a_saved_program_plays_from_a_fresh_session(app, kin, tmp_path):
    """The file is the deliverable: another process must be able to fly it.

    This is what carrying a program from a laptop to the robot depends on.
    """
    app.ui["step_mm"].value = 40.0
    for i in range(2):
        add_waypoint(app, f"w{i}")
        app._on_jog_translate(2)(event("+"))
        pump(app)
    app._save()

    from almond_axol.motion import MotionCommander

    program = Program.load(app.cfg.file, kin)
    assert len(program) == 2

    q_start = kin.ready_q()
    robot = VirtualRobot(
        np.append(kin.arm_q(q_start, Arm.LEFT), 1.0),
        np.append(kin.arm_q(q_start, Arm.RIGHT), 1.0),
    )
    commander = MotionCommander(kin, robot, rate_hz=50.0)
    report = await commander.execute(commander.plan_program(program, q_start))
    assert report.legs_completed == 2

    q_final, _ = await commander.current_state()
    for arm in ARMS:
        assert (
            kin.fk(q_final, program.tool)[arm].distance_to(program[-1].pose(arm)) < 5e-3
        )


def test_asyncio_is_not_required_for_edits(app):
    """Edit operations must not need a running loop — they run on the UI thread."""
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()
    add_waypoint(app)
    assert len(app.program) == 1


# ---------------------------------------------------------------------------
# Coverage the adversarial review named as missing
# ---------------------------------------------------------------------------


async def test_loops_zero_actually_repeats(app):
    """The UI says "0 = forever". It used to play exactly one cycle."""
    app.cfg.rate_hz = 50.0
    app.commander.rate_hz = 50.0
    app.ui["step_mm"].value = 40.0
    add_waypoint(app, "a")
    app._on_jog_translate(2)(event("+"))
    pump(app)
    add_waypoint(app, "b")

    passes = 0
    original = app.commander.execute

    async def counting(planned, **kwargs):
        nonlocal passes
        passes += 1
        if passes >= 3:
            app.stop_playback.set()
        return await original(planned, **kwargs)

    app.commander.execute = counting
    await app._run_playback(loops=0)
    assert passes >= 3, f"loops=0 ran the program {passes} time(s)"
    assert not app._playing


async def test_operator_actions_are_ignored_during_playback(app):
    """Playback owns the arm; a jog landing mid-stream desynchronises the app."""
    app.cfg.rate_hz = 50.0
    app.commander.rate_hz = 50.0
    app.ui["step_mm"].value = 40.0
    add_waypoint(app, "a")
    app._on_jog_translate(2)(event("+"))
    pump(app)
    add_waypoint(app, "b")

    app._playing = True
    try:
        before = app.q.copy()
        app._when_idle(app._go_rest)()
        app._on_jog_translate(2)(event("+"))
        for fn in app.pending.drain():
            fn()
        assert np.allclose(app.q, before), "an action ran while playing"
        assert "Stop" in app._status
    finally:
        app._playing = False


async def test_a_failed_playback_still_re_reads_the_robot(app):
    """After a mid-stream failure the app must not describe a stale pose.

    Left stale, the operator's next small jog becomes a large step the robot
    silently drops — and the pendant never recovers.
    """
    app.cfg.rate_hz = 50.0
    app.commander.rate_hz = 50.0
    app.ui["step_mm"].value = 40.0
    add_waypoint(app, "a")
    app._on_jog_translate(2)(event("+"))
    pump(app)
    add_waypoint(app, "b")

    calls = 0
    original = app.robot.motion_control

    async def fail_partway(left=None, right=None):
        nonlocal calls
        calls += 1
        if calls == 20:
            raise RuntimeError("CAN write failed")
        await original(left=left, right=right)

    app.robot.motion_control = fail_partway
    await app._run_playback(loops=1)
    app.robot.motion_control = original

    assert not app._playing
    q_robot, _ = await app.commander.current_state()
    assert np.allclose(app.q, q_robot, atol=1e-5), "app state diverged from the robot"
    assert np.allclose(app._commanded, q_robot, atol=1e-5)


async def test_the_commander_knows_what_the_jog_loop_commanded(app):
    """Otherwise playback and park ease in from a pose the arm left long ago."""
    app.ui["step_mm"].value = 50.0
    for _ in range(3):
        app._on_jog_translate(1)(event("-"))
        pump(app)
    assert app.commander._last_commanded is not None
    assert np.allclose(app.commander._last_commanded, app._commanded, atol=1e-6)


def test_reset_to_ready_ramps_rather_than_teleporting(app):
    """The bug: Reset synced ``_commanded``, skipping the rate limiter.

    Measured before the fix: a single 1.597 rad step against a 0.5 rad limit,
    silently dropped by the robot, after which the pendant never recovered.
    """
    from almond_axol.robot.config import AxolConfig

    app.ui["step_mm"].value = 100.0
    for _ in range(4):
        app._on_jog_translate(1)(event("-"))
    pump(app)

    app._go_rest()
    first_step = np.max(np.abs(app.q - app._commanded))
    assert first_step > AxolConfig().max_step_rad, "test no longer exercises a big move"

    previous = app._commanded.copy()
    while app._advance_commanded(1.0 / 30.0):
        assert np.max(np.abs(app._commanded - previous)) <= AxolConfig().max_step_rad
        previous = app._commanded.copy()
    assert np.allclose(app._commanded, app.q, atol=1e-6)


def test_a_stranded_drag_flag_expires(app):
    """viser's thread pool can deliver drag-end before drag-start.

    A permanently-set flag means the marker is never re-seated, and the next
    nudge of it commands the whole accumulated gap.
    """
    app._dragging.add(Arm.LEFT)
    app.ui["step_mm"].value = 60.0
    app._on_jog_translate(2)(event("+"))
    pump(app)
    assert Arm.LEFT in app._dragging
    for _ in range(200):
        app._expire_stale_drags()
        if Arm.LEFT not in app._dragging:
            break
    assert Arm.LEFT not in app._dragging, "stale drag flag never expired"
    assert gizmo_pose(app).distance_to(tool_pose(app)) < 1e-4


def test_the_render_map_sends_each_joint_to_its_own_viser_slot(app):
    """A mis-mapped joint renders the wrong arm bending."""
    names = app.viser_urdf.get_actuated_joint_names()
    solver_names = app.kin.solver.joint_names
    for viser_index, solver_index in enumerate(app._viser_map):
        if solver_index >= 0:
            assert names[viser_index] == solver_names[solver_index]
    for viser_index, arm in app._viser_jaws.items():
        assert names[viser_index].startswith(arm.value)
        assert "jaw" in names[viser_index]
    mapped = {i for i in app._viser_map if i >= 0}
    assert len(mapped) == len(solver_names), "not every arm joint is rendered"
