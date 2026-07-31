"""
axol jog

A Cartesian teach pendant in the browser. Drag either gripper to a pose and the
arm follows by inverse kinematics; step-jog it along world or tool axes; record
where it is as a waypoint; reorder, retime, and edit the resulting program;
play it back; and save a file that runs unchanged on the real robot.

This is the piece the SDK was missing. ``axol waypoints`` can only *record*
poses the operator has already hand-guided the arm into, which needs hardware
and a person standing next to it. Everything here works from a pose — typed,
dragged, or stepped — so a program can be built and rehearsed in simulation
and carried to the robot.

Two motion primitives per waypoint, the same pair every industrial controller
exposes: ``MoveJ`` interpolates in joint space (fast, always feasible, tool
sweeps an arc) and ``MoveL`` holds the tool to a straight world-frame line
(what an approach or an insertion needs). Both are planned in full before
anything moves, so an unreachable pose is reported while the robot is standing
still.

    axol jog                          # simulator, default program file
    axol jog --file bolt_test.json    # a named program
    axol jog --sim False              # drive the real robot (Linux + CAN)

The simulator is a *visualiser*, not a physics engine: it replays exactly what
it is commanded and will show a flawless insertion regardless of what the real
arm would do. It answers reachability, path shape, and program logic. It cannot
answer repeatability or absolute accuracy — only hardware can.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import visual_urdf_path
from ..kinematics.config import KinematicsConfig
from ..kinematics.path import PathPlanningError
from ..motion import (
    ARMS,
    Arm,
    CartesianKinematics,
    MotionCommander,
    MoveType,
    Pose,
    Program,
    UnreachableError,
    VirtualRobot,
    default_tool,
)
from ..motion.commander import Grips, PlannedLeg
from ..motion.gripper import MAX_TRAVEL as JAW_TRAVEL
from ..motion.kinematics import SINGULARITY_THRESHOLD
from ..motion.program import GRIP_OPEN, ArmTarget, Waypoint
from ..robot.base import RobotBase
from ..robot.config import AxolConfig
from ..teleop.config import VRTeleopConfig
from ..utils.ports import reclaim_port
from .config import LogLevel, normalize_bool_flags, parse

_logger = logging.getLogger(__name__)

DEFAULT_PORT = 8010
"""Viewer port. Deliberately not 8002 — ``Sim`` owns that, and an operator may
well want a teleop preview and this pendant open at the same time."""

UI_RATE_HZ = 30.0
"""How often the app solves IK and repaints. IK costs about 2 ms, so this is
far from the limiting factor; it is chosen to feel immediate while dragging."""

DRAG_IDLE_TICKS = 45
"""App-loop ticks (1.5 s at 30 Hz) a drag may go without a gizmo update before
its flag is treated as stranded. See :meth:`JogApp._expire_stale_drags`."""

MAX_IO_FAILURES = 30
"""Consecutive robot I/O failures tolerated before the loop gives up.

A single CAN timeout should not kill a pendant mid-session, but a robot that
has genuinely gone away should not be commanded into the void for ever."""

JOG_JOINT_SPEED = 0.6
"""Peak joint speed (rad/s) while jogging — about 34 deg/s.

IK gives the configuration the operator asked for; it does not give a way to
*get* there. Commanding it outright is a step change in the setpoint, which on
hardware the impedance controller chases as hard as its gains allow. A typed
pose or a long gizmo drag can be a whole workspace away, so the setpoint is
walked toward the target at this rate instead of jumping to it.

Deliberately slow. This is a pendant: a jog is an operator nudging a real arm
near a real fixture, not a trajectory. Playback is unaffected — those legs are
already speed-profiled by the planner and stream at the control rate.

It also keeps commands under ``AxolConfig.max_step_rad`` (0.5 rad), which the
robot enforces by **silently dropping** the command — leaving the app believing
the arm moved when it did not."""


def _default_file() -> str:
    return str(Path.home() / ".almond" / "program.json")


@dataclass
class JogCmdConfig:
    """Config for ``axol jog``.

    The program file is the deliverable: it is written on every edit, holds
    Cartesian poses plus the joint configuration that reaches each one, and is
    read by this command and by playback on hardware alike.

    Playback speeds follow the program's own settings, editable in the UI and
    stored in the file, so a program that has been tuned slow for an insertion
    stays slow wherever it runs.
    """

    axol: AxolConfig = field(default_factory=AxolConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    file: str = field(default_factory=_default_file)
    """Program file to edit and play back (JSON)."""
    sim: bool = True
    """Run against the built-in visualiser instead of the robot."""
    port: int = DEFAULT_PORT
    """Port for the browser UI."""
    left_channel: str | None = None
    """CAN channel for the left arm. Defaults to the standard channel on hardware."""
    right_channel: str | None = None
    """CAN channel for the right arm."""
    rate_hz: float = 250.0
    """Control rate for planning and playback."""
    telemetry_hz: float = 500.0
    log_level: LogLevel = "INFO"


class _Pending:
    """A thread-safe queue of work handed from viser callbacks to the app loop.

    viser dispatches GUI callbacks on its own websocket threads. Solving IK,
    planning, and streaming a trajectory all touch state the app loop owns, so
    callbacks do nothing but enqueue a closure and return — which also keeps the
    UI responsive while a plan is being computed.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[Callable[[], Any]] = queue.Queue()

    def push(self, fn: Callable[[], Any]) -> None:
        self._q.put(fn)

    def drain(self) -> list[Callable[[], Any]]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out


class JogApp:
    """The pendant: viser scene, GUI, and the loop that ties them to the robot.

    Args:
        cfg:   Parsed command configuration.
        kin:   Shared Cartesian kinematics (built once; the JAX compile is slow).
        robot: The robot to command — virtual or real.
    """

    def __init__(
        self, cfg: JogCmdConfig, kin: CartesianKinematics, robot: RobotBase
    ) -> None:
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf

        self.cfg = cfg
        self.kin = kin
        self.robot = robot
        self.tool = kin.tool
        self.pending = _Pending()
        self.stop_playback = threading.Event()
        self.quit = threading.Event()

        self.program = Program.load(cfg.file, kin)
        self.program.tool = self.tool
        self.selected = 0 if len(self.program) else -1

        # Live state: the joint vector the app believes the robot is at, and
        # the pose each gizmo is asking for.
        self.q = kin.ready_q()
        # What has actually been sent, as distinct from what IK was asked for.
        # The gap between them is walked down at JOG_JOINT_SPEED.
        self._commanded = self.q.copy()
        self.grips: dict[Arm, float] = {arm: GRIP_OPEN for arm in ARMS}
        self.targets: dict[Arm, Pose] = kin.fk(self.q, self.tool)
        self._target_dirty = False
        self._suppress = False
        self._dragging: set[Arm] = set()
        self._drag_seen: dict[Arm, int] = {}
        self._io_failures = 0
        self._playback_task: asyncio.Task | None = None
        self._fields_written: tuple[float, ...] = ()
        self._gizmo_offset: dict[Arm, Pose] = {arm: Pose.identity() for arm in ARMS}
        self._playing = False
        self._status = "Ready."
        self._ik_note = ""

        reclaim_port(cfg.port)
        self.server = viser.ViserServer(port=cfg.port, label="Axol Cartesian Jog")
        # Display model, not the control model: it carries the real gripper CAD
        # and a prismatic joint per jaw. Falls back to the control model when
        # tools/build_visual_urdf.py has not been run.
        urdf_path = visual_urdf_path()
        urdf = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(urdf_path.parent))
        self.viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=urdf,
            root_node_name="/robot",
            load_meshes=True,
            load_collision_meshes=False,
        )
        # Map the viewer's joint order onto the solver's joint vector by name,
        # so neither side has to assume the other's ordering.
        solver_index = {n: i for i, n in enumerate(kin.solver.joint_names)}
        viser_names = self.viser_urdf.get_actuated_joint_names()
        self._viser_map = [solver_index.get(name, -1) for name in viser_names]
        self._n_viser = len(self._viser_map)
        # The jaw joints exist only in the display model, so they are driven
        # from the commanded gripper opening rather than from the joint vector.
        self._viser_jaws = {
            i: (Arm.LEFT if name.startswith("left") else Arm.RIGHT)
            for i, name in enumerate(viser_names)
            if "jaw" in name
        }
        if self._viser_jaws:
            _logger.info(
                "display model has articulated jaws (%d joints)", len(self._viser_jaws)
            )

        self.commander = MotionCommander(
            kin,
            robot,
            rate_hz=cfg.rate_hz,
            has_gripper=cfg.axol.has_gripper,
        )

        self._waypoint_nodes: list[Any] = []
        # GUI first: the gizmos read the Jog frame setting to decide which way
        # their arrows point, so the control has to exist before the scene.
        self._build_gui()
        self._build_scene()
        self._render(self.q)
        self._refresh_waypoints()

    # -- scene -----------------------------------------------------------

    def _build_scene(self) -> None:
        """Grid, world axes, and a draggable gizmo per arm."""
        self.server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0, 0, 0))
        # A visible reminder of which way the world axes point — the frame is
        # X-left / Y-back / Z-up, which is not what anyone assumes.
        self.server.scene.add_frame(
            "/world_axes", axes_length=0.25, axes_radius=0.006, origin_radius=0.012
        )
        self.gizmos: dict[Arm, Any] = {}
        for arm in ARMS:
            gizmo = self._gizmo_pose(arm)
            self.gizmos[arm] = self.server.scene.add_transform_controls(
                f"/target_{arm.value}",
                scale=0.22,
                position=tuple(float(v) for v in gizmo.position),
                wxyz=tuple(float(v) for v in gizmo.quat),
            )
            self.gizmos[arm].on_update(self._on_gizmo(arm))
            self.gizmos[arm].on_drag_start(self._on_drag(arm, True))
            self.gizmos[arm].on_drag_end(self._on_drag(arm, False))
        self._sync_gizmos()

    @property
    def _tool_frame(self) -> bool:
        """True when jogging (and the gizmo) should follow the tool's own axes."""
        return self.ui["frame_mode"].value == "Tool"

    def _gizmo_pose(self, arm: Arm) -> Pose:
        """Where the draggable marker sits, and which way its arrows point.

        The marker always sits *on* the tool point. Its orientation follows the
        **Jog frame** setting, so the arrows the operator drags are the same
        axes the ± buttons step along:

        - **World** — axis-aligned with the world, so dragging the red arrow
          moves along world X wherever the wrist happens to be pointing.
        - **Tool** — aligned with the tool, so dragging the blue arrow runs
          along the approach direction.

        Showing tool axes while the jog buttons move in world coordinates is
        the mismatch this fixes: the two controls disagreed about what "X"
        meant.
        """
        target = self.targets[arm]
        if self._tool_frame:
            return target
        return Pose(target.position, np.eye(3, dtype=np.float32))

    def _on_drag(self, arm: Arm, started: bool) -> Callable[[Any], None]:
        """Track which gizmos are being held, so a repaint cannot fight a drag."""

        def handler(_event: Any) -> None:
            if started:
                self._dragging.add(arm)
                self._drag_seen[arm] = 0
            else:
                self._dragging.discard(arm)
                self._drag_seen.pop(arm, None)
                # Re-seat the marker once the operator lets go: in world mode
                # a rotation drag leaves it turned, and it should go back to
                # showing world axes.
                self.pending.push(self._sync_gizmos)

        return handler

    def _expire_stale_drags(self) -> None:
        """Clear a drag flag that no longer has a drag behind it.

        viser dispatches every scene callback onto a thread pool with no
        ordering guarantee between them, so on a quick click the drag-end can
        be delivered *before* the drag-start. The arm would then stay flagged
        for ever, its marker never re-seated — and the next nudge of that
        marker, now far from the tool, commands the whole accumulated gap. A
        drag with no gizmo updates for this long was not a real drag.
        """
        for arm in list(self._dragging):
            self._drag_seen[arm] = self._drag_seen.get(arm, 0) + 1
            if self._drag_seen[arm] > DRAG_IDLE_TICKS:
                _logger.debug("clearing a stale drag flag on the %s arm", arm.value)
                self._dragging.discard(arm)
                self._drag_seen.pop(arm, None)
                self._sync_gizmos()

    def _on_gizmo(self, arm: Arm) -> Callable[[Any], None]:
        """Gizmo drag handler: record the request, let the loop solve it."""

        def handler(event: Any) -> None:
            if self._playing:
                return
            handle = event.target
            self._drag_seen[arm] = 0
            gizmo = Pose.from_quat(
                np.asarray(handle.position, dtype=np.float32),
                np.asarray(handle.wxyz, dtype=np.float32),
            )
            # The marker and the tool are not the same frame in world mode, so
            # map through the offset recorded when the marker was last seated.
            self.targets[arm] = gizmo * self._gizmo_offset[arm]
            self._target_dirty = True

        return handler

    def _sync_gizmos(self) -> None:
        """Re-seat the markers on the current targets.

        Also records, per arm, the fixed transform from marker frame to tool
        target. Any later marker pose maps to a tool pose through it, which is
        what lets the marker show world axes while still driving a tool pose.
        An arm being dragged right now is skipped — writing to a handle under
        the operator's cursor makes it stutter.
        """
        for arm in ARMS:
            if arm in self._dragging:
                continue
            gizmo = self._gizmo_pose(arm)
            self._gizmo_offset[arm] = gizmo.inverse() * self.targets[arm]
            handle = self.gizmos[arm]
            handle.position = tuple(float(v) for v in gizmo.position)
            handle.wxyz = tuple(float(v) for v in gizmo.quat)

    def _render(self, q: np.ndarray, grips: Grips | None = None) -> None:
        """Draw the robot at joint vector ``q``, with its jaws at ``grips``."""
        q = np.asarray(q, dtype=np.float32)
        out = np.zeros(self._n_viser, dtype=float)
        for i, source in enumerate(self._viser_map):
            if source >= 0:
                out[i] = float(q[source])
        openings = (
            {arm: self.grips[arm] for arm in ARMS}
            if grips is None
            else {Arm.LEFT: grips[0], Arm.RIGHT: grips[1]}
        )
        for index, arm in self._viser_jaws.items():
            out[index] = float(np.clip(openings[arm], 0.0, 1.0)) * JAW_TRAVEL
        self.viser_urdf.update_cfg(out)

    # -- gui -------------------------------------------------------------

    def _build_gui(self) -> None:
        gui = self.server.gui
        self.ui: dict[str, Any] = {}

        with gui.add_folder("Status"):
            self.ui["status"] = gui.add_text(
                "State", initial_value=self._status, disabled=True
            )
            self.ui["ik"] = gui.add_text("IK", initial_value="—", disabled=True)

        with gui.add_folder("Target"):
            self.ui["arm"] = gui.add_dropdown(
                "Arm", options=[a.label for a in ARMS], initial_value=Arm.LEFT.label
            )
            self.ui["arm"].on_update(
                lambda _: self.pending.push(self._sync_pose_fields)
            )
            self.ui["frame_mode"] = gui.add_dropdown(
                "Jog frame", options=["World", "Tool"], initial_value="World"
            )
            self.ui["frame_mode"].on_update(self._guard(self._sync_gizmos))
            self.ui["sync_both"] = gui.add_checkbox("Move both arms together", False)

        with gui.add_folder("Pose (world, metres / degrees)"):
            self.ui["x"] = gui.add_number("X  (left +)", initial_value=0.0, step=0.001)
            self.ui["y"] = gui.add_number("Y  (back +)", initial_value=0.0, step=0.001)
            self.ui["z"] = gui.add_number("Z  (up +)", initial_value=0.0, step=0.001)
            self.ui["roll"] = gui.add_number("Roll", initial_value=0.0, step=1.0)
            self.ui["pitch"] = gui.add_number("Pitch", initial_value=0.0, step=1.0)
            self.ui["yaw"] = gui.add_number("Yaw", initial_value=0.0, step=1.0)
            for key in ("x", "y", "z", "roll", "pitch", "yaw"):
                self.ui[key].on_update(
                    lambda _: self.pending.push(self._apply_pose_fields)
                )

        with gui.add_folder("Jog"):
            self.ui["step_mm"] = gui.add_slider(
                "Step (mm)", min=0.1, max=100.0, step=0.1, initial_value=10.0
            )
            self.ui["step_deg"] = gui.add_slider(
                "Step (deg)", min=0.1, max=45.0, step=0.1, initial_value=5.0
            )
            for axis, label in enumerate(("X", "Y", "Z")):
                group = gui.add_button_group(f"{label} translate", ("−", "+"))
                group.on_click(self._on_jog_translate(axis))
            for axis, label in enumerate(("Roll", "Pitch", "Yaw")):
                group = gui.add_button_group(f"{label} rotate", ("−", "+"))
                group.on_click(self._on_jog_rotate(axis))
            gui.add_button("Reset to ready pose").on_click(
                lambda _: self.pending.push(self._when_idle(self._go_rest))
            )

        with gui.add_folder("Grippers"):
            self.ui["grip_left"] = gui.add_slider(
                "Left", min=0.0, max=1.0, step=0.01, initial_value=GRIP_OPEN
            )
            self.ui["grip_right"] = gui.add_slider(
                "Right", min=0.0, max=1.0, step=0.01, initial_value=GRIP_OPEN
            )
            for arm, key in ((Arm.LEFT, "grip_left"), (Arm.RIGHT, "grip_right")):
                self.ui[key].on_update(self._on_grip(arm))

        with gui.add_folder("Waypoints"):
            self.ui["list"] = gui.add_markdown("_No waypoints yet._")
            self.ui["select"] = gui.add_dropdown(
                "Selected", options=["—"], initial_value="—"
            )
            self.ui["select"].on_update(self._guard(self._on_select))
            gui.add_button("➕ Add waypoint here").on_click(
                lambda _: self.pending.push(self._when_idle(self._add_waypoint))
            )
            gui.add_button("⟳ Update to current pose").on_click(
                lambda _: self.pending.push(self._when_idle(self._update_waypoint))
            )
            gui.add_button("🡒 Go to selected").on_click(
                lambda _: self.pending.push(self._when_idle(self._goto_waypoint))
            )
            move_row = gui.add_button_group("Reorder", ("▲ Up", "▼ Down"))
            move_row.on_click(self._on_reorder)
            edit_row = gui.add_button_group("Edit", ("Duplicate", "Delete"))
            edit_row.on_click(self._on_edit)
            self.ui["move_type"] = gui.add_dropdown(
                "Move type",
                options=[m.label for m in MoveType],
                initial_value=MoveType.LINEAR.label,
            )
            self.ui["move_type"].on_update(
                lambda _: self.pending.push(self._set_move_type)
            )
            self.ui["label"] = gui.add_text("Label", initial_value="")
            self.ui["label"].on_update(self._guard(self._set_label))
            gui.add_button("🗑 Clear all").on_click(
                lambda _: self.pending.push(self._when_idle(self._clear_program))
            )

        with gui.add_folder("Program"):
            self.ui["speed"] = gui.add_slider(
                "Linear speed (m/s)",
                min=0.01,
                max=0.5,
                step=0.01,
                initial_value=self.program.speed,
            )
            self.ui["speed"].on_update(lambda _: self.pending.push(self._sync_settings))
            self.ui["dwell"] = gui.add_slider(
                "Dwell (s)",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=self.program.dwell,
            )
            self.ui["dwell"].on_update(lambda _: self.pending.push(self._sync_settings))
            self.ui["loops"] = gui.add_number(
                "Loops (0 = forever)", initial_value=1, step=1
            )
            self.ui["play"] = gui.add_button("▶ Play program")
            self.ui["play"].on_click(lambda _: self.pending.push(self._play))
            self.ui["stopbtn"] = gui.add_button("■ Stop")
            self.ui["stopbtn"].on_click(lambda _: self.stop_playback.set())
            gui.add_button("💾 Save").on_click(lambda _: self.pending.push(self._save))
            gui.add_button("📂 Reload from file").on_click(
                lambda _: self.pending.push(self._reload)
            )

        self._sync_pose_fields()

    def _when_idle(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Wrap an action so it is dropped while a trajectory is streaming.

        Playback owns the arm. A jog or a waypoint edit landing mid-stream
        writes ``self.q`` under the executor, so the app's belief about the arm
        diverges from what is actually being commanded — and on the failure
        path that divergence is what the operator's next jog is seeded from.
        """

        def action() -> None:
            if self._playing:
                self._set_status("Playing — press Stop before jogging or editing.")
                return
            fn()

        return action

    def _guard(self, fn: Callable[[], None]) -> Callable[[Any], None]:
        """Wrap a GUI callback so programmatic ``.value`` writes do not feed back.

        Repainting a field fires its own ``on_update``, which would otherwise
        re-enter as an operator edit. The flag tested here is only a cheap
        first filter and **cannot** be relied on: viser runs plain (non-async)
        callbacks in a thread pool, so a handler can arrive well after the
        repaint that caused it has finished and cleared the flag. Measured, six
        of six repaints slipped through.

        What actually makes repaints safe is that every action reached this way
        is idempotent — each compares against current state and returns early
        when nothing changed. See :meth:`_apply_pose_fields`.
        """

        def handler(_event: Any) -> None:
            if self._suppress:
                return
            self.pending.push(fn)

        return handler

    def _pose_fields(self, pose: Pose) -> tuple[float, ...]:
        """The exact numbers the pose fields show for ``pose``.

        Rounded the same way :meth:`_sync_pose_fields` writes them, so a
        repaint echo compares equal and an operator edit does not.
        """
        x, y, z = (round(float(v), 4) for v in pose.position)
        roll, pitch, yaw = (round(float(np.degrees(v)), 2) for v in pose.rpy)
        return (x, y, z, roll, pitch, yaw)

    def _current_fields(self) -> tuple[float, ...]:
        """What the pose fields currently hold."""
        return tuple(
            float(self.ui[key].value) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        )

    # -- gui helpers -----------------------------------------------------

    @property
    def arm(self) -> Arm:
        """The arm the pose fields and jog buttons act on."""
        label = self.ui["arm"].value
        return Arm.LEFT if label == Arm.LEFT.label else Arm.RIGHT

    @property
    def _arms_in_play(self) -> tuple[Arm, ...]:
        """Arms a jog applies to — one, or both when the operator asked."""
        return ARMS if self.ui["sync_both"].value else (self.arm,)

    def _set_status(self, text: str) -> None:
        self._status = text
        self.ui["status"].value = text

    def _sync_pose_fields(self) -> None:
        """Repaint the numeric pose fields from the current target."""
        values = self._pose_fields(self.targets[self.arm])
        self._suppress = True
        try:
            for key, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), values):
                self.ui[key].value = value
        finally:
            self._suppress = False
        # Remember exactly what was written. Anything that still matches this
        # later is an echo of this repaint, not an operator edit.
        self._fields_written = values

    def _apply_pose_fields(self) -> None:
        """Adopt a pose typed into the numeric fields.

        Compared against the values *this app last wrote*, not against the
        current target. The difference matters: viser delivers callbacks from a
        thread pool, so a repaint's ``on_update`` can arrive after a drag or a
        jog button has already moved the target on. Comparing against the
        target would make that late echo look like a fresh operator edit and
        quietly put the arm back where it was — a drag that undid itself.
        Comparing against what was written identifies the echo exactly.
        """
        if self._playing:
            return
        if self._current_fields() == self._fields_written:
            return
        self.targets[self.arm] = Pose.from_rpy(
            (self.ui["x"].value, self.ui["y"].value, self.ui["z"].value),
            np.radians(self.ui["roll"].value),
            np.radians(self.ui["pitch"].value),
            np.radians(self.ui["yaw"].value),
        )
        self._target_dirty = True

    def _on_jog_translate(self, axis: int) -> Callable[[Any], None]:
        def handler(event: Any) -> None:
            sign = 1.0 if str(event.target.value).endswith("+") else -1.0
            step = float(self.ui["step_mm"].value) * 1e-3 * sign
            delta = np.zeros(3, dtype=np.float32)
            delta[axis] = step
            tool_frame = self.ui["frame_mode"].value == "Tool"

            def apply() -> None:
                for arm in self._arms_in_play:
                    pose = self.targets[arm]
                    self.targets[arm] = (
                        pose.translated_local(delta)
                        if tool_frame
                        else pose.translated(delta)
                    )
                self._target_dirty = True
                axis_name = "XYZ"[axis]
                self._set_status(
                    f"Jog {axis_name} {step * 1e3:+.1f} mm "
                    f"({'tool' if tool_frame else 'world'} frame)"
                )

            self.pending.push(self._when_idle(apply))

        return handler

    def _on_jog_rotate(self, axis: int) -> Callable[[Any], None]:
        def handler(event: Any) -> None:
            sign = 1.0 if str(event.target.value).endswith("+") else -1.0
            step = float(np.radians(self.ui["step_deg"].value)) * sign
            rpy = [0.0, 0.0, 0.0]
            rpy[axis] = step
            tool_frame = self.ui["frame_mode"].value == "Tool"

            def apply() -> None:
                for arm in self._arms_in_play:
                    pose = self.targets[arm]
                    self.targets[arm] = (
                        pose.rotated_local(*rpy) if tool_frame else pose.rotated(*rpy)
                    )
                self._target_dirty = True
                axis_name = ("Roll", "Pitch", "Yaw")[axis]
                self._set_status(
                    f"Jog {axis_name} {np.degrees(step):+.1f}\u00b0 "
                    f"({'tool' if tool_frame else 'world'} frame)"
                )

            self.pending.push(self._when_idle(apply))

        return handler

    def _on_grip(self, arm: Arm) -> Callable[[Any], None]:
        def handler(event: Any) -> None:
            if self._suppress:
                return
            value = float(event.target.value)

            def apply() -> None:
                self.grips[arm] = value
                self._push_to_robot()

            self.pending.push(self._when_idle(apply))

        return handler

    def _go_rest(self) -> None:
        # The *ready* pose, not the park pose: rest is 88% extended and jogging
        # out of it immediately meets the reach clamp.
        # Set the *target* only. Syncing ``_commanded`` here would skip the
        # rate limiter and offer the robot a single step of over 1.5 rad
        # against its 0.5 rad limit — which it drops silently, leaving the app
        # believing the arm moved and every later jog seeded from a phantom
        # pose. ``_goto_waypoint`` relies on the same ramp.
        self.q = self.kin.ready_q()
        self.targets.update(self.kin.fk(self.q, self.tool))
        self._sync_gizmos()
        self._sync_pose_fields()
        self._push_to_robot()
        self._set_status("Returned to the ready pose.")

    # -- IK loop ---------------------------------------------------------

    def _solve_targets(self) -> None:
        """Resolve the gizmo targets to joints and command them."""
        solution = self.kin.solve(self.targets, seed=self.q, tool=self.tool)
        self.q = solution.q
        actual = self.kin.fk(self.q, self.tool)
        if solution.reached:
            note = "on target"
        else:
            note = (
                f"{solution.worst_position_error * 1e3:.1f} mm off — "
                f"{solution.message.split(':')[0].lower()}"
            )
            # Snap back to what the arm can actually do, so the operator sees
            # the reachable pose rather than a marker floating somewhere the
            # arm never went.
            # Update in place rather than rebinding: viser threads write into
            # this dict from drag callbacks, and a rebind silently discards any
            # write that landed on the old object.
            self.targets.update(actual)
        # Repaint on *every* solve, not only on failure. A jog button changes
        # the target directly, so without this the arm moves while the marker
        # and the numeric fields sit where they were.
        self._sync_gizmos()
        self._sync_pose_fields()
        margins = {
            arm: self.kin.singularity_margin(self.q, arm, self.tool) for arm in ARMS
        }
        if min(margins.values()) < SINGULARITY_THRESHOLD:
            note += " · near singularity"
        if solution.collides:
            note += f" · ⚠ {abs(solution.clearance) * 1e3:.0f} mm inside the torso"
        self.ui["ik"].value = note
        self._push_to_robot()

    def _advance_commanded(self, dt: float) -> bool:
        """Step the commanded configuration toward the target. True if it moved.

        A pure rate limit, applied per joint. Small jogs arrive in one tick; a
        large one is walked out over as many as it takes, so neither the
        operator nor the robot ever sees the setpoint teleport.
        """
        gap = self.q - self._commanded
        worst = float(np.max(np.abs(gap)))
        if worst < 1e-9:
            return False
        step = JOG_JOINT_SPEED * dt
        self._commanded = (
            self.q.copy() if worst <= step else self._commanded + gap * (step / worst)
        ).astype(np.float32)
        return True

    def _push_to_robot(self) -> None:
        """Queue the commanded configuration for the robot (and so the view)."""
        left, right = self.commander.arm_command(
            self._commanded, (self.grips[Arm.LEFT], self.grips[Arm.RIGHT])
        )
        # The jog loop writes to the robot directly, so the commander would
        # otherwise still believe the last playback's final pose is current and
        # ease in from there — a teleport on the first tick of the next Play.
        self.commander.adopt_commanded(self._commanded)
        self._to_send = (left, right)

    # -- waypoint operations ---------------------------------------------

    def _current_waypoint(self) -> Waypoint:
        """A waypoint capturing exactly where the robot is right now."""
        return Waypoint(
            targets={
                arm: ArmTarget(
                    pose=self.kin.fk_arm(self.q, arm, self.tool),
                    joints=self.kin.arm_q(self.q, arm),
                    grip=self.grips[arm],
                )
                for arm in ARMS
            },
            move=MoveType.LINEAR,
        )

    def _add_waypoint(self) -> None:
        index = self.program.insert(self.selected + 1, self._current_waypoint())
        self.selected = index
        self._after_program_change(f"Added waypoint {index + 1}.")

    def _update_waypoint(self) -> None:
        if not self._has_selection():
            return
        keep = self.program[self.selected]
        fresh = self._current_waypoint()
        fresh.move, fresh.label = keep.move, keep.label
        fresh.speed, fresh.dwell = keep.speed, keep.dwell
        self.program.replace(self.selected, fresh)
        self._after_program_change(f"Updated waypoint {self.selected + 1}.")

    def _goto_waypoint(self) -> None:
        if not self._has_selection():
            return
        waypoint = self.program[self.selected]
        q = waypoint.full_q(self.kin)
        if q is None:
            failed = self.program.resolve(self.kin, seed=self.q)
            if self.selected in failed:
                self._set_status(f"Waypoint {self.selected + 1} cannot be reached.")
                return
            q = self.program[self.selected].full_q(self.kin)
        self.q = q
        for arm in ARMS:
            self.grips[arm] = waypoint.grip(arm)
        self.targets.update(self.kin.fk(self.q, self.tool))
        self._sync_gizmos()
        self._sync_pose_fields()
        # Repaint the grip sliders too: _goto_waypoint adopts the waypoint's
        # gripper state, and a slider still showing the old value snaps the
        # gripper back the moment it is touched.
        self._suppress = True
        try:
            self.ui["grip_left"].value = self.grips[Arm.LEFT]
            self.ui["grip_right"].value = self.grips[Arm.RIGHT]
        finally:
            self._suppress = False
        self._push_to_robot()
        self._set_status(f"Moved to waypoint {self.selected + 1}.")

    def _on_reorder(self, event: Any) -> None:
        up = "Up" in str(event.target.value)

        def apply() -> None:
            if not self._has_selection():
                return
            self.selected = (
                self.program.move_up(self.selected)
                if up
                else self.program.move_down(self.selected)
            )
            self._after_program_change(
                f"Moved waypoint {'up' if up else 'down'} to position {self.selected + 1}."
            )

        self.pending.push(self._when_idle(apply))

    def _on_edit(self, event: Any) -> None:
        duplicate = "Duplicate" in str(event.target.value)

        def apply() -> None:
            if not self._has_selection():
                return
            if duplicate:
                self.selected = self.program.duplicate(self.selected)
                self._after_program_change(
                    f"Duplicated to waypoint {self.selected + 1}."
                )
            else:
                removed = self.selected
                self.program.remove(self.selected)
                self.selected = min(self.selected, len(self.program) - 1)
                self._after_program_change(f"Deleted waypoint {removed + 1}.")

        self.pending.push(self._when_idle(apply))

    def _set_move_type(self) -> None:
        if not self._has_selection():
            return
        chosen = next(m for m in MoveType if m.label == self.ui["move_type"].value)
        if self.program[self.selected].move is chosen:
            return
        self.program[self.selected].move = chosen
        self._after_program_change(
            f"Waypoint {self.selected + 1} is now {chosen.label}."
        )

    def _set_label(self) -> None:
        if not self._has_selection():
            return
        label = str(self.ui["label"].value)
        if self.program[self.selected].label == label:
            return
        self.program[self.selected].label = label
        self._after_program_change("Renamed.")

    def _on_select(self) -> None:
        value = str(self.ui["select"].value)
        if not value.startswith("#"):
            return
        index = int(value[1:].split(" ")[0]) - 1
        if index == self.selected:
            return
        self.selected = index
        self._sync_selection_fields()

    def _clear_program(self) -> None:
        self.program.clear()
        self.selected = -1
        self._after_program_change("Cleared the program.")

    def _has_selection(self) -> bool:
        if 0 <= self.selected < len(self.program):
            return True
        self._set_status("Select a waypoint first.")
        return False

    def _sync_selection_fields(self) -> None:
        if not (0 <= self.selected < len(self.program)):
            return
        waypoint = self.program[self.selected]
        self._suppress = True
        try:
            self.ui["move_type"].value = waypoint.move.label
            self.ui["label"].value = waypoint.label
            self.ui["select"].value = self._option_for(self.selected)
        finally:
            self._suppress = False

    def _option_for(self, index: int) -> str:
        waypoint = self.program[index]
        name = waypoint.label or f"waypoint {index + 1}"
        return f"#{index + 1} {name}"

    def _after_program_change(self, message: str) -> None:
        self.program.save(self.cfg.file)
        self._refresh_waypoints()
        self._set_status(message)

    def _refresh_waypoints(self) -> None:
        """Repaint the waypoint list, dropdown, and scene markers."""
        for node in self._waypoint_nodes:
            node.remove()
        self._waypoint_nodes = []

        if not len(self.program):
            self._suppress = True
            try:
                self.ui["list"].content = "_No waypoints yet._"
                self.ui["select"].options = ["—"]
                self.ui["select"].value = "—"
            finally:
                self._suppress = False
            return

        rows = ["| # | name | move | L grip | R grip |", "|---|---|---|---|---|"]
        for i, waypoint in enumerate(self.program):
            marker = "**▶**" if i == self.selected else ""
            rows.append(
                f"| {marker} {i + 1} | {waypoint.label or '—'} | "
                f"{'L' if waypoint.move is MoveType.LINEAR else 'J'} | "
                f"{waypoint.grip(Arm.LEFT):.2f} | {waypoint.grip(Arm.RIGHT):.2f} |"
            )
            for arm in ARMS:
                pose = self.program.world_pose(i, arm)
                self._waypoint_nodes.append(
                    self.server.scene.add_frame(
                        f"/wp_{i}_{arm.value}",
                        axes_length=0.05,
                        axes_radius=0.0025,
                        origin_radius=0.008,
                        position=tuple(float(v) for v in pose.position),
                        wxyz=tuple(float(v) for v in pose.quat),
                    )
                )
            self._waypoint_nodes.append(
                self.server.scene.add_label(
                    f"/wp_{i}_label",
                    text=str(i + 1),
                    position=tuple(
                        float(v)
                        for v in self.program.world_pose(i, Arm.LEFT).position
                        + np.array([0.0, 0.0, 0.04], dtype=np.float32)
                    ),
                )
            )
        self.ui["list"].content = "\n".join(rows)
        options = [self._option_for(i) for i in range(len(self.program))]
        self._suppress = True
        try:
            self.ui["select"].options = options
            if 0 <= self.selected < len(options):
                self.ui["select"].value = options[self.selected]
        finally:
            self._suppress = False
        self._sync_selection_fields()

    def _sync_settings(self) -> None:
        self.program.speed = float(self.ui["speed"].value)
        self.program.dwell = float(self.ui["dwell"].value)
        self.program.save(self.cfg.file)

    def _save(self) -> None:
        self.program.save(self.cfg.file)
        self._set_status(f"Saved {len(self.program)} waypoints to {self.cfg.file}")

    def _reload(self) -> None:
        self.program = Program.load(self.cfg.file, self.kin)
        self.program.tool = self.tool
        self.selected = 0 if len(self.program) else -1
        self._refresh_waypoints()
        self._set_status(f"Loaded {len(self.program)} waypoints.")

    # -- playback --------------------------------------------------------

    def _play(self) -> None:
        if self._playing:
            self._set_status("Already playing.")
            return
        if len(self.program) < 1:
            self._set_status("Record a waypoint before playing.")
            return
        # Clear here, on the operator's Play press — not inside the playback
        # task. Clearing there would discard a Stop requested in the window
        # between pressing Play and the task starting.
        self.stop_playback.clear()
        self._playback_request = int(self.ui["loops"].value)

    async def _run_playback(self, loops: int) -> None:
        """Plan the whole program, then stream it."""
        self._playing = True
        try:
            self._set_status(f"Planning {len(self.program)} waypoints…")
            plan = await asyncio.to_thread(
                self.commander.plan_program,
                self.program,
                self.q,
                start_grips=(self.grips[Arm.LEFT], self.grips[Arm.RIGHT]),
                loops=loops,
            )
            self._set_status(f"Playing {len(plan.legs)} legs, {plan.duration:.1f}s")

            def progress(index: int, leg: Any, fraction: float) -> None:
                self._set_status(
                    f"[{index + 1}/{len(plan.legs)}] {leg.label} — {fraction * 100:.0f}%"
                )

            report = await self.commander.execute(
                plan, stop_event=self.stop_playback, on_progress=progress
            )
            while loops == 0 and not self.stop_playback.is_set() and not report.stopped:
                # "0 = forever". plan_program closed the cycle, so replaying the
                # same legs continues seamlessly from where the last pass ended.
                report = await self.commander.execute(
                    plan, stop_event=self.stop_playback, on_progress=progress
                )
            self._set_status(report.message)
        except (PathPlanningError, UnreachableError) as exc:
            self._set_status(f"Cannot play: {exc}")
            _logger.warning("playback planning failed: %s", exc)
        except Exception as exc:
            self._set_status(f"Playback error: {exc}")
            _logger.exception("playback failed")
        finally:
            # Re-adopt on *every* path. Skipping this after a failure leaves the
            # app describing a pose the arm is not at, so the operator's next
            # small jog becomes a large step the robot drops.
            try:
                await self._adopt_robot_state()
            except Exception:
                _logger.exception("could not re-read the robot after playback")
            self._playing = False

    async def park(self) -> None:
        """Drive back to the rest pose before the motors are released.

        Leaves the robot somewhere known rather than wherever the operator
        stopped jogging. Planned as a ``MoveJ`` so it keeps clear of the torso,
        and not interruptible — abandoning it halfway is worse than finishing.
        """
        try:
            q_now, grips = await self.commander.current_state()
            trajectory = await asyncio.to_thread(
                self.commander.plan_joint,
                q_now,
                self.kin.rest_q(),
                speed=VRTeleopConfig().reset_speed,
                min_duration=VRTeleopConfig().reset_min_duration,
            )
            leg = PlannedLeg(
                label="return to rest",
                move=MoveType.JOINT,
                q=list(trajectory),
                grips=[grips] * len(trajectory),
            )
            self._set_status("Returning to the rest pose…")
            await self.commander.execute(leg)
        except Exception:
            _logger.warning("return-to-rest during shutdown failed", exc_info=True)

    async def _adopt_robot_state(self) -> None:
        """Reconcile every belief about the arm with what the robot reports.

        The app carries three: the IK target ``q``, the rate-limited
        ``_commanded``, and the commander's ``_last_commanded``. Each of the
        state-divergence bugs found in review was one of them being written
        without the others, and each ended the same way — a command over
        ``max_step_rad`` that the robot drops without telling anyone. This is
        the only place allowed to resynchronise them.
        """
        self.q, grips = await self.commander.current_state()
        self._commanded = self.q.copy()
        self.commander.adopt_commanded(self._commanded)
        self.grips = {Arm.LEFT: grips[0], Arm.RIGHT: grips[1]}
        self.targets.update(self.kin.fk(self.q, self.tool))
        self._sync_gizmos()
        self._sync_pose_fields()
        self._suppress = True
        try:
            self.ui["grip_left"].value = grips[0]
            self.ui["grip_right"].value = grips[1]
        finally:
            self._suppress = False

    # -- main loop -------------------------------------------------------

    async def run(self) -> None:
        """Drive the app until the operator quits."""
        self._to_send: tuple[np.ndarray, np.ndarray] | None = None
        self._playback_request: int | None = None
        dt = 1.0 / UI_RATE_HZ

        # Adopt whatever the robot reports rather than imposing a pose on it.
        # On hardware that is wherever the arms actually are, which is the only
        # safe thing to assume; in simulation it is the seeded ready pose.
        await self._adopt_robot_state()

        self._set_status(
            f"Ready — {len(self.program)} waypoints. Drag a gripper, or jog with the buttons."
        )
        try:
            await self._loop(dt)
        finally:
            # Never leave a playback task streaming: the caller's teardown runs
            # park() next, and two writers on one arm interleave at the control
            # rate with neither winning.
            self.stop_playback.set()
            task = self._playback_task
            if task is not None and not task.done():
                with contextlib.suppress(Exception):
                    await task

    async def _loop(self, dt: float) -> None:
        """The 30 Hz body: drain callbacks, solve, rate-limit, command, render."""
        while not self.quit.is_set():
            loop_start = time.monotonic()

            for fn in self.pending.drain():
                try:
                    fn()
                except Exception as exc:
                    _logger.exception("UI action failed")
                    self._set_status(f"Error: {exc}")

            self._expire_stale_drags()

            if self._target_dirty and not self._playing:
                self._target_dirty = False
                try:
                    self._solve_targets()
                except Exception as exc:
                    _logger.exception("IK failed")
                    self._set_status(f"IK error: {exc}")

            if self._playback_request is not None and not self._playing:
                loops = self._playback_request
                self._playback_request = None
                self._playback_task = asyncio.create_task(self._run_playback(loops))

            # Robot I/O is guarded: a CAN timeout is not exotic, and letting
            # one out of this loop kills the pendant while a playback task may
            # still be streaming — two writers on one arm.
            try:
                if not self._playing:
                    if self._advance_commanded(dt) or self._to_send is not None:
                        self._push_to_robot()
                    if self._to_send is not None:
                        left, right = self._to_send
                        self._to_send = None
                        await self.robot.motion_control(left=left, right=right)

                # Render whatever the robot reports, so the view is the robot's
                # state and not the app's intention — the same on hardware.
                left, right = await self.robot.get_positions()
                if left is not None and right is not None:
                    left = np.asarray(left, dtype=np.float32)
                    right = np.asarray(right, dtype=np.float32)
                    q = self.q.copy()
                    q[self.kin.indices(Arm.LEFT)] = left[:7]
                    q[self.kin.indices(Arm.RIGHT)] = right[:7]
                    self._render(q, (float(left[7]), float(right[7])))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._io_failures += 1
                _logger.exception("robot I/O failed (%d)", self._io_failures)
                self._set_status(f"Robot error: {exc}")
                if self._io_failures >= MAX_IO_FAILURES:
                    self._set_status(
                        f"Robot unreachable after {self._io_failures} attempts — stopping."
                    )
                    break
            else:
                self._io_failures = 0

            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------


def main(argv: list[str]) -> None:
    """Parse the CLI config and run the jog pendant."""
    cfg = parse(JogCmdConfig, normalize_bool_flags(argv, "sim"))
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    try:
        asyncio.run(_session(cfg))
    except KeyboardInterrupt:
        print("\nExiting jog ...")
    except ValueError as exc:
        _logger.error("%s", exc)
        sys.exit(1)


async def _session(cfg: JogCmdConfig) -> None:
    """Build the kinematics, the robot, and the app, then run."""
    print("Loading kinematics (JIT compile takes a few seconds) ...", flush=True)
    tool = default_tool(has_gripper=cfg.axol.has_gripper)
    kin = await asyncio.to_thread(CartesianKinematics, cfg.kinematics, tool)

    robot: RobotBase
    if cfg.sim:
        # Seed the simulator at the ready pose rather than the park pose, so a
        # session opens somewhere an operator can actually jog from.
        q_ready = kin.ready_q()
        robot = VirtualRobot(
            np.append(kin.arm_q(q_ready, Arm.LEFT), 1.0),
            np.append(kin.arm_q(q_ready, Arm.RIGHT), 1.0),
        )
    else:
        from ..constants import CAN_LEFT, CAN_RIGHT
        from ..robot import Axol

        robot = Axol(
            config=cfg.axol,
            left_channel=cfg.left_channel or CAN_LEFT,
            right_channel=cfg.right_channel or CAN_RIGHT,
        )

    async with robot:
        if not cfg.sim:
            await robot.start_telemetry(cfg.telemetry_hz)
            await robot.wait_for_telemetry()
            # The arms may have been hand-guided since the last session, so the
            # cached command history no longer describes where they are. Without
            # this the first command reads as a jump and the robot's max-step
            # check silently drops it.
            robot.reset_command_state()
        app = JogApp(cfg, kin, robot)
        print(f"\n  Cartesian jog UI:  http://localhost:{cfg.port}\n", flush=True)
        try:
            await app.run()
        finally:
            if not cfg.sim:
                await app.park()
            app.server.stop()
