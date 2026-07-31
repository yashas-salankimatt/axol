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

from ..constants import ARM_JOINTS, visual_urdf_path
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
"""How often the app solves IK and repaints.

Deliberately *not* the rate the robot is commanded at — see
:meth:`JogApp._control_loop`. IK costs about 2 ms, so this is far from the
limiting factor; it is chosen to feel immediate while dragging."""

DRAG_IDLE_TICKS = 45
"""App-loop ticks (1.5 s at 30 Hz) a drag may go without a gizmo update before
its flag is treated as stranded. See :meth:`JogApp._expire_stale_drags`."""

_SIDE: dict[Arm, str] = {Arm.LEFT: "L", Arm.RIGHT: "R"}
"""Short side labels. The tracking panel is narrow enough that spelling out
"left"/"right" wraps every row onto two lines."""

TRACKING_FAULT_RAD = 0.26
"""Tracking error (rad, ~15°) at which jogging halts and the setpoint re-adopts.

A joint that has stopped following its setpoint is not a tracking problem, it
is a dead joint — a motor that hit a protection limit and disabled its output
latches the fault and stops producing torque, and the SDK's telemetry callback
throws the status byte away, so nothing else notices.

What makes it dangerous is *windup*. The impedance command is
``kp * (target - actual)``; with the left shoulder_2 gain at 158 and the arm
sagging 36.5° away, that is a demand for roughly 100 Nm standing by. The
instant the motor clears its fault or is re-enabled, it slams to the setpoint.

So on a large error the setpoint is re-adopted to where the arm actually is.
Every healthy joint keeps its holding torque and gravity compensation — the
arm does not go limp — but nothing is left wound up against a dead one."""

TRACKING_REPORT_HZ = 8.0
"""How often the commanded-vs-actual readout is refreshed. Fast enough to watch
an oscillation build, slow enough not to flood the websocket."""

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
    park_on_exit: bool = False
    """Drive back to the rest pose when the session ends (hardware only).

    Off by default. A pendant should leave the arm where the operator left it,
    so that closing and reopening a session resumes from the same pose instead
    of driving home and back. ``axol waypoints`` parks because it ends a
    program *run*; a jog session ends mid-task.

    It is also the safer default: the return is planned clear of the robot's
    own body, and nothing in this SDK models the world around it, so a MoveJ
    home from inside a fixture is not something to trigger on a keystroke."""
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
        self._track_last = 0.0
        self._tracking_fault = False
        self._needs_readopt = False
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

        with gui.add_folder("Tracking (commanded vs actual)"):
            # Markdown, not add_text: a disabled text box renders as a narrow
            # labelled input that truncates mid-value — which is how the whole
            # Cartesian readout ended up invisible behind "(dx -149...".
            self.ui["track_worst"] = gui.add_markdown("_waiting for telemetry_")
            self.ui["track_cartesian"] = gui.add_markdown("")
            self.ui["track_table"] = gui.add_markdown("")

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
            gui.add_button("⚠ Clear fault and resume").on_click(
                lambda _: self.pending.push(self._clear_tracking_fault)
            )
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
            if self._tracking_fault:
                self._set_status(
                    "A joint stopped following its setpoint — check the motor, "
                    "then press 'Clear fault' to resume."
                )
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

    def _clear_tracking_fault(self) -> None:
        """Resume jogging after a halted joint has been dealt with.

        Re-reads the arm first, so resuming cannot itself command a jump from a
        setpoint recorded before the joint gave way.
        """
        if not self._tracking_fault:
            self._set_status("No fault to clear.")
            return
        self._tracking_fault = False
        self._set_status("Fault cleared — re-reading the arm before resuming.")
        self._needs_readopt = True

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

    def _joint_label(self, index: int) -> str:
        """Operator-facing name for a joint index, e.g. ``L shoulder_2``.

        The solver names joints as the URDF does (``left_s2_0``), which is not
        what is written on the robot or in the gain tables an operator will be
        editing. Abbreviated to ``L``/``R`` because the panel is narrow and
        "left shoulder_1" wraps onto two lines in every row.
        """
        for arm in ARMS:
            positions = self.kin.indices(arm)
            if index in positions:
                return f"{_SIDE[arm]} {ARM_JOINTS[positions.index(index)].value}"
        return str(index)

    def _check_tracking_fault(self, q_actual: np.ndarray) -> bool:
        """Halt and re-adopt if a joint has stopped following its setpoint.

        Returns True if a fault was just latched. See :data:`TRACKING_FAULT_RAD`
        for why winding a setpoint up against a dead joint is the thing to
        avoid.
        """
        error = np.asarray(q_actual, dtype=np.float32) - self._commanded
        worst = int(np.argmax(np.abs(error)))
        if abs(float(error[worst])) < TRACKING_FAULT_RAD or self._playing:
            return False
        if self._tracking_fault:
            return False

        self._tracking_fault = True
        # Follow the arm rather than fight it: zero the error so no joint is
        # left demanding torque it cannot deliver.
        self.q = np.asarray(q_actual, dtype=np.float32).copy()
        self._commanded = self.q.copy()
        self.commander.adopt_commanded(self._commanded)
        self.targets.update(self.kin.fk(self.q, self.tool))
        self._sync_gizmos()
        self._sync_pose_fields()
        message = (
            f"⚠ {self._joint_label(worst)} stopped following its setpoint "
            f"({np.degrees(error[worst]):+.1f}°). Jogging halted and the setpoint "
            "re-adopted so nothing is wound up against it. Check the motor: stop "
            "this session and run `axol motor.health`."
        )
        self._set_status(message)
        _logger.error("%s", message)
        return True

    def _report_tracking(self, q_actual: np.ndarray) -> None:
        """Publish how far the arm is lagging what it was told to do.

        The gap between ``_commanded`` and telemetry is the impedance
        controller's tracking error: with soft gains the arm sags under its own
        weight and trails a moving setpoint, and both show up here. Reported in
        joint space (where the gains live) and at the tool (where it matters).

        Read-only. Nothing in the control path uses these numbers.
        """
        now = time.monotonic()
        if now - self._track_last < 1.0 / TRACKING_REPORT_HZ:
            return
        self._track_last = now

        commanded = self._commanded
        error = np.asarray(q_actual, dtype=np.float32) - commanded

        worst = int(np.argmax(np.abs(error)))
        self.ui["track_worst"].content = (
            f"**worst joint** &nbsp; `{self._joint_label(worst)}` &nbsp; "
            f"**{np.degrees(error[worst]):+.2f}°**"
        )

        # Cartesian deviation: forward kinematics of the commanded joint vector
        # against forward kinematics of the measured one, at the tool. This is
        # what the joint error actually costs at the fingertips, which is not
        # something the per-joint numbers show — a degree at the shoulder and a
        # degree at the wrist are very different distances out there.
        cartesian = [
            "| arm | dx | dy | dz | dist | angle |",
            "|:--|--:|--:|--:|--:|--:|",
        ]
        joints = ["| joint | cmd | act | err |", "|:--|--:|--:|--:|"]
        for arm in ARMS:
            want = self.kin.fk_arm(commanded, arm, self.tool)
            have = self.kin.fk_arm(q_actual, arm, self.tool)
            delta = (have.position - want.position) * 1e3
            cartesian.append(
                f"| **{_SIDE[arm]}** | {delta[0]:+.2f} | {delta[1]:+.2f} | {delta[2]:+.2f} "
                f"| **{have.distance_to(want) * 1e3:.2f}** | **{np.degrees(have.angle_to(want)):.2f}** |"
            )
            for index in self.kin.indices(arm):
                joints.append(
                    f"| {self._joint_label(index)} | {np.degrees(commanded[index]):+.2f} "
                    f"| {np.degrees(q_actual[index]):+.2f} "
                    f"| **{np.degrees(error[index]):+.2f}** |"
                )
        self.ui["track_cartesian"].content = (
            "**tool deviation** (mm, deg)\n\n" + "\n".join(cartesian)
        )
        self.ui["track_table"].content = "**joint deviation** (deg)\n\n" + "\n".join(
            joints
        )

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
        control = asyncio.create_task(self._control_loop())
        try:
            await self._loop(dt)
        finally:
            self.quit.set()
            control.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await control
            # Never leave a playback task streaming: the caller's teardown runs
            # park() next, and two writers on one arm interleave at the control
            # rate with neither winning.
            self.stop_playback.set()
            task = self._playback_task
            if task is not None and not task.done():
                with contextlib.suppress(Exception):
                    await task

    async def _control_loop(self) -> None:
        """Stream the commanded configuration at the robot's control rate.

        Separate from the UI loop, and running far faster than it, because
        ``Axol.motion_control`` differentiates the commanded positions to get
        the velocity and acceleration feedforward terms it sends with every
        impedance command. Those differentiators are tuned for a 250 Hz stream
        with a 20 Hz cutoff; driving them from a 30 Hz UI loop samples them
        above half-Nyquist and turns each tick into a torque impulse.

        Measured at ``JOG_JOINT_SPEED``: 250 Hz steps the setpoint 2.4 mrad per
        tick for a 0.044 rad/s velocity feedforward step, while 30 Hz steps it
        20 mrad for 0.240 rad/s — five and a half times the kick, arriving at a
        frequency a compliant arm resonates at. That is a jog that visibly
        wobbles even with the gains set correctly.

        It also commands *continuously* rather than only when something moved.
        An irregular interval makes the differentiator's own ``Ts`` jump around,
        and a steady stream is what lets the velocity estimate settle to zero
        when the operator stops jogging.
        """
        dt = 1.0 / max(1.0, self.cfg.rate_hz)
        while not self.quit.is_set():
            loop_start = time.monotonic()
            try:
                if not self._playing:
                    self._advance_commanded(dt)
                    left, right = self.commander.arm_command(
                        self._commanded, (self.grips[Arm.LEFT], self.grips[Arm.RIGHT])
                    )
                    self.commander.adopt_commanded(self._commanded)
                    await self.robot.motion_control(left=left, right=right)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._io_failures += 1
                _logger.exception("control loop write failed (%d)", self._io_failures)
                self._set_status(f"Robot error: {exc}")
                if self._io_failures >= MAX_IO_FAILURES:
                    self._set_status("Robot unreachable — stopping.")
                    self.quit.set()
                    return
            else:
                self._io_failures = 0
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)

    async def _loop(self, dt: float) -> None:
        """The UI body: drain callbacks, solve IK, repaint, report tracking."""
        while not self.quit.is_set():
            loop_start = time.monotonic()

            for fn in self.pending.drain():
                try:
                    fn()
                except Exception as exc:
                    _logger.exception("UI action failed")
                    self._set_status(f"Error: {exc}")

            if self._needs_readopt:
                self._needs_readopt = False
                with contextlib.suppress(Exception):
                    await self._adopt_robot_state()

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
                # Render whatever the robot reports, so the view is the robot's
                # state and not the app's intention — the same on hardware.
                left, right = await self.commander.read_positions()
                if left is not None and right is not None:
                    left = np.asarray(left, dtype=np.float32)
                    right = np.asarray(right, dtype=np.float32)
                    q_actual = self.q.copy()
                    q_actual[self.kin.indices(Arm.LEFT)] = left[:7]
                    q_actual[self.kin.indices(Arm.RIGHT)] = right[:7]
                    self._render(q_actual, (float(left[7]), float(right[7])))
                    self._check_tracking_fault(q_actual)
                    self._report_tracking(q_actual)
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
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n  Interrupted.", flush=True)
        finally:
            # asyncio.run cancels this task on SIGINT (Python 3.11+), so every
            # await below would re-raise CancelledError immediately and the
            # shutdown would be skipped silently — including the return to
            # rest. Same trap ``axol waypoints`` documents.
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            app.quit.set()
            if not cfg.sim and cfg.park_on_exit:
                print(
                    "\n  Returning to the rest pose — the arms will MOVE.\n"
                    "  Ctrl-C again to abort and leave them where they are.\n",
                    flush=True,
                )
                try:
                    await app.park()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    print("  Park aborted — arms left in place.", flush=True)
            elif not cfg.sim:
                print(
                    "\n  Leaving the arms where they are (--park_on_exit False).\n",
                    flush=True,
                )
            app.server.stop()
