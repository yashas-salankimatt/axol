"""Planning and execution — the MoveGroup layer.

:class:`MotionCommander` is the one object an application talks to: it turns
poses into trajectories and streams trajectories at the control rate, against
any :class:`~almond_axol.robot.base.RobotBase`. The simulator and the real arm
differ by which object you hand the constructor and nothing else, which is what
makes it possible to build a program on a laptop and run it unchanged on
hardware.

Two motion primitives, matching every industrial controller:

- :attr:`~.program.MoveType.JOINT` (``MoveJ``) — joint-space interpolation,
  each sample projected onto the joint-limit and self-collision manifold by
  :func:`~almond_axol.teleop.trajectory.plan_collision_aware_trajectory`. The
  tool sweeps an arc. Cannot stall partway on a wrist singularity, so it is
  what free-air repositioning and the initial approach should use.

  Its name oversells it, and the distinction matters: that projection is a
  *local repair*, not a planner. It nudges each interpolated sample toward
  feasibility from where the interpolation put it, with no mechanism to search
  for a qualitatively different route. Measured over random clear-endpoint
  pairs whose straight interpolation penetrates the torso: a 0.9 mm graze is
  recovered to +7.5 mm clear, while a 37 mm intrusion is only improved to
  2.8 mm — still colliding. So it fixes grazes and cannot go around anything.
  When the straight path is genuinely blocked, the gate below refuses and the
  operator adds a via-point, which is exactly how a teach pendant works.
- :attr:`~.program.MoveType.LINEAR` (``MoveL``) — the tool held to a straight
  world-frame line by :func:`~almond_axol.kinematics.path.plan_linear_segment`,
  which resolves the whole path before anything moves and raises rather than
  stalling an arm mid-move.

Everything is planned before anything is executed. A program that cannot be
flown is reported while the robot is still standing still — the property that
makes it safe to press Play on a path you built in a simulator.

Soft costs, hard gate
---------------------

Every collision term inside the solver is an L2 cost, so it can be outvoted and
is never a guarantee. The guarantee lives outside it: :meth:`MotionCommander.plan_program`
checks **every tick of every leg** against the collision model and refuses the
whole program if any of them penetrates. This mirrors how MoveIt splits the
job — optimising planners use soft costs, and a separate hard validity check
runs before execution.

Two limits worth stating plainly. The check covers **self-collision only**;
there is no model of the world around the robot, so a path through a fixture
or a machine enclosure is invisible to it. And it runs at plan time only —
nothing re-checks while a trajectory is streaming.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from ..kinematics.path import PathPlanningError, ease, plan_linear_segment
from ..robot.base import RobotBase
from ..teleop.config import VRTeleopConfig
from ..teleop.trajectory import plan_collision_aware_trajectory
from .frames import ARMS, Arm, Pose, Tool
from .kinematics import CartesianKinematics
from .program import N_ARM_JOINTS, MoveType, Program

_logger = logging.getLogger(__name__)

DEFAULT_RATE_HZ = 250.0
"""Control rate trajectories are sampled at, matching the rest of the SDK."""

DEFAULT_EASE_IN = 0.4
"""Seconds spent blending onto the start of a planned leg.

A plan begins at the configuration IK settled into, which reaches the same tool
pose as wherever the arm is now but can hold the elbow a degree or so
differently. Commanding it outright puts that whole difference into one tick.
"""

Grips = tuple[float, float]
"""Commanded gripper openings as ``(left, right)``."""


@dataclass
class PlannedLeg:
    """One planned move: joint vectors and gripper openings, one pair per tick.

    Keeping the grippers alongside the joints — rather than as a separate phase
    the executor has to sequence — means :meth:`MotionCommander.execute` is a
    single loop with nothing to get out of order, and a leg can be inspected,
    timed, or replayed on its own.

    Attributes:
        label: Operator-facing description, used in progress callbacks.
        move:  Which primitive produced this leg.
        q:     Full joint vector per tick.
        grips: Commanded ``(left, right)`` gripper opening per tick.
    """

    label: str
    move: MoveType
    q: list[np.ndarray]
    grips: list[Grips]

    def __post_init__(self) -> None:
        if len(self.q) != len(self.grips):
            raise ValueError(
                f"leg {self.label!r} has {len(self.q)} joint samples but "
                f"{len(self.grips)} gripper samples"
            )

    def __len__(self) -> int:
        return len(self.q)

    def duration(self, rate: float = DEFAULT_RATE_HZ) -> float:
        """Wall-clock seconds this leg takes at ``rate``."""
        return len(self.q) / rate


@dataclass
class PlannedProgram:
    """A whole program resolved to ticks, ready to stream.

    Attributes:
        legs: Ordered legs, including the approach from wherever the arm was.
        rate: Tick rate the legs were sampled at.
    """

    legs: list[PlannedLeg] = field(default_factory=list)
    rate: float = DEFAULT_RATE_HZ

    def __len__(self) -> int:
        return len(self.legs)

    @property
    def ticks(self) -> int:
        """Total ticks across every leg."""
        return sum(len(leg) for leg in self.legs)

    @property
    def duration(self) -> float:
        """Total wall-clock seconds."""
        return self.ticks / self.rate


@dataclass
class ExecutionReport:
    """What actually happened when a plan was streamed.

    Attributes:
        legs_completed: Legs streamed to the end.
        ticks:          Ticks actually sent.
        stopped:        True if a stop was requested before the end.
        duration:       Measured wall-clock seconds.
        message:        Operator-facing summary.
    """

    legs_completed: int = 0
    ticks: int = 0
    stopped: bool = False
    duration: float = 0.0
    message: str = ""


def ease_in_offsets(
    q_now: np.ndarray | None, q_first: np.ndarray, seconds: float, rate: float
) -> np.ndarray:
    """Per-tick corrections that walk a leg's opening ticks back to ``q_now``.

    Starts at exactly the difference between where the arm is and where the
    plan begins, and fades to zero on the same minimum-jerk curve the legs
    themselves use, so the arm rejoins the plan with no step in position,
    velocity, or acceleration. Empty when there is nothing to take up.
    """
    if q_now is None:
        return np.empty((0, 0), dtype=np.float32)
    offset = np.asarray(q_now, dtype=np.float32) - np.asarray(q_first, dtype=np.float32)
    n = max(0, round(seconds * rate))
    if n == 0 or not offset.any():
        return np.empty((0, 0), dtype=np.float32)
    decay = 1.0 - ease(np.arange(n) / n)
    return (decay[:, None] * offset).astype(np.float32)


class MotionCommander:
    """Plans Cartesian and joint moves, and streams them to a robot.

    Args:
        kinematics:  Shared :class:`~.kinematics.CartesianKinematics`.
        robot:       Any :class:`~almond_axol.robot.base.RobotBase` — a
            :class:`~.virtual.VirtualRobot` for simulation, an
            :class:`~almond_axol.robot.axol.Axol` for hardware.
        rate_hz:     Control rate for planning and streaming.
        has_gripper: False on the gripperless SKU; gripper commands are then
            planned but ignored on write, exactly as the hardware does.

    Example::

        commander = MotionCommander(kinematics, robot)
        plan = commander.plan_program(program, start_q=await commander.current_q())
        report = await commander.execute(plan)
    """

    def __init__(
        self,
        kinematics: CartesianKinematics,
        robot: RobotBase,
        *,
        rate_hz: float = DEFAULT_RATE_HZ,
        has_gripper: bool = True,
        ease_in: float = DEFAULT_EASE_IN,
    ) -> None:
        self.kinematics = kinematics
        self.robot = robot
        self.rate_hz = rate_hz
        self.has_gripper = has_gripper
        self.ease_in = ease_in
        self._last_commanded: np.ndarray | None = None

    # -- state -----------------------------------------------------------

    async def current_state(self) -> tuple[np.ndarray, Grips]:
        """Read the robot's joint vector and gripper openings.

        Falls back to the rest pose for an arm the robot reports as absent, so
        a single-arm machine still plans.
        """
        left, right = await self.robot.get_positions()
        q = self.kinematics.rest_q()
        grips = [1.0, 1.0]
        for i, (arm, values) in enumerate(((Arm.LEFT, left), (Arm.RIGHT, right))):
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float32)
            q[self.kinematics.indices(arm)] = values[:N_ARM_JOINTS]
            if values.size > N_ARM_JOINTS:
                grips[i] = float(values[N_ARM_JOINTS])
        return q, (grips[0], grips[1])

    async def current_q(self) -> np.ndarray:
        """The robot's current full joint vector."""
        return (await self.current_state())[0]

    async def current_poses(self, tool: Tool | None = None) -> dict[Arm, Pose]:
        """Current tool pose of each arm, in world coordinates."""
        return self.kinematics.fk(await self.current_q(), tool)

    def arm_command(self, q: np.ndarray, grips: Grips) -> tuple[np.ndarray, np.ndarray]:
        """Split a full joint vector into the two ``(8,)`` arrays the robot takes."""
        q = np.asarray(q, dtype=np.float32)
        out = []
        for arm, grip in zip(ARMS, grips):
            values = np.zeros(N_ARM_JOINTS + 1, dtype=np.float32)
            values[:N_ARM_JOINTS] = q[self.kinematics.indices(arm)]
            values[N_ARM_JOINTS] = grip
            out.append(values)
        return out[0], out[1]

    # -- planning --------------------------------------------------------

    def plan_joint(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        *,
        speed: float,
        min_duration: float = 0.5,
    ) -> list[np.ndarray]:
        """Plan a ``MoveJ``: joint-space interpolation clear of the torso."""
        with self.kinematics.config_lock:
            return plan_collision_aware_trajectory(
                self.kinematics.solver.robot,
                self.kinematics.solver.robot_coll,
                np.asarray(q_from, dtype=np.float32),
                np.asarray(q_to, dtype=np.float32),
                speed=speed,
                rate=self.rate_hz,
                min_duration=min_duration,
            )

    def plan_linear(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        *,
        speed: float,
        ang_speed: float,
        tool: Tool | None = None,
        label: str = "segment",
    ) -> list[np.ndarray]:
        """Plan a ``MoveL``: the tool held to a straight world-frame line.

        Raises:
            PathPlanningError: The line leaves the workspace, or a sample
                cannot be resolved. Raised before anything moves.
        """
        tool = tool or self.kinematics.tool
        # plan_linear_segment scales the solver's pose weights and restores
        # them, exactly as solve() does — and against the same process-global
        # config object. Overlapping the two lets one capture the other's
        # scaled values as its "saved" baseline and restore those permanently.
        with self.kinematics.config_lock:
            return plan_linear_segment(
                self.kinematics.solver,
                np.asarray(q_from, dtype=np.float32),
                np.asarray(q_to, dtype=np.float32),
                speed=speed,
                ang_speed=ang_speed,
                rate=self.rate_hz,
                tool_offset=tool.vector,
                label=label,
            )

    def plan_to_poses(
        self,
        targets: Mapping[Arm, Pose],
        q_from: np.ndarray,
        *,
        move: MoveType = MoveType.LINEAR,
        speed: float | None = None,
        ang_speed: float | None = None,
        tool: Tool | None = None,
        grips: Grips = (1.0, 1.0),
        label: str = "move",
    ) -> PlannedLeg:
        """Solve for ``targets`` and plan a single leg reaching them.

        Raises:
            UnreachableError:  No joint configuration reaches the targets.
            PathPlanningError: A linear path between here and there does not exist.
        """
        tool = tool or self.kinematics.tool
        solution = self.kinematics.solve(
            targets, seed=q_from, tool=tool, raise_on_failure=True
        )
        if move is MoveType.LINEAR:
            trajectory = self.plan_linear(
                q_from,
                solution.q,
                speed=speed if speed is not None else 0.10,
                ang_speed=ang_speed if ang_speed is not None else 0.8,
                tool=tool,
                label=label,
            )
        else:
            trajectory = self.plan_joint(
                q_from, solution.q, speed=speed if speed is not None else 0.6
            )
        return PlannedLeg(
            label=label, move=move, q=trajectory, grips=[grips] * len(trajectory)
        )

    def plan_program(
        self,
        program: Program,
        start_q: np.ndarray,
        *,
        start_grips: Grips = (1.0, 1.0),
        loops: int = 1,
        approach: bool = True,
    ) -> PlannedProgram:
        """Resolve and plan a whole program, from wherever the arm is now.

        The approach leg is always a ``MoveJ``: the arm starts somewhere that
        is not on any waypoint's straight line, and forcing a linear move from
        an arbitrary pose is how you drive an elbow through the torso.

        Args:
            program:     The program to fly. Unresolved waypoints are solved
                in place, each seeded from the previous one.
            start_q:     Current full joint vector.
            start_grips: Current gripper openings.
            loops:       Times to run the waypoint list. ``0`` plans a single
                cycle that closes back to waypoint 1, for a caller that will
                replay it indefinitely. Negative values are rejected.
            approach:    Include the initial move from ``start_q``.

        Returns:
            A :class:`PlannedProgram`.

        Raises:
            ValueError:        The program has no waypoints.
            UnreachableError:  A waypoint cannot be solved.
            PathPlanningError: A linear leg cannot be flown.
        """
        if len(program) == 0:
            raise ValueError("the program has no waypoints to fly")

        failed = program.resolve(self.kinematics, seed=start_q)
        if failed:
            labels = ", ".join(program[i].label or f"waypoint {i + 1}" for i in failed)
            # Re-solve the first failure so the operator gets the actual reason
            # rather than a generic one.
            first = self.kinematics.solve(
                {arm: program.world_pose(failed[0], arm) for arm in ARMS},
                seed=start_q,
                tool=program.tool,
            )
            raise PathPlanningError(
                f"cannot fly {labels}. {first.message} "
                "Jog to it first to see how close the arm gets."
            )

        rest = VRTeleopConfig()
        legs: list[PlannedLeg] = []
        q_cursor = np.asarray(start_q, dtype=np.float32)
        grip_cursor = start_grips

        def waypoint_grips(index: int) -> Grips:
            return (program[index].grip(Arm.LEFT), program[index].grip(Arm.RIGHT))

        def settle(
            q_end: np.ndarray, held: Grips, arrival: Grips, dwell: float
        ) -> tuple[list[np.ndarray], list[Grips]]:
            """Work the grippers with the arm stationary, then hold.

            Grippers move at the waypoint rather than during the leg, so a
            grasp closes on the object where it is meant to and not somewhere
            along the way.
            """
            steps = (
                max(1, round(program.grip_time * self.rate_hz))
                if held != arrival and self.has_gripper
                else 1
            )
            hold = max(0, round(dwell * self.rate_hz))
            qs, gs = [], []
            for step in range(steps + hold):
                alpha = min(1.0, (step + 1) / steps)
                qs.append(q_end)
                gs.append(
                    (
                        held[0] + (arrival[0] - held[0]) * alpha,
                        held[1] + (arrival[1] - held[1]) * alpha,
                    )
                )
            return qs, gs

        if loops < 0:
            raise ValueError(f"loops must be 0 (forever) or positive, got {loops}")

        order = list(range(len(program)))
        if loops != 1 and len(program) > 1:
            # Close the cycle so a repeating run flows back to waypoint 1 the
            # same way it moves anywhere else. Needed for *every* repeat, not
            # just the infinite one: without it the second pass starts from
            # waypoint 1's configuration while the arm is at the last
            # waypoint's, and ``execute`` absorbs the whole difference into its
            # 0.4 s ease-in. Measured on a 3-waypoint program, that seam was
            # 0.217 rad against a within-leg peak of 0.002.
            order.append(0)

        for position, index in enumerate(order):
            waypoint = program[index]
            q_target = waypoint.full_q(self.kinematics)
            label = waypoint.label or f"waypoint {index + 1}"
            is_approach = position == 0

            if is_approach and not approach:
                q_cursor = q_target
                grip_cursor = waypoint_grips(index)
                continue

            move = MoveType.JOINT if is_approach else waypoint.move
            speed = waypoint.speed
            if move is MoveType.LINEAR:
                trajectory = self.plan_linear(
                    q_cursor,
                    q_target,
                    speed=speed if speed is not None else program.speed,
                    ang_speed=program.ang_speed,
                    tool=program.tool,
                    label=label,
                )
            else:
                trajectory = self.plan_joint(
                    q_cursor,
                    q_target,
                    speed=(
                        speed
                        if speed is not None
                        else (rest.reset_speed if is_approach else program.joint_speed)
                    ),
                    min_duration=rest.reset_min_duration if is_approach else 0.5,
                )

            arrival = waypoint_grips(index)
            dwell = waypoint.dwell if waypoint.dwell is not None else program.dwell
            settle_q, settle_g = settle(trajectory[-1], grip_cursor, arrival, dwell)
            legs.append(
                PlannedLeg(
                    label=("approach " + label) if is_approach else label,
                    move=move,
                    q=list(trajectory) + settle_q,
                    grips=[grip_cursor] * len(trajectory) + settle_g,
                )
            )
            q_cursor = trajectory[-1]
            grip_cursor = arrival

        # Hard gate. Every cost in the solver is soft — including
        # self-collision — so "the planner tried to avoid the torso" is not the
        # same claim as "this trajectory is clear of it". Endpoints being clear
        # is not enough either: a straight line between two clear waypoints can
        # sweep straight through the column. Check every tick, and refuse.
        for index, leg in enumerate(legs):
            clearance, tick = self.kinematics.clearance_along(leg.q)
            if clearance < 0.0:
                raise PathPlanningError(
                    f"{leg.label}: the arm passes {abs(clearance) * 1e3:.0f} mm "
                    f"through the torso {tick / max(1, len(leg.q)):.0%} of the way "
                    f"along (leg {index + 1} of {len(legs)}). Move the waypoints "
                    "out from the centre column, or use MoveJ to go around."
                )

        planned = PlannedProgram(legs=legs, rate=self.rate_hz)
        if loops > 1:
            # Repeats replay every leg except the approach, which would throw
            # the arm back to where playback began.
            body = legs[1:] if approach else legs
            for _ in range(loops - 1):
                # Copies, not the same objects: a caller that relabels or trims
                # one leg should not silently edit every repeat of it.
                planned.legs.extend(replace(leg) for leg in body)
        _logger.info(
            "planned %d legs, %d ticks, %.1fs",
            len(planned.legs),
            planned.ticks,
            planned.duration,
        )
        return planned

    # -- execution -------------------------------------------------------

    async def execute(
        self,
        planned: PlannedProgram | PlannedLeg,
        *,
        stop_event: threading.Event | None = None,
        on_progress: Callable[[int, PlannedLeg, float], None] | None = None,
        blend_in: bool = True,
    ) -> ExecutionReport:
        """Stream a plan to the robot at the control rate.

        Args:
            planned:     A whole program or a single leg.
            stop_event:  Checked every tick; setting it stops cleanly between
                ticks, leaving the arm wherever it was commanded last.
            on_progress: Called as ``(leg_index, leg, fraction)`` at the start
                of each leg and periodically within it.
            blend_in:    Ease onto the first leg from the current commanded
                configuration. Disable only when the caller has already
                arranged continuity.

        Returns:
            An :class:`ExecutionReport`.
        """
        program = (
            planned
            if isinstance(planned, PlannedProgram)
            else PlannedProgram(legs=[planned], rate=self.rate_hz)
        )
        dt = 1.0 / program.rate
        report = ExecutionReport()
        started = time.monotonic()

        q_previous = self._last_commanded
        if q_previous is None:
            q_previous = await self.current_q()

        for leg_index, leg in enumerate(program.legs):
            if not len(leg):
                report.legs_completed += 1
                continue
            blend = (
                ease_in_offsets(q_previous, leg.q[0], self.ease_in, program.rate)
                if blend_in
                else np.empty((0, 0), dtype=np.float32)
            )
            if len(blend):
                # plan_program's gate checks ``leg.q``; these opening ticks are
                # ``leg.q + blend``, a raw joint interpolation from wherever the
                # arm actually is. Normally the offset is a milliradian, but it
                # is arbitrary whenever the arm is not where the plan assumed,
                # so the streamed path has to be checked too.
                blended = [leg.q[i] + blend[i] for i in range(len(blend))]
                clearance, tick = self.kinematics.clearance_along(blended)
                if clearance < 0.0:
                    report.stopped = True
                    report.duration = time.monotonic() - started
                    report.message = (
                        f"Refused to start {leg.label}: easing in from the arm's "
                        f"current pose would pass {abs(clearance) * 1e3:.0f} mm "
                        f"through the torso at tick {tick}. Move the arm closer "
                        "to the start of the program first."
                    )
                    _logger.warning("%s", report.message)
                    return report
            if on_progress is not None:
                on_progress(leg_index, leg, 0.0)

            for step, (q, grips) in enumerate(zip(leg.q, leg.grips)):
                if stop_event is not None and stop_event.is_set():
                    report.stopped = True
                    report.ticks += step  # ticks actually sent before stopping
                    report.duration = time.monotonic() - started
                    report.message = f"Stopped during {leg.label}."
                    return report
                loop_start = time.monotonic()
                if step < len(blend):
                    q = q + blend[step]
                left, right = self.arm_command(q, grips)
                await self.robot.motion_control(left=left, right=right)
                self._last_commanded = q
                q_previous = q
                if on_progress is not None and step % 25 == 0:
                    on_progress(leg_index, leg, (step + 1) / len(leg))
                spent = time.monotonic() - loop_start
                if spent < dt:
                    await asyncio.sleep(dt - spent)

            report.legs_completed += 1
            report.ticks += len(leg)
            if on_progress is not None:
                on_progress(leg_index, leg, 1.0)

        report.duration = time.monotonic() - started
        report.message = (
            f"Completed {report.legs_completed} leg(s) in {report.duration:.1f}s."
        )
        return report

    async def move_to_poses(
        self,
        targets: Mapping[Arm, Pose],
        *,
        move: MoveType = MoveType.LINEAR,
        speed: float | None = None,
        ang_speed: float | None = None,
        tool: Tool | None = None,
        stop_event: threading.Event | None = None,
    ) -> ExecutionReport:
        """Plan and run a single move to absolute world poses.

        The one-liner an application wants: ``await commander.move_to_poses({Arm.LEFT: pose})``.
        """
        q_now, grips = await self.current_state()
        leg = self.plan_to_poses(
            targets,
            q_now,
            move=move,
            speed=speed,
            ang_speed=ang_speed,
            tool=tool,
            grips=grips,
            label="move to pose",
        )
        return await self.execute(leg, stop_event=stop_event)

    def adopt_commanded(self, q: np.ndarray) -> None:
        """Record ``q`` as the configuration most recently commanded.

        For a caller that drives :meth:`RobotBase.motion_control` itself — the
        jog pendant streams its own rate-limited setpoints — so the next
        :meth:`execute` blends from where the arm actually is. Left stale, the
        blend's first tick commands a configuration the arm left minutes ago,
        which on hardware is a step change large enough to be rejected outright.
        """
        self._last_commanded = np.asarray(q, dtype=np.float32).copy()

    def reset_command_state(self) -> None:
        """Forget the last commanded configuration.

        Call after the arm has been moved by anything other than this
        commander — hand-guiding, another operation — so the next leg blends
        from where the arm actually is instead of from a stale setpoint.
        """
        self._last_commanded = None
