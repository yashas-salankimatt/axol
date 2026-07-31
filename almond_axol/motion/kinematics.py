"""Absolute Cartesian kinematics — FK and IK against a pose you actually typed.

:class:`~almond_axol.kinematics.solver.KinematicsSolver` already solves for
absolute end-effector poses, but its defaults are tuned for *live VR teleop*:
tracking a hand that jitters, where the rest and posture costs usefully damp
the target's noise. Pointed at an exact target those same costs pull the
gripper off it and leave it there. Measured against a target 12 cm forward and
8 cm up of the rest pose, the solver plateaus **42 mm** (left) and **114 mm**
(right) away and stops improving.

This module is the adapter between "a pose an operator meant" and that solver:

- **Pose weights are scaled while solving.** The same trick
  :func:`~almond_axol.kinematics.path.plan_linear_segment` already uses, taken
  further because a jog target is exact rather than merely close. Measured
  residual against the case above: 42/114 mm at 1x, 0.6/7.6 mm at 10x,
  0.01/0.08 mm at 100x. :data:`DEFAULT_WEIGHT_SCALE` is 100.
- **It iterates to a tolerance instead of a fixed count.**
  ``KinematicsSolver.ik`` clamps every call to
  ``KinematicsConfig.max_joint_delta`` (about 2°), so one call cannot cross a
  workspace. Solving is pure arithmetic — the clamp that matters for safety is
  the planner's speed limit and the robot's own ``max_step_rad`` — so the
  clamp is opened up during a solve and the loop runs until the error stops
  falling.
- **It refuses to seed from a singularity.** At ``q = 0`` both arms hang
  straight down, and ``shoulder_3`` and ``wrist_1`` are roll joints along the
  link axis: they move the tip *not at all*. The Jacobian loses two ranks and
  the solver returns its seed unchanged forever — measured at 400 consecutive
  calls with zero joint motion. Seeds default to the rest pose, and
  :meth:`CartesianKinematics.singularity_margin` reports how close any
  configuration is to the same trap.
- **Both arms are always passed to the solver.** Passing ``None`` for one arm
  is a different JAX trace and costs a multi-second recompile. An arm with no
  target is pinned to its seed pose instead, and any null-space wander the
  solve introduces is discarded.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..kinematics.config import KinematicsConfig
from ..kinematics.solver import KinematicsSolver
from ..teleop.config import VRTeleopConfig
from .frames import ARMS, Arm, Pose, Tool, orthonormalize

_logger = logging.getLogger(__name__)

DEFAULT_WEIGHT_SCALE = 100.0
"""Factor applied while solving to the pose costs **and the constraints**.

Joint limits and self-collision are scaled by the same factor as the pose
terms, so their balance against it is exactly what the tuned defaults set.
Only the *regularisers* — rest, posture, manipulability — are left alone, and
so become relatively negligible. That is the whole intent: they are what pulls
a solve off an exact target, and they are not what keeps the arm out of the
torso.

Scaling the pose terms alone is a trap worth naming, because it looks like it
works. Measured against a target on the far side of the torso pillar, with the
pose weights at 100x and the constraints left at their defaults, the solver
drove the arm **186 mm** into the torso to land the pose to 0.1 mm — the
collision cost had become 1.5% of the pose cost and was simply outvoted.
Scaling both together reaches the same 0.0 mm on a clear target while holding
penetration to the same ~24 mm the untouched defaults give.
"""

SOLVE_COLLISION_MARGIN = 0.005
"""Self-collision standoff (m) used while solving, replacing the teleop default.

``KinematicsConfig.self_collision_margin`` is 25 mm, which is a *standoff* for
live teleop: it gently discourages a hand-tracked arm from creeping toward the
torso. But ordinary working poses only clear the torso by about 10 mm, so at 25
mm that penalty is **chronically active** — and once its weight is scaled with
the pose cost, a gentle discouragement becomes a dominant force that shoves the
arm off the target it was asked for. Measured: holding a pose the arm was
already in came out 2.04 mm off.

Shrinking the margin turns it from a chronic pull into a barrier that engages
only at genuine contact, which is where a large weight belongs. Measured
against a target on the far side of the torso: at 25 mm the arm ended 18.6 mm
*inside* the column; at 5 mm it never enters at all, stopping at the surface
and reporting the pose as unreached — which is what a caller can act on."""

DEFAULT_SOLVE_STEP = 0.25
"""Per-call joint clamp (rad) used while solving, replacing the teleop default.

Roughly 14°, so a solve crosses the workspace in a few dozen calls instead of
a few hundred, while still being small enough that each call's linearisation
holds.
"""

POSITION_TOLERANCE = 1e-4
"""Position error (m) at which a solve is called converged — 0.1 mm."""

ORIENTATION_TOLERANCE = 1e-3
"""Orientation error (rad) at which a solve is called converged — 0.06°."""

SINGULARITY_THRESHOLD = 0.02
"""Smallest Jacobian singular value below which a pose is flagged as singular.

Calibrated against the two poses whose behaviour is known: the straight-down
zero configuration (genuinely rank-deficient, margin 0) and the rest pose
(solves fine). See ``tests/test_kinematics.py``.
"""

_JACOBIAN_STEP = 1e-4
"""Central-difference step (rad) for the numerical Jacobian."""

_STALL_PATIENCE = 5
"""Consecutive non-improving solver calls tolerated before giving up."""

_READY_SHOULDER_1 = -0.6
"""``shoulder_1`` angle (rad) of :meth:`CartesianKinematics.ready_q`."""

_READY_SHOULDER_2 = -0.6
"""``shoulder_2`` angle (rad) of :meth:`CartesianKinematics.ready_q`.

The one that swings the arm laterally clear of the torso. Leaving it at zero
is what left the original ready pose with only 9.7 mm of torso clearance —
close enough that merely *rotating* the tool in place drove the gripper into
the centre column."""

_READY_ELBOW = 1.6
"""``elbow`` angle (rad) of :meth:`CartesianKinematics.ready_q`."""


class UnreachableError(RuntimeError):
    """A requested pose could not be reached within tolerance.

    Carries the solution the solver settled on so a caller can show the
    operator *how far off* it ended up rather than only that it failed.
    """

    def __init__(self, message: str, solution: IKSolution) -> None:
        super().__init__(message)
        self.solution = solution


@dataclass
class IKSolution:
    """The outcome of one :meth:`CartesianKinematics.solve` call.

    Attributes:
        q:                 Full ``(N,)`` joint vector, always populated — even
            when ``reached`` is False, so a UI can show the closest approach.
        reached:           True if every requested arm landed within tolerance.
        position_error:    Per-arm residual position error, metres.
        orientation_error: Per-arm residual orientation error, radians.
        iterations:        Solver calls actually spent.
        clearance:         Smallest gap (m) between any checked link pair.
            **Negative means the arm is inside the torso.** Self-collision is a
            soft cost, not a hard constraint, so a target that can only be
            reached through the torso yields a solution that penetrates it —
            reporting the number is what lets a caller refuse.
        message:           Operator-facing summary; empty when ``reached``.
    """

    q: np.ndarray
    reached: bool
    position_error: dict[Arm, float] = field(default_factory=dict)
    orientation_error: dict[Arm, float] = field(default_factory=dict)
    iterations: int = 0
    clearance: float = float("inf")
    message: str = ""

    @property
    def collides(self) -> bool:
        """True if the solution puts an arm inside the torso."""
        return self.clearance < 0.0

    @property
    def usable(self) -> bool:
        """Reached its target *and* clear of the torso — safe to command."""
        return self.reached and not self.collides

    @property
    def worst_position_error(self) -> float:
        """Largest per-arm position error (m), or 0 if nothing was requested."""
        return max(self.position_error.values(), default=0.0)

    @property
    def worst_orientation_error(self) -> float:
        """Largest per-arm orientation error (rad), or 0 if nothing was requested."""
        return max(self.orientation_error.values(), default=0.0)


def _matrix_log(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle vector (the SO(3) logarithm)."""
    m = np.asarray(rotation, dtype=np.float64)
    cos = max(-1.0, min(1.0, (float(np.trace(m)) - 1.0) * 0.5))
    theta = float(np.arccos(cos))
    if theta < 1e-9:
        # First-order: the skew part is already the rotation vector.
        return (
            np.array(
                [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]],
                dtype=np.float64,
            )
            * 0.5
        )
    if theta > np.pi - 1e-6:
        # Near π the skew part vanishes, so the axis has to come from the
        # symmetric part. Taking the elementwise sqrt of the diagonal recovers
        # only magnitudes and loses the axis components' signs *relative to
        # each other* — it returns [+a, +a, 0] for an axis of [+a, -a, 0].
        # ``R + I = 2·a·aᵀ`` near π, so any non-degenerate column of it is
        # parallel to the axis with its signs intact.
        candidate = m + np.eye(3)
        axis = candidate[:, int(np.argmax(np.diag(candidate)))]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return np.zeros(3, dtype=np.float64)
        return axis / norm * theta
    return (
        np.array(
            [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]], dtype=np.float64
        )
        * theta
        / (2.0 * np.sin(theta))
    )


class CartesianKinematics:
    """FK and absolute-pose IK for the Axol, in world coordinates.

    Wraps a :class:`~almond_axol.kinematics.solver.KinematicsSolver` — building
    one triggers a multi-second JAX compile, so share an instance.

    Args:
        config: Solver cost weights. The defaults are the teleop-tuned ones;
            this class scales the pose terms itself while solving.
        tool:   Default tool centre point every pose is expressed at. Override
            per call where a flow mixes tools.

    Example::

        kin = CartesianKinematics()
        here = kin.fk(kin.rest_q())[Arm.LEFT]
        there = here.translated((0.0, -0.10, 0.05))   # 10 cm forward, 5 cm up
        solution = kin.solve({Arm.LEFT: there})
        assert solution.reached
    """

    def __init__(
        self,
        config: KinematicsConfig | None = None,
        tool: Tool | None = None,
    ) -> None:
        from .frames import GRIPPER_TIP

        self.solver = KinematicsSolver(config or KinematicsConfig())
        self.tool = tool or GRIPPER_TIP
        self._indices = {
            Arm.LEFT: list(self.solver.left_indices),
            Arm.RIGHT: list(self.solver.right_indices),
        }
        # ``solver.config`` is process-global mutable state that solve() scales
        # and restores. Two overlapping solves would each capture the other's
        # scaled values as "saved" and restore those, corrupting the config for
        # the life of the process — including ``max_joint_delta``, which live
        # teleop depends on through the same solver.
        self._config_lock = threading.RLock()
        self._shoulder = {
            Arm.LEFT: np.asarray(self.solver._left_shoulder_pos, dtype=np.float32),
            Arm.RIGHT: np.asarray(self.solver._right_shoulder_pos, dtype=np.float32),
        }

    # -- structure -------------------------------------------------------

    @property
    def num_joints(self) -> int:
        """Total actuated joints across both arms."""
        return self.solver.num_joints

    @property
    def max_reach(self) -> float:
        """Largest shoulder-to-target distance (m) the solver will accept."""
        return self.solver.config.max_reach

    @property
    def config_lock(self) -> threading.RLock:
        """Guards the shared ``solver.config`` while it is scaled for a solve.

        Anything that scales those weights and restores them — this class, and
        :func:`~almond_axol.kinematics.path.plan_linear_segment` via
        :meth:`~.commander.MotionCommander.plan_linear` — must hold this, or
        two overlapping callers each capture the other's scaled values as their
        baseline and restore *those*, corrupting the config for the life of the
        process. ``max_joint_delta`` is among them, and live teleop depends on
        it through the same solver.
        """
        return self._config_lock

    def indices(self, arm: Arm) -> list[int]:
        """Indices of ``arm``'s seven joints within a full joint vector."""
        return list(self._indices[arm])

    def shoulder_position(self, arm: Arm) -> np.ndarray:
        """World position of ``arm``'s shoulder — the centre of its reach sphere."""
        return self._shoulder[arm].copy()

    def rest_q(self) -> np.ndarray:
        """The configured rest pose as a full joint vector.

        The default IK seed. Unlike the zero vector this is well away from the
        straight-down singularity, which is the whole reason it is the default.
        """
        cfg = VRTeleopConfig()
        q = np.zeros(self.num_joints, dtype=np.float32)
        q[self._indices[Arm.LEFT]] = cfg.rest_pose_left
        q[self._indices[Arm.RIGHT]] = cfg.rest_pose_right
        return q

    def ready_q(self) -> np.ndarray:
        """A bent, well-conditioned configuration to start Cartesian work from.

        :meth:`rest_q` is where the robot *parks*, and it is a poor place to
        work: measured, the rest pose puts the gripper mount 0.707 m from the
        shoulder against a 0.80 m reach limit — 88% extended — with a
        singularity margin of 0.043. Nearly every direction an operator jogs
        from there runs into the reach clamp or the degraded conditioning of a
        straight arm, which reads as "the IK is broken" when it is really "the
        arm is already at the edge of its envelope".

        This pose lifts the shoulder, swings it laterally, and bends the elbow,
        mirrored across the two arms so both tools reach forward and up.
        Measured: the tool sits 0.41 m out, 0.23 m forward and 0.33 m up, using
        0.531 m of a 0.80 m reach, with a singularity margin of 0.184 and
        **211 mm of torso clearance**.

        That last number is why ``shoulder_2`` is in here. An earlier version
        left it at zero and cleared the torso by 9.7 mm — enough that rotating
        the tool in place pushed the gripper into the centre column, which is
        not a state to hand an operator as the starting pose.
        """
        q = self.rest_q()
        for arm in ARMS:
            index = self._indices[arm]
            sign = 1.0 if arm is Arm.LEFT else -1.0
            q[index[0]] = _READY_SHOULDER_1 * sign
            q[index[1]] = _READY_SHOULDER_2 * sign
            q[index[3]] = _READY_ELBOW * sign
        return q

    def arm_q(self, q: np.ndarray, arm: Arm) -> np.ndarray:
        """Slice ``arm``'s seven joint angles out of a full joint vector."""
        return np.asarray(q, dtype=np.float32)[self._indices[arm]].copy()

    # -- forward kinematics ----------------------------------------------

    def fk(self, q: np.ndarray, tool: Tool | None = None) -> dict[Arm, Pose]:
        """Tool poses for both arms at joint configuration ``q``."""
        tool = tool or self.tool
        left, right = self.solver.fk(np.asarray(q, dtype=np.float32))
        out = {}
        for arm, se3 in ((Arm.LEFT, left), (Arm.RIGHT, right)):
            mount = Pose(
                np.asarray(se3.translation(), dtype=np.float32),
                orthonormalize(
                    np.asarray(se3.rotation().as_matrix(), dtype=np.float32)
                ),
            )
            out[arm] = mount.apply_tool(tool)
        return out

    def fk_arm(self, q: np.ndarray, arm: Arm, tool: Tool | None = None) -> Pose:
        """Tool pose for a single arm at ``q``."""
        return self.fk(q, tool)[arm]

    # -- reachability and conditioning -----------------------------------

    def reach_excess(self, pose: Pose, arm: Arm, tool: Tool | None = None) -> float:
        """Metres by which ``pose`` exceeds ``arm``'s reach sphere. ``<= 0`` is fine.

        Checked against the *mount*, since that is the point the solver clamps.
        """
        mount = pose.remove_tool(tool or self.tool)
        return (
            float(np.linalg.norm(mount.position - self._shoulder[arm])) - self.max_reach
        )

    def jacobian(self, q: np.ndarray, arm: Arm, tool: Tool | None = None) -> np.ndarray:
        """Numerical ``(6, 7)`` tool Jacobian for ``arm`` at ``q``.

        Rows 0-2 are translational, 3-5 rotational, both in world axes. Central
        differences: an analytic Jacobian would need the joint axes pulled out
        of the URDF, and at 14 FK calls this is comfortably fast enough for a
        UI readout and is far harder to get subtly wrong.
        """
        tool = tool or self.tool
        q = np.asarray(q, dtype=np.float32)
        columns = []
        for index in self._indices[arm]:
            plus, minus = q.copy(), q.copy()
            plus[index] += _JACOBIAN_STEP
            minus[index] -= _JACOBIAN_STEP
            pose_plus = self.fk_arm(plus, arm, tool)
            pose_minus = self.fk_arm(minus, arm, tool)
            linear = (pose_plus.position - pose_minus.position) / (2.0 * _JACOBIAN_STEP)
            angular = _matrix_log(
                pose_plus.rotation.astype(np.float64)
                @ pose_minus.rotation.astype(np.float64).T
            ) / (2.0 * _JACOBIAN_STEP)
            columns.append(np.concatenate([linear, angular]))
        return np.column_stack(columns).astype(np.float64)

    def self_collision_clearance(self, q: np.ndarray) -> float:
        """Smallest gap (m) between any checked link pair at ``q``.

        Negative means penetration. The checked set is the solver's own —
        torso against arm, with pairs that already overlap at the folded home
        pose excluded (see
        :func:`~almond_axol.kinematics.solver._build_robot_collision`). The
        capsule fit is conservative, so a millimetre or two negative can be the
        approximation rather than real contact; a couple of centimetres is not.
        """
        distances = self.solver.robot_coll.compute_self_collision_distance(
            self.solver.robot, jnp.asarray(q, dtype=jnp.float32)
        )
        return float(np.min(np.asarray(distances)))

    def clearance_along(self, trajectory: Sequence[np.ndarray]) -> tuple[float, int]:
        """Worst clearance over a whole trajectory, and where it occurs.

        Returns ``(clearance, index)``; negative clearance means that tick puts
        an arm inside the torso. Evaluated for every sample, batched, because
        checking only the endpoints is exactly how a move whose two ends are
        clear sweeps straight through the column in between.
        """
        if not len(trajectory):
            return float("inf"), -1
        batch = jnp.asarray(np.asarray(trajectory, dtype=np.float32))
        distances = jax.vmap(
            lambda q: self.solver.robot_coll.compute_self_collision_distance(
                self.solver.robot, q
            )
        )(batch)
        per_tick = np.asarray(distances).min(axis=1)
        index = int(np.argmin(per_tick))
        return float(per_tick[index]), index

    def singularity_margin(
        self, q: np.ndarray, arm: Arm, tool: Tool | None = None
    ) -> float:
        """Smallest singular value of ``arm``'s Jacobian — distance to singularity.

        Zero means a direction of tool motion no combination of joint rates can
        produce. Below :data:`SINGULARITY_THRESHOLD` the IK will struggle and
        the arm may move in large null-space sweeps for small tool motions.
        """
        return float(np.linalg.svd(self.jacobian(q, arm, tool), compute_uv=False)[-1])

    def is_singular(self, q: np.ndarray, arm: Arm, tool: Tool | None = None) -> bool:
        """True if ``arm`` at ``q`` is at or near a singularity."""
        return self.singularity_margin(q, arm, tool) < SINGULARITY_THRESHOLD

    def desingularize(
        self,
        q: np.ndarray,
        arms: tuple[Arm, ...] = ARMS,
        *,
        tool: Tool | None = None,
        target_margin: float = 2.0 * SINGULARITY_THRESHOLD,
        blend: float = 0.3,
        max_tries: int = 12,
    ) -> np.ndarray:
        """Nudge a rank-deficient seed toward the rest pose until IK can move.

        A singular configuration is not a hard target to reach, it is a place
        the solver cannot *leave*: the Jacobian has no column that moves the
        tool in some direction, every proposed step is rejected, and the seed
        comes back unchanged. Blending toward the rest pose breaks the
        degeneracy, and it is the only way a caller who seeded from zeros gets
        a usable answer.

        ``target_margin`` deliberately overshoots
        :data:`SINGULARITY_THRESHOLD`. Stopping the instant the margin crosses
        it leaves the arm *just* conditioned enough to pass the check and still
        frozen in practice: measured from the zero pose, clearing to 0.021
        produced a solver that returned its seed unchanged for 25 consecutive
        calls. Near-singular geometry and an active self-collision cost reject
        steps together, so the seed has to land somewhere comfortable — the
        rest pose itself sits at about 0.043.

        Returns ``q`` unchanged when every requested arm is already well
        conditioned.
        """
        q = np.asarray(q, dtype=np.float32).copy()
        rest = self.rest_q()
        for arm in arms:
            idx = self._indices[arm]
            # Only an arm that is *actually* singular gets moved. ``target_margin``
            # deliberately overshoots the threshold, and applying that overshoot
            # as the entry condition too would drag any merely-mediocre arm
            # toward rest — including an arm the caller gave no target, whose
            # configuration is supposed to be preserved exactly.
            if self.singularity_margin(q, arm, tool) >= SINGULARITY_THRESHOLD:
                continue
            for _ in range(max_tries):
                if self.singularity_margin(q, arm, tool) >= target_margin:
                    break
                q[idx] = (1.0 - blend) * q[idx] + blend * rest[idx]
            else:
                _logger.debug(
                    "%s arm seed still poorly conditioned (margin %.4f) after %d nudges",
                    arm.value,
                    self.singularity_margin(q, arm, tool),
                    max_tries,
                )
        return q

    # -- inverse kinematics ----------------------------------------------

    def solve(
        self,
        targets: Mapping[Arm, Pose],
        seed: np.ndarray | None = None,
        *,
        tool: Tool | None = None,
        position_tolerance: float = POSITION_TOLERANCE,
        orientation_tolerance: float = ORIENTATION_TOLERANCE,
        weight_scale: float = DEFAULT_WEIGHT_SCALE,
        max_iterations: int = 120,
        solve_step: float = DEFAULT_SOLVE_STEP,
        raise_on_failure: bool = False,
    ) -> IKSolution:
        """Solve for joint angles that put each arm's tool at its target pose.

        Args:
            targets:  Target **tool** pose per arm, in world coordinates. An
                arm left out holds the configuration it has in ``seed``.
            seed:     Full ``(N,)`` starting configuration. Defaults to
                :meth:`rest_q` — never to zeros, which is singular.
            tool:     Tool centre point the targets are expressed at.
            position_tolerance:    Convergence threshold, metres.
            orientation_tolerance: Convergence threshold, radians.
            weight_scale:  Pose-cost multiplier while solving. See the module
                docstring; 1.0 reproduces the teleop-tuned behaviour and its
                centimetre-scale steady-state offset.
            max_iterations: Cap on solver calls.
            solve_step:  Per-call joint clamp (rad) during the solve.
            raise_on_failure: Raise :class:`UnreachableError` instead of
                returning an unusable solution — one that missed its target
                **or** that only reaches it through the torso. See
                :attr:`IKSolution.usable`.

        Returns:
            An :class:`IKSolution`. ``q`` is always the closest configuration
            found, so a UI can render the near miss.

        Raises:
            UnreachableError: Only when ``raise_on_failure`` is set.
            ValueError:       If ``targets`` is empty.
        """
        if not targets:
            raise ValueError("solve() needs at least one arm target")
        tool = tool or self.tool
        q_seed = self.rest_q() if seed is None else np.asarray(seed, dtype=np.float32)
        # Desingularize *both* arms, including any arm with no target of its
        # own. The solver poses one coupled least-squares problem over all 14
        # joints, so a rank-deficient block anywhere in it makes the whole
        # trust-region step get rejected: measured with the left arm well
        # conditioned and the right left at zeros, a solve moved *no* joint at
        # all, while nudging both moved the full 0.25 rad step.
        #
        # Only genuinely singular arms are touched (see :meth:`desingularize`),
        # so an arm with no target keeps its configuration exactly unless it was
        # sitting somewhere the solver cannot work from at all.
        q_seed = self.desingularize(q_seed, ARMS, tool=tool)

        # An arm with no target is pinned to its seed pose. Passing None would
        # be a fresh JAX trace and a multi-second recompile.
        seed_poses = self.fk(q_seed, tool)
        mount_targets = {
            arm: targets.get(arm, seed_poses[arm]).remove_tool(tool) for arm in ARMS
        }
        held = [arm for arm in ARMS if arm not in targets]

        blocked = [
            f"{arm.value} target is {self.reach_excess(targets[arm], arm, tool) * 1e3:.0f} mm "
            f"beyond the {self.max_reach:.2f} m reach limit"
            for arm in targets
            if self.reach_excess(targets[arm], arm, tool) > 0.0
        ]

        with self._config_lock:
            return self._solve_locked(
                targets,
                q_seed,
                mount_targets,
                held,
                blocked,
                tool,
                position_tolerance,
                orientation_tolerance,
                weight_scale,
                max_iterations,
                solve_step,
                raise_on_failure,
            )

    def _solve_locked(
        self,
        targets,
        q_seed,
        mount_targets,
        held,
        blocked,
        tool,
        position_tolerance,
        orientation_tolerance,
        weight_scale,
        max_iterations,
        solve_step,
        raise_on_failure,
    ) -> IKSolution:
        """The body of :meth:`solve`, run with the solver config lock held."""
        q = q_seed.copy()
        solver = self.solver
        saved = (
            solver.config.pos_weight,
            solver.config.ori_weight,
            solver.config.max_joint_delta,
            solver.config.limit_weight,
            solver.config.self_collision_weight,
            solver.config.self_collision_margin,
        )
        solver.config.pos_weight = saved[0] * weight_scale
        solver.config.ori_weight = saved[1] * weight_scale
        # Constraints scale with the objective. Leaving them behind is what
        # let a solve drive 186 mm through the torso — see DEFAULT_WEIGHT_SCALE.
        solver.config.limit_weight = saved[3] * weight_scale
        solver.config.self_collision_weight = saved[4] * weight_scale
        solver.config.self_collision_margin = SOLVE_COLLISION_MARGIN
        solver.config.max_joint_delta = solve_step

        # Track the best configuration seen rather than the last. A solve that
        # wanders after its closest approach should still report the closest
        # approach — that is the number an operator uses to decide whether a
        # pose is worth nudging or is simply not there.
        best_q = q.copy()
        best_error = np.inf  # combined position + orientation score
        stalled = 0
        iterations = 0
        try:
            for iterations in range(1, max_iterations + 1):
                q = solver.ik(
                    q,
                    left_pose=(
                        mount_targets[Arm.LEFT].position,
                        mount_targets[Arm.LEFT].rotation,
                    ),
                    right_pose=(
                        mount_targets[Arm.RIGHT].position,
                        mount_targets[Arm.RIGHT].rotation,
                    ),
                )
                # An arm with no target has already met it; anything the solve
                # did to it is null-space wander, so put it back.
                for arm in held:
                    q[self._indices[arm]] = q_seed[self._indices[arm]]

                actual = self.fk(q, tool)
                worst = max(actual[arm].distance_to(targets[arm]) for arm in targets)
                worst_angle = max(actual[arm].angle_to(targets[arm]) for arm in targets)
                # Improvement is measured against the previous best, so this
                # has to be read before the best is updated.
                # Score on both modalities. Tracking position alone discards a
                # strictly better answer whenever orientation is still falling —
                # measured, a 60°-off iterate was returned in place of a 24°-off
                # one because their position errors were within 1 mm.
                score = worst / position_tolerance + worst_angle / orientation_tolerance
                improved = score < best_error * 0.999
                if score < best_error:
                    best_error = score
                    best_q = q.copy()
                if worst <= position_tolerance and worst_angle <= orientation_tolerance:
                    break
                # Give up only after several consecutive non-improving calls.
                # One is not enough: the clamped step means an early call can
                # trade position error for the joint room to fix it next time.
                stalled = 0 if improved else stalled + 1
                if stalled >= _STALL_PATIENCE:
                    break
        finally:
            (
                solver.config.pos_weight,
                solver.config.ori_weight,
                solver.config.max_joint_delta,
                solver.config.limit_weight,
                solver.config.self_collision_weight,
                solver.config.self_collision_margin,
            ) = saved

        q = best_q
        actual = self.fk(q, tool)
        position_error = {arm: actual[arm].distance_to(targets[arm]) for arm in targets}
        orientation_error = {arm: actual[arm].angle_to(targets[arm]) for arm in targets}
        reached = all(v <= position_tolerance for v in position_error.values()) and all(
            v <= orientation_tolerance for v in orientation_error.values()
        )

        clearance = self.self_collision_clearance(q)

        message = ""
        if clearance < 0.0:
            message = (
                f"This pose puts the arm {abs(clearance) * 1e3:.0f} mm inside the "
                "torso. Move the target out from the centre column."
            )
        if not reached:
            arm_worst = max(position_error, key=lambda a: position_error[a])
            detail = (
                f"{arm_worst.value} arm ends {position_error[arm_worst] * 1e3:.1f} mm "
                f"and {np.degrees(orientation_error[arm_worst]):.1f}° from its target"
            )
            if blocked:
                message = f"Out of reach — {'; '.join(blocked)}."
            elif any(self.is_singular(q, arm, tool) for arm in targets):
                message = (
                    f"Near a singularity, so the pose cannot be resolved: {detail}. "
                    "Move the target away from the arm's fully-extended line."
                )
            elif clearance < 0.0:
                message = (
                    f"Blocked by the torso: {detail}, and the arm is "
                    f"{abs(clearance) * 1e3:.0f} mm inside the centre column."
                )
            else:
                message = (
                    f"Could not reach the pose: {detail} "
                    f"(torso clearance {clearance * 1e3:.0f} mm). It may be out of "
                    "reach, at a joint limit, or blocked by the centre column."
                )
            _logger.debug("IK did not converge: %s", message)

        solution = IKSolution(
            q=q.astype(np.float32),
            reached=reached,
            position_error=position_error,
            orientation_error=orientation_error,
            iterations=iterations,
            clearance=clearance,
            message=message,
        )
        if raise_on_failure and not solution.usable:
            raise UnreachableError(message, solution)
        return solution
