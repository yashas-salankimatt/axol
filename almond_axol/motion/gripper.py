"""Jaw geometry for the Axol gripper, measured from Almond's published CAD.

The bundled URDF models the gripper as a **single fixed lump**: one
20-triangle placeholder mesh welded to the wrist by ``left_gripper_0`` /
``right_gripper_0``, both of type ``fixed``. There are no finger links and no
finger joints anywhere in the model, so nothing can move — which is why the
gripper sliders drive real hardware but produce no visible motion in the
simulator (``robot/sim.py`` says so outright).

The real geometry does exist, just not in this repository. Almond publishes it
alongside the hardware guide:

- ``https://www.almond.bot/resources/gripper.step`` — 8.0 MB, AP242
- ``https://www.almond.bot/resources/gripper.pdf`` — the dimensioned drawing

The numbers below were measured from that STEP (converted with ``cascadio``,
tessellated at 0.05 mm). Keeping them here — rather than hard-coding them into
the viewer — means the eventual articulated URDF and the viewer overlay agree
by construction.

What the CAD shows
------------------

A **parallel two-jaw gripper**. A central motor (a 4310-class BLDC) drives two
carrier plates in opposite directions; each carrier is bolted to one jaw, so
the jaws translate symmetrically. That is the standard URDF gripper case: two
mirrored prismatic joints, not the internal linkage.

Measured in the CAD's own frame, in the state the model ships in:

===========================  ==========
overall envelope             119.8 x 68 x 146.6 mm  (matches the drawing)
one jaw                      23.74 x 68.00 x 79.34 mm
jaw inner faces              x = +/-36.16 mm  ->  **72.31 mm apart**
jaw outer faces              x = +/-59.90 mm  ->  119.8 mm overall
carrier plates               71.57 mm wide, offset in Y so they slide past
base plate                   102 x 65 mm
===========================  ==========

.. warning::
   **The stroke is an inference, not a specification.**
   ``GRIPPER_TRAVEL`` in :mod:`almond_axol.robot.axol` is 290 degrees of
   *motor* rotation; nothing in the SDK or the drawing maps that to jaw
   opening in millimetres, and the CAD ships in a single state. That state is
   taken to be **fully open**, because its 119.8 mm envelope is exactly the
   overall dimension the drawing calls out and drawings dimension the maximum
   envelope. :data:`MAX_TRAVEL` then follows from closing to contact.

   So a commanded ``1.0`` is treated as a 72.31 mm gap and ``0.0`` as closed.
   If the mechanism bottoms out before contact the real closed gap is larger.
   One measurement on the robot settles it: command 1.0, measure the gap;
   command 0.0, measure again. Correct :data:`OPEN_GAP` and
   :data:`CLOSED_GAP` and everything downstream follows.

.. warning::
   **Which way the jaws open is a convention that needs confirming.**
   The CAD opens along its own X. The URDF placeholder is a crude box whose
   extents (67 x 50 mm) match neither the CAD's 119.8 nor its 68, so it cannot
   be used to recover the mapping. :data:`JAW_AXIS` assumes the CAD's X maps to
   the gripper link's X. If the rendered jaws sit 90 degrees off the real ones,
   change it to ``1`` — nothing else needs touching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import GRIPPER_TIP_OFFSET
from .frames import Arm, Pose

# -- measured from gripper.step ---------------------------------------------

JAW_THICKNESS = 0.02374
"""One jaw's extent along the opening axis (m)."""

JAW_DEPTH = 0.06800
"""One jaw's extent across the opening axis (m)."""

JAW_HEIGHT = 0.07934
"""One jaw's extent along the tool axis (m). CAD z 22.30 -> 101.64 mm."""

OPEN_GAP = 0.07231
"""Gap between the jaw gripping faces when fully open (m). See the warning above."""

CLOSED_GAP = 0.0
"""Gap when fully closed (m). Assumes the jaws meet; see the warning above."""

MAX_TRAVEL = (OPEN_GAP - CLOSED_GAP) / 2.0
"""How far each jaw travels between closed and open (m) — 36.2 mm."""

JAW_AXIS = 0
"""Index of the gripper-link axis the jaws separate along. 0 = X. See above."""

TIP_Z = float(GRIPPER_TIP_OFFSET[2])
"""Tool-axis coordinate of the fingertips in the gripper link frame (-0.145 m)."""


@dataclass(frozen=True)
class Jaw:
    """One jaw's box, ready to draw.

    Attributes:
        pose:       World pose of the jaw's centre.
        dimensions: ``(x, y, z)`` box size in the jaw's own frame, metres.
    """

    pose: Pose
    dimensions: tuple[float, float, float]


def gap_for(opening: float) -> float:
    """Gap between the gripping faces (m) for a normalised opening in ``[0, 1]``."""
    opening = float(np.clip(opening, 0.0, 1.0))
    return CLOSED_GAP + opening * (OPEN_GAP - CLOSED_GAP)


def jaws(gripper_pose: Pose, opening: float, arm: Arm | None = None) -> list[Jaw]:
    """Where both jaws sit, given the gripper link's world pose.

    Args:
        gripper_pose: World pose of the gripper **link** — the frame the URDF
            mount sits in, i.e. :data:`~.frames.FLANGE`, not the fingertips.
        opening:      Normalised command in ``[0, 1]``.
        arm:          Unused today; both arms carry the same gripper. Accepted
            so a caller can pass it through and so a future left/right
            difference has somewhere to live.

    Returns:
        Two :class:`Jaw` entries, one per side of the opening axis.
    """
    del arm
    half = gap_for(opening) / 2.0
    centre = half + JAW_THICKNESS / 2.0

    dimensions = [0.0, 0.0, 0.0]
    dimensions[JAW_AXIS] = JAW_THICKNESS
    dimensions[1 - JAW_AXIS] = JAW_DEPTH
    dimensions[2] = JAW_HEIGHT

    out = []
    for sign in (-1.0, 1.0):
        offset = np.zeros(3, dtype=np.float32)
        offset[JAW_AXIS] = sign * centre
        # The jaws run from the fingertips back toward the mount.
        offset[2] = TIP_Z + JAW_HEIGHT / 2.0
        out.append(
            Jaw(
                pose=gripper_pose.translated_local(offset),
                dimensions=(dimensions[0], dimensions[1], dimensions[2]),
            )
        )
    return out
