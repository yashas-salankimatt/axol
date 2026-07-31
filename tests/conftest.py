"""Shared fixtures.

The Cartesian kinematics fixture is session-scoped on purpose: constructing one
loads the URDF, builds the pyroki collision model, and triggers a JAX compile
that costs the better part of ten seconds. Per-test construction would make the
suite unusable.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def kin():
    """A shared :class:`CartesianKinematics` (expensive to build)."""
    from almond_axol.motion import CartesianKinematics

    return CartesianKinematics()


@pytest.fixture(scope="session")
def q_rest(kin):
    """The rest-pose joint vector."""
    return kin.rest_q()


@pytest.fixture(scope="session")
def q_ready(kin):
    """A bent, well-conditioned configuration.

    Cartesian tests target poses around *this*, not the rest pose: rest is 88%
    extended, so a 12 cm move from it runs into the reach clamp and the
    conditioning of a nearly-straight arm rather than testing the code.
    """
    return kin.ready_q()


@pytest.fixture(scope="session")
def q_penetrating(kin):
    """A configuration whose arms are genuinely inside the torso.

    Built by swinging ``shoulder_2`` inward from the ready pose until the
    collision model reports penetration (measured -25.8 mm). Deterministic, so
    the collision gate can be tested against a known-bad trajectory instead of
    a search that may find nothing.
    """
    import numpy as np

    from almond_axol.motion import ARMS, Arm

    q = kin.ready_q().copy()
    for arm in ARMS:
        index = kin.indices(arm)
        sign = 1.0 if arm is Arm.LEFT else -1.0
        q[index[1]] = 0.1 * sign
    assert kin.self_collision_clearance(q) < -0.02, "fixture no longer penetrates"
    return np.asarray(q, dtype=np.float32)
