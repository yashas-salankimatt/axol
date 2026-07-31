"""The display model: real gripper geometry with articulated jaws.

Guards the two properties that make it safe to load — it must not disturb the
control model, and its jaws must actually move — plus the frame mapping used to
bring Almond's CAD into the gripper link frame.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import yourdfpy

from almond_axol.constants import URDF_PATH, URDF_VISUAL_PATH, visual_urdf_path
from almond_axol.motion.gripper import MAX_TRAVEL, OPEN_GAP

pytestmark = pytest.mark.skipif(
    not URDF_VISUAL_PATH.exists(),
    reason="run tools/build_visual_urdf.py to generate the display model",
)


# Every jaw test runs against BOTH grippers. They were left-only, so the
# inverted-axis bug that shipped once could have been reintroduced on the right
# gripper alone and the whole suite would still have passed.
@pytest.fixture(params=["left", "right"])
def side(request):
    return request.param


@pytest.fixture(scope="module")
def visual():
    return yourdfpy.URDF.load(
        str(URDF_VISUAL_PATH), mesh_dir=str(URDF_VISUAL_PATH.parent)
    )


@pytest.fixture(scope="module")
def control():
    return yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))


def test_visual_path_is_preferred_when_present():
    assert visual_urdf_path() == URDF_VISUAL_PATH


def test_control_model_is_untouched(control):
    """The solver's model must keep exactly 14 actuated joints."""
    assert len(control.actuated_joint_names) == 14
    assert not any("jaw" in n for n in control.actuated_joint_names)


def test_display_model_adds_four_jaw_joints(visual):
    assert len(visual.actuated_joint_names) == 18
    jaws = [n for n in visual.actuated_joint_names if "jaw" in n]
    assert sorted(jaws) == [
        "left_jaw_a_0",
        "left_jaw_b_0",
        "right_jaw_a_0",
        "right_jaw_b_0",
    ]


def test_arm_kinematics_are_identical(visual, control):
    """Only the gripper differs — every arm joint must be byte-identical."""
    for name in control.actuated_joint_names:
        a, b = control.joint_map[name], visual.joint_map[name]
        assert a.type == b.type
        assert a.parent == b.parent and a.child == b.child
        assert np.allclose(a.origin, b.origin, atol=1e-12)
        assert np.allclose(a.axis, b.axis, atol=1e-12)


def test_jaw_joints_are_prismatic_with_the_measured_travel(visual):
    for name in (n for n in visual.actuated_joint_names if "jaw" in n):
        joint = visual.joint_map[name]
        assert joint.type == "prismatic"
        assert joint.limit.lower == pytest.approx(0.0)
        assert joint.limit.upper == pytest.approx(MAX_TRAVEL, abs=1e-5)


def test_jaws_of_one_gripper_oppose_each_other(visual):
    a = visual.joint_map["left_jaw_a_0"].axis
    b = visual.joint_map["left_jaw_b_0"].axis
    assert np.allclose(a, -np.asarray(b), atol=1e-12)


def cfg_at(visual, opening: float) -> dict[str, float]:
    return {
        n: (opening * MAX_TRAVEL if "jaw" in n else 0.0)
        for n in visual.actuated_joint_names
    }


def jaw_span(visual, tag: str, side: str = "left") -> tuple[float, float]:
    """A jaw's extent along the opening axis, **in the gripper link frame**.

    Measured from the mesh, not from the joint origin. The origins alone cannot
    tell an opening gripper from one whose jaws pass through each other and
    swap sides — the distance between them is identical either way, which is
    exactly how an inverted axis shipped.
    """
    import trimesh

    scene = visual.scene
    mesh = trimesh.load(URDF_VISUAL_PATH.parent / "meshes" / f"Jaw_{tag.upper()}.stl")
    to_world = scene.graph.get(f"{side}_jaw_{tag.lower()}")[0]
    world = mesh.vertices @ to_world[:3, :3].T + to_world[:3, 3]
    link = scene.graph.get(f"{side}_gripper")[0]
    local = (world - link[:3, 3]) @ link[:3, :3]
    return float(local[:, 0].min()), float(local[:, 0].max())


def test_closed_brings_both_gripping_faces_onto_the_tool_axis(visual, side):
    visual.update_cfg(cfg_at(visual, 0.0))
    _, a_hi = jaw_span(visual, "a", side)
    b_lo, _ = jaw_span(visual, "b", side)
    assert a_hi == pytest.approx(0.0, abs=5e-4), "jaw A should close onto the axis"
    assert b_lo == pytest.approx(0.0, abs=5e-4), "jaw B should close onto the axis"


@pytest.mark.parametrize("opening", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_jaws_never_cross_the_centreline(visual, side, opening):
    """The regression: the jaws opened *through* each other and swapped sides.

    Each jaw must stay wholly on its own side of the tool axis at every
    opening, which is a statement the joint origins cannot make.
    """
    visual.update_cfg(cfg_at(visual, opening))
    _, a_hi = jaw_span(visual, "a", side)
    b_lo, _ = jaw_span(visual, "b", side)
    assert a_hi <= 5e-4, f"jaw A crossed to +X at opening {opening}"
    assert b_lo >= -5e-4, f"jaw B crossed to -X at opening {opening}"


@pytest.mark.parametrize("opening", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_gripping_faces_are_the_commanded_gap_apart(visual, side, opening):
    visual.update_cfg(cfg_at(visual, opening))
    _, a_hi = jaw_span(visual, "a", side)
    b_lo, _ = jaw_span(visual, "b", side)
    assert b_lo - a_hi == pytest.approx(opening * OPEN_GAP, abs=1e-3)


def test_opening_moves_the_jaws_apart_monotonically(visual, side):
    """Increasing the command must increase the gap, at every step."""
    gaps = []
    for opening in np.linspace(0.0, 1.0, 9):
        visual.update_cfg(cfg_at(visual, float(opening)))
        _, a_hi = jaw_span(visual, "a", side)
        b_lo, _ = jaw_span(visual, "b", side)
        gaps.append(b_lo - a_hi)
    assert all(later > earlier for earlier, later in pairwise(gaps)), gaps
    assert gaps[-1] == pytest.approx(OPEN_GAP, abs=1e-3)


def test_each_jaw_travels_outward_from_the_axis(visual, side):
    """Each jaw individually moves *away* from the centreline as it opens."""
    visual.update_cfg(cfg_at(visual, 0.0))
    closed_a, closed_b = jaw_span(visual, "a", side)[0], jaw_span(visual, "b", side)[1]
    visual.update_cfg(cfg_at(visual, 1.0))
    open_a, open_b = jaw_span(visual, "a", side)[0], jaw_span(visual, "b", side)[1]
    assert open_a < closed_a, "jaw A must travel toward -X to open"
    assert open_b > closed_b, "jaw B must travel toward +X to open"


def test_jaw_axes_point_outward(visual, side):
    """The joint axes themselves must face away from the tool axis."""
    for tag, expected in (("a", -1.0), ("b", 1.0)):
        axis = np.asarray(visual.joint_map[f"{side}_jaw_{tag}_0"].axis, dtype=float)
        assert axis[0] == pytest.approx(expected)
        assert np.allclose(axis[1:], 0.0)


def test_geometry_is_real_not_a_placeholder(visual, control):
    """The gripper boxes were 20 faces each; the CAD is three orders bigger."""
    total = sum(len(g.faces) for g in visual.scene.geometry.values())
    base = sum(len(g.faces) for g in control.scene.geometry.values())
    assert base == 1084, "control model is Almond's decimated stand-in"
    assert total > 15000


def test_fingertips_land_near_the_documented_tool_offset(visual):
    """An independent check on the CAD -> gripper link frame mapping.

    ``GRIPPER_TIP_OFFSET`` puts the fingertips 145 mm along the link's -Z. The
    CAD, positioned purely from its own mount datum, has to land there too —
    nothing in the conversion was fitted to make that happen.
    """
    import trimesh

    visual.update_cfg(cfg_at(visual, 1.0))
    scene = visual.scene
    jaw = trimesh.load(URDF_VISUAL_PATH.parent / "meshes" / "Jaw_A.stl")
    to_world = scene.graph.get("left_jaw_a")[0]
    tips = (jaw.vertices @ to_world[:3, :3].T + to_world[:3, 3])[:, 2].min()
    gripper_z = scene.graph.get("left_gripper")[0][2, 3]
    assert gripper_z - tips == pytest.approx(0.145, abs=0.003)
