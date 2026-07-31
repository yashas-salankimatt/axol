"""Build a viewer URDF with the real gripper: articulated jaws from Almond's CAD.

Why this exists
---------------

The bundled ``axol.urdf`` is a **1,084-triangle** model. ``Base.stl`` and
``S1.stl`` — the whole torso — are 12 faces each, i.e. plain boxes, and each
gripper is a 20-face box welded on by a ``fixed`` joint. Those are Almond's
shipped stand-ins, not something this repo introduced, and no higher-fidelity
robot model is published: ``gripper.step`` is the only real geometry on
almond.bot (``axol.step``, ``axol.stl``, a full URDF — all 404).

So the gripper is the one part that *can* be made real, and this script does
it: it downloads Almond's 8 MB STEP, splits it into a body and two jaws, and
writes a second URDF that is the bundled one with the gripper boxes replaced by
that geometry plus a **prismatic joint per jaw**.

Why a separate URDF
-------------------

``axol.urdf`` is the *control* model. It feeds the pyroki IK solver, the
self-collision model, and the MuJoCo gravity compensator, and it is what teleop
and the waypoint planner run against. Adding four jaw joints to it would take
the actuated count from 14 to 18 and perturb all of that for a purely visual
gain. Splitting visual from collision/control geometry is standard practice,
so the viewer loads ``axol_visual.urdf`` and the solver keeps ``axol.urdf``.

Frame mapping
-------------

Measured from the CAD: the wrist mounts at the motor end (the assembly's
minimum Z) and the jaws point away from it. The URDF gripper link has ``Z = 0``
at the mount and the fingertips at ``Z = -0.145``. So CAD maps to link by a
180-degree rotation about X plus a shift::

    link_x =  cad_x
    link_y = -cad_y
    link_z = -(cad_z - cad_z_at_mount)

which puts the fingertips at ``z = -0.1466`` — within 1.5 mm of the placeholder
box's 145 mm, an independent check that the mapping is right.

Usage::

    uv run python tools/build_visual_urdf.py
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

STEP_URL = "https://www.almond.bot/resources/gripper.step"

URDF_DIR = Path(__file__).resolve().parent.parent / "almond_axol" / "kinematics" / "urdf"
SOURCE_URDF = URDF_DIR / "axol.urdf"
OUTPUT_URDF = URDF_DIR / "axol_visual.urdf"
MESH_DIR = URDF_DIR / "meshes"

JAW_PARTS = ("Part 12", "Part 12_1")
"""The two jaws in the STEP assembly, at CAD x = -48 mm and +48 mm."""

TESSELLATION = 0.6
"""Linear tolerance (mm) for the STEP conversion. The 0.05 mm pass produces
180k triangles, which is far more than a browser scene wants."""

HULL_ABOVE = 2000
"""Replace any body part heavier than this with its convex hull. The motor's
internal detail — windings, fasteners — is invisible once assembled and costs
most of the triangle budget. Jaws are never hulled: their faces are the
gripping surfaces."""

OPEN_INNER_HALF_GAP = 0.036155
"""Half the fully-open gap (m). Each jaw's closed position is this far in."""


def download(cache: Path) -> Path:
    """Fetch the STEP once, caching next to the outputs."""
    if cache.exists() and cache.stat().st_size > 1_000_000:
        print(f"  using cached {cache.name} ({cache.stat().st_size / 1e6:.1f} MB)")
        return cache
    print(f"  downloading {STEP_URL} ...")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(STEP_URL, timeout=300) as response, cache.open("wb") as out:
        shutil.copyfileobj(response, out)
    print(f"  got {cache.stat().st_size / 1e6:.1f} MB")
    return cache


def load_assembly(step: Path, work: Path) -> trimesh.Scene:
    """Convert the STEP to a scene, tessellated for a viewer."""
    import cascadio

    glb = work / "gripper.glb"
    if not glb.exists():
        cascadio.step_to_glb(str(step), str(glb), tol_linear=TESSELLATION, tol_angular=0.6)
    return trimesh.load(glb)


def to_link_frame(vertices: np.ndarray, mount_z: float) -> np.ndarray:
    """CAD coordinates -> gripper link frame. See the module docstring."""
    out = np.empty_like(vertices)
    out[:, 0] = vertices[:, 0]
    out[:, 1] = -vertices[:, 1]
    out[:, 2] = -(vertices[:, 2] - mount_z)
    return out


def build_meshes(scene: trimesh.Scene) -> dict[str, trimesh.Trimesh]:
    """Split the assembly into a body and two jaws, in the gripper link frame."""
    placed = {}
    for name, geom in scene.geometry.items():
        transform = scene.graph.get(name)[0]
        placed[name] = (geom.vertices @ transform[:3, :3].T + transform[:3, 3], geom.faces)

    mount_z = min(v[:, 2].min() for v, _ in placed.values())
    print(f"  mount face at CAD z = {mount_z * 1000:.2f} mm")

    body_parts, out, sides = [], {}, {}
    for name, (verts, faces) in placed.items():
        mesh = trimesh.Trimesh(to_link_frame(verts, mount_z), faces, process=False)
        if name in JAW_PARTS:
            # Which side of the tool axis this jaw lives on. This is the single
            # source of truth for both where the mesh gets parked and which way
            # its joint travels — deriving the two independently is how they
            # came to disagree, giving jaws that crossed through each other
            # instead of opening.
            side_sign = float(np.sign(mesh.vertices[:, 0].mean()))
            # Park it closed: slide inward until the gripping face is on the
            # tool axis, so joint value 0 means shut.
            mesh.vertices[:, 0] -= side_sign * OPEN_INNER_HALF_GAP
            tag = "A" if side_sign < 0 else "B"
            out[f"Jaw_{tag}"] = mesh
            sides[tag] = side_sign
        else:
            body_parts.append(
                mesh.convex_hull if len(mesh.faces) > HULL_ABOVE else mesh
            )
    out["Gripper_Body"] = trimesh.util.concatenate(body_parts)
    for tag, sign in sorted(sides.items()):
        print(f"  Jaw_{tag} sits on {'-X' if sign < 0 else '+X'}, opens toward {'-X' if sign < 0 else '+X'}")
    return out, sides


def gripper_link(root: ET.Element, side: str) -> ET.Element:
    for link in root.findall("link"):
        if link.get("name") == f"{side}_gripper":
            return link
    raise SystemExit(f"{side}_gripper link not found in {SOURCE_URDF}")


def mesh_geometry(filename: str) -> ET.Element:
    geometry = ET.Element("geometry")
    ET.SubElement(geometry, "mesh", {"filename": f"package://assembly/meshes/{filename}", "scale": "1 1 1"})
    return geometry


def retarget(link: ET.Element, filename: str) -> None:
    """Point a link's visual/collision at a new mesh, at the link origin.

    The bundled meshes carry a large ``<origin>`` offset because their vertices
    are in assembly coordinates. The generated meshes are already in link
    coordinates, so the offset has to go.
    """
    for tag in ("visual", "collision"):
        for node in link.findall(tag):
            link.remove(node)
        node = ET.SubElement(link, tag)
        ET.SubElement(node, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        node.append(mesh_geometry(filename))


def add_jaw(root: ET.Element, side: str, tag: str, side_sign: float) -> None:
    """Add one jaw link and the prismatic joint that slides it.

    ``side_sign`` is which side of the tool axis the jaw's geometry sits on,
    and is therefore also the direction it has to travel to *open*. Passing the
    same number for both keeps them from disagreeing.
    """
    link = ET.SubElement(root, "link", {"name": f"{side}_jaw_{tag.lower()}"})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 -0.105", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.05"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "1e-5", "ixy": "0", "ixz": "0", "iyy": "1e-5", "iyz": "0", "izz": "1e-5"},
    )
    for element in ("visual", "collision"):
        node = ET.SubElement(link, element)
        ET.SubElement(node, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        node.append(mesh_geometry(f"Jaw_{tag}.stl"))

    joint = ET.SubElement(root, "joint", {"name": f"{side}_jaw_{tag.lower()}_0", "type": "prismatic"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": f"{side}_gripper"})
    ET.SubElement(joint, "child", {"link": f"{side}_jaw_{tag.lower()}"})
    ET.SubElement(joint, "axis", {"xyz": f"{side_sign:+.0f} 0 0"})
    ET.SubElement(
        joint,
        "limit",
        {"lower": "0", "upper": f"{OPEN_INNER_HALF_GAP:.6f}", "effort": "50", "velocity": "0.1"},
    )


def main() -> None:
    work = Path.home() / ".almond" / "cad"
    work.mkdir(parents=True, exist_ok=True)

    print("Gripper CAD")
    step = download(work / "gripper.step")
    scene = load_assembly(step, work)

    print("Meshes")
    meshes, sides = build_meshes(scene)
    for name, mesh in sorted(meshes.items()):
        path = MESH_DIR / f"{name}.stl"
        mesh.export(path)
        print(f"  {path.name:20} {len(mesh.faces):6d} faces  {path.stat().st_size / 1024:7.1f} kB")

    print("URDF")
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    for side in ("left", "right"):
        retarget(gripper_link(root, side), "Gripper_Body.stl")
        for tag, side_sign in sorted(sides.items()):
            add_jaw(root, side, tag, side_sign)
    ET.indent(tree, space="    ")
    tree.write(OUTPUT_URDF, encoding="unicode", xml_declaration=True)

    actuated = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    print(f"  wrote {OUTPUT_URDF.name}: {len(actuated)} actuated joints")
    print(f"  added: {[n for n in actuated if 'jaw' in n]}")


if __name__ == "__main__":
    sys.exit(main())
