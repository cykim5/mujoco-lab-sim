"""Printable parts for the SO-101 build, with dimensions taken from the simulated parts.

Run:
  ~/.venvs/mujoco-lab/bin/python make_print_parts.py

Writes STL files (millimeters) to print/:
  toothpick_rack.stl    2 rows of 12 through-holes, 7 mm pitch, rows 25 mm apart, 25 mm tall
  waste_cup.stl         square cup, 48 mm inside, 80 mm tall
  stripper_lid.stl      lid for the waste cup: plate over half the opening with a slot for stripping tips
  slot_test.stl         test card with slot widths 4.0 to 7.0 mm, to choose the stripper slot for your tips

Print settings: PLA or PETG, 0.2 mm layers, 3 walls, no supports.
  rack: print upright. cup: print upright. lid and test card: print plate side down.
"""

import os

import numpy as np
from manifold3d import Manifold

import so101_world as sw

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "print")
SEGMENTS = 48  # facets per circle

# Printed holes come out about 0.2 mm smaller than drawn, so holes get extra clearance.
HOLE_CLEARANCE = 0.2

# Rack (from so101_world.py)
RACK_PITCH = sw.RACK_PITCH * 1000  # 7 mm between holes along a row
RACK_ROW_GAP = (sw.RACK_ROW_X[1] - sw.RACK_ROW_X[0]) * 1000  # 25 mm between rows
RACK_HEIGHT = sw.RACK_DEPTH * 1000  # 25 mm tall; toothpicks stand on the bench through the holes
RACK_HOLE_D = (sw.TOOTHPICK_RADIUS * 2 + 0.001) * 1000 + HOLE_CLEARANCE  # 3 mm hole for a 2 mm toothpick, plus clearance
RACK_HOLES_PER_ROW = 12
RACK_MARGIN = 5.5  # wall from the hole centers to the rack edge

# Waste cup (from so101_world.bin_xml)
CUP_INNER = 48.0
CUP_WALL = 2.4
CUP_HEIGHT = 80.0
CUP_FLOOR = 2.0

# Stripper lid (from liquid.py: 3 mm plate over half the cup, 5 mm slot, slot ends 20 mm from the cup center)
LID_PLATE = 3.0
SLOT_WIDTH = 5.0  # set this from the slot test card: the tip body must pass, the tip collar must not
SLOT_LENGTH = 20.0
SKIRT_HEIGHT = 8.0
SKIRT_WALL = 2.0
SKIRT_CLEARANCE = 0.4  # gap between the skirt and the cup wall, so the lid drops on easily

# Slot test card
TEST_WIDTHS = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]


def box(x0, y0, z0, x1, y1, z1):
    """Solid box between two corners."""
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def toothpick_rack():
    length = (RACK_HOLES_PER_ROW - 1) * RACK_PITCH + 2 * RACK_MARGIN
    width = RACK_ROW_GAP + 2 * RACK_MARGIN
    rack = box(0, 0, 0, length, width, RACK_HEIGHT)
    for row in range(2):
        y = RACK_MARGIN + row * RACK_ROW_GAP
        for col in range(RACK_HOLES_PER_ROW):
            x = RACK_MARGIN + col * RACK_PITCH
            hole = Manifold.cylinder(RACK_HEIGHT + 2, RACK_HOLE_D / 2, circular_segments=SEGMENTS).translate([x, y, -1])
            rack = rack - hole
    return rack


def waste_cup():
    outer = CUP_INNER + 2 * CUP_WALL
    cup = box(0, 0, 0, outer, outer, CUP_HEIGHT)
    cup = cup - box(CUP_WALL, CUP_WALL, CUP_FLOOR, CUP_WALL + CUP_INNER, CUP_WALL + CUP_INNER, CUP_HEIGHT + 1)
    return cup


def stripper_lid():
    """Printed plate side down. Once flipped onto the cup, the plate rests on the cup rim and the skirt hangs outside."""
    cup_outer = CUP_INNER + 2 * CUP_WALL
    inside = cup_outer + 2 * SKIRT_CLEARANCE  # skirt inner size
    outer = inside + 2 * SKIRT_WALL
    center = outer / 2
    # full plate, then cut away the open half (leaving a frame over the cup rim)
    lid = box(0, 0, 0, outer, outer, LID_PLATE)
    rim = SKIRT_WALL + SKIRT_CLEARANCE + CUP_WALL  # frame width, from the lid edge to the cup's inner wall
    lid = lid - box(rim, center, -1, outer - rim, outer - rim, LID_PLATE + 1)
    # slot: opens into the open half and runs SLOT_LENGTH into the plate
    lid = lid - box(center - SLOT_WIDTH / 2, center - SLOT_LENGTH, -1, center + SLOT_WIDTH / 2, center + 1, LID_PLATE + 1)
    # skirt around the outside of the cup
    skirt = box(0, 0, 0, outer, outer, LID_PLATE + SKIRT_HEIGHT)
    skirt = skirt - box(SKIRT_WALL, SKIRT_WALL, -1, outer - SKIRT_WALL, outer - SKIRT_WALL, LID_PLATE + SKIRT_HEIGHT + 1)
    return lid + skirt


def slot_test():
    """One slot per width, narrowest on the left. The cut corner marks the left end."""
    spacing = 12.0
    length = spacing * (len(TEST_WIDTHS) + 1)
    card = box(0, 0, 0, length, 30, LID_PLATE)
    for i, w in enumerate(TEST_WIDTHS):
        x = spacing * (i + 1)
        card = card - box(x - w / 2, 10, -1, x + w / 2, 31, LID_PLATE + 1)
    corner = Manifold.cube([8, 8, LID_PLATE + 2]).rotate([0, 0, 45]).translate([0, -5.7, -1])
    return card - corner


def save_stl(part, name):
    """Write a binary STL file."""
    mesh = part.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    tris = np.array(mesh.tri_verts)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths
    record = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    record["n"] = normals
    record["v"] = np.stack([a, b, c], axis=1)
    path = os.path.join(OUT_DIR, name)
    with open(path, "wb") as f:
        f.write(b"mujoco-lab-sim printed part".ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(record.tobytes())
    size = verts.max(axis=0) - verts.min(axis=0)
    print(f"{name}: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm, {len(tris)} triangles, volume {part.volume() / 1000:.1f} cm3")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    save_stl(toothpick_rack(), "toothpick_rack.stl")
    save_stl(waste_cup(), "waste_cup.stl")
    save_stl(stripper_lid(), "stripper_lid.stl")
    save_stl(slot_test(), "slot_test.stl")


if __name__ == "__main__":
    main()
