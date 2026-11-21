"""Builds the cyano plate scene as a MuJoCo XML file.

The scene is regenerated for every run because the colonies, agar thickness
and tool are random or user-chosen. Layout on the bench (top view, robot at x = 0):

    tool rack   (0.32, 0.00)     source plate (0.45, -0.12)    source lid parking (0.63, -0.12)
    waste bin   (0.30, -0.25)    fresh plate  (0.45,  0.12)    fresh lid parking  (0.63,  0.12)

Sizes are for a standard 90 mm round Petri dish (about 89 mm across, 15 mm tall).
"""

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

BENCH_Z = 0.10

# Plate base (the bottom half of the dish)
PLATE_OUTER_R = 0.0445
PLATE_WALL_TOP = 0.0145  # height of the base wall above the bench
PLATE_FLOOR = 0.001
AGAR_R = 0.0432
AGAR_MIN = 0.0055  # agar thickness range in meters. "Up to half the plate height."
AGAR_MAX = 0.0072

# Lid
LID_INNER_R = 0.0465
LID_OUTER_R = 0.0475
LID_WALL_HEIGHT = 0.008
LID_TOP_THICKNESS = 0.0012

SOURCE_XY = (0.45, -0.12)
DEST_XY = (0.45, 0.12)
LID_PARK_XY = {"source": (0.63, -0.12), "dest": (0.63, 0.12)}
TOOL_RACK_XY = (0.32, 0.0)  # center of the 24-tool rack
RACK_ROWS = 3  # rows run along x, the direction the Panda fingers close
RACK_COLS = 8
RACK_PITCH_X = 0.03  # 3 cm apart where the fingers close (the fingers are thick that way)
RACK_PITCH_Y = 0.02  # 2 cm apart the other way
BIN_XY = (0.30, -0.25)

CAMERA_HEIGHT = 0.35  # overhead cameras sit 35 cm above the bench
CAMERA_FOVY = 20

TOOLS = {
    # Disposable plastic inoculation loop: flat handle, thin neck, small ring held flat to the agar.
    "loop": {
        "length": 0.12,
        "handle_half": 0.002,
        "tip_radius": 0.00125,  # ring about 2.5 mm across
        "rgba": "0.35 0.6 0.95 1",
        "rack_depth": 0.06,  # tall tool, so a deep rack keeps it from leaning into the fingers
    },
    # Wooden toothpick: round 2 mm shaft with a tapered point. The Panda fingertips cannot close tightly
    # enough on a bare 2 mm stick, so the top 25 mm sits in a 4 mm printed sleeve the gripper holds.
    "toothpick": {
        "length": 0.065,
        "handle_half": 0.002,
        "shaft_radius": 0.001,
        "sleeve_length": 0.025,
        "tip_radius": 0.0004,
        "rgba": "0.82 0.66 0.42 1",
        "rack_depth": 0.035,
    },
}


# Stiffer contact on tool handles so a squeezed handle does not sink deep into the fingers.
GRIP_CONTACT = 'solref="0.004 1" solimp="0.95 0.99 0.001"'


def quat_z(angle):
    return f"{math.cos(angle / 2):.6f} 0 0 {math.sin(angle / 2):.6f}"


def ring_walls(radius_mid, center_z, half_height, rgba, n_segments=24, extra=""):
    """A circular wall made of n flat boxes (MuJoCo has no hollow cylinder)."""
    half_len = radius_mid * math.tan(math.pi / n_segments) * 1.05
    lines = []
    for i in range(n_segments):
        a = 2 * math.pi * i / n_segments
        x = radius_mid * math.cos(a)
        y = radius_mid * math.sin(a)
        lines.append(
            f'<geom type="box" pos="{x:.5f} {y:.5f} {center_z:.5f}" quat="{quat_z(a)}" '
            f'size="0.0005 {half_len:.5f} {half_height:.5f}" rgba="{rgba}" {extra}/>'
        )
    return "\n        ".join(lines)


def sector_sticker_xml(dest_xy, agar_top):
    """8 pie-slice lines, like the sticker under the plate that divides it for 8 colonies."""
    lines = []
    for k in range(8):
        a = math.radians(45 * k)
        r = AGAR_R / 2
        x = dest_xy[0] + r * math.cos(a)
        y = dest_xy[1] + r * math.sin(a)
        lines.append(f'<geom name="sticker_line_{k}" type="box" pos="{x:.5f} {y:.5f} {agar_top + 0.00006:.5f}" '
                     f'quat="{quat_z(a)}" size="{r:.5f} 0.0003 0.00005" rgba="0.15 0.15 0.2 1" contype="0" conaffinity="0"/>')
    return "\n    ".join(lines)


def random_colonies(rng):
    """Visible dots (0.8 to 2 mm across), some scattered and some in clumps, like a real plate."""
    colonies = []
    n_scattered = int(rng.integers(8, 16))
    n_clumps = int(rng.integers(2, 5))
    for _ in range(n_scattered):
        r = 0.036 * math.sqrt(rng.random())
        a = 2 * math.pi * rng.random()
        colonies.append([r * math.cos(a), r * math.sin(a)])
    for _ in range(n_clumps):
        r = 0.03 * math.sqrt(rng.random())
        a = 2 * math.pi * rng.random()
        cx, cy = r * math.cos(a), r * math.sin(a)
        for _ in range(int(rng.integers(3, 7))):
            colonies.append([cx + rng.normal(0, 0.0025), cy + rng.normal(0, 0.0025)])
    result = []
    for x, y in colonies:
        if math.hypot(x, y) < 0.037:
            result.append({"x": x, "y": y, "r": float(rng.uniform(0.0004, 0.001))})
    return result


def plate_xml(name, xy, agar_thickness, colonies, rng):
    agar_top = BENCH_Z + PLATE_FLOOR + agar_thickness
    colony_lines = []
    for i, c in enumerate(colonies):
        green = rng.uniform(0.42, 0.6)
        colony_lines.append(
            f'<geom name="{name}_colony{i}" type="cylinder" pos="{c["x"]:.5f} {c["y"]:.5f} {agar_top + 0.0002:.5f}" '
            f'size="{c["r"]:.5f} 0.0002" rgba="{rng.uniform(0.12, 0.25):.2f} {green:.2f} {rng.uniform(0.08, 0.18):.2f} 1" '
            f'contype="0" conaffinity="0"/>'
        )
    return f"""
    <body name="plate_{name}" pos="{xy[0]} {xy[1]} 0">
        <geom name="{name}_floor" type="cylinder" pos="0 0 {BENCH_Z + PLATE_FLOOR / 2:.5f}" size="{PLATE_OUTER_R} {PLATE_FLOOR / 2}" rgba="0.92 0.93 0.97 0.5"/>
        <geom name="{name}_agar" type="cylinder" pos="0 0 {BENCH_Z + PLATE_FLOOR + agar_thickness / 2:.5f}" size="{AGAR_R} {agar_thickness / 2:.5f}"
              rgba="0.88 0.84 0.62 1" solref="0.03 1" friction="0.6 0.005 0.0001"/>
        {ring_walls((PLATE_OUTER_R - 0.0005), BENCH_Z + PLATE_WALL_TOP / 2, PLATE_WALL_TOP / 2, "0.92 0.93 0.97 0.5")}
        {"".join(chr(10) + "        " + line for line in colony_lines)}
    </body>
    <body name="lid_{name}" pos="{xy[0]} {xy[1]} {BENCH_Z + PLATE_WALL_TOP + 0.0001:.5f}">
        <freejoint/>
        <geom name="lid_{name}_top" type="cylinder" pos="0 0 {LID_TOP_THICKNESS / 2}" size="{LID_OUTER_R} {LID_TOP_THICKNESS / 2}" mass="0.006" rgba="0.92 0.93 0.97 0.35"/>
        {ring_walls((LID_INNER_R + LID_OUTER_R) / 2, -LID_WALL_HEIGHT / 2, LID_WALL_HEIGHT / 2, "0.92 0.93 0.97 0.35", extra='mass="0.0002"')}
    </body>
    <camera name="{name}_cam" pos="{xy[0]} {xy[1]} {BENCH_Z + CAMERA_HEIGHT}" xyaxes="1 0 0 0 1 0" fovy="{CAMERA_FOVY}"/>
"""


def rack_slot_xy(slot):
    """Center of rack slot 0..23. Rows run along x (away from the robot), columns along y."""
    row = slot // RACK_COLS
    col = slot % RACK_COLS
    x = TOOL_RACK_XY[0] + (row - (RACK_ROWS - 1) / 2) * RACK_PITCH_X
    y = TOOL_RACK_XY[1] + (col - (RACK_COLS - 1) / 2) * RACK_PITCH_Y
    return x, y


def one_tool_xml(tool_name, i):
    t = TOOLS[tool_name]
    L = t["length"]
    h = t["handle_half"]
    x, y = rack_slot_xy(i)
    if tool_name == "loop":
        geoms = f"""
        <geom name="tool{i}_handle" type="box" pos="0 0 {(0.012 + L) / 2:.4f}" size="{h} {h} {(L - 0.012) / 2:.4f}" mass="0.003" rgba="{t['rgba']}" friction="1.5 0.01 0.001" {GRIP_CONTACT}/>
        <geom name="tool{i}_neck" type="capsule" fromto="0 0 0.0006 0 0 0.012" size="0.0006" mass="0.0001" rgba="{t['rgba']}"/>
        <geom name="tool{i}_tip" type="cylinder" pos="0 0 0.0003" size="{t['tip_radius']} 0.0003" mass="0.0001" rgba="{t['rgba']}" friction="0.5 0.005 0.0001"/>"""
    else:
        geoms = f"""
        <geom name="tool{i}_handle" type="box" pos="0 0 {L - t['sleeve_length'] / 2:.4f}" size="{h} {h} {t['sleeve_length'] / 2:.4f}" mass="0.003" rgba="0.3 0.3 0.3 1" friction="1.5 0.01 0.001" {GRIP_CONTACT}/>
        <geom name="tool{i}_shaft" type="capsule" fromto="0 0 0.006 0 0 {L - t['sleeve_length']:.4f}" size="{t['shaft_radius']}" mass="0.0002" rgba="{t['rgba']}"/>
        <geom name="tool{i}_taper" type="capsule" fromto="0 0 0.0004 0 0 0.006" size="0.0007" mass="0.0001" rgba="{t['rgba']}"/>
        <geom name="tool{i}_tip" type="sphere" pos="0 0 {t['tip_radius']}" size="{t['tip_radius']}" mass="0.0001" rgba="{t['rgba']}" friction="0.7 0.005 0.0001"/>"""
    return f"""
    <body name="tool_{i}" pos="{x:.4f} {y:.4f} {BENCH_Z + 0.0002}">
        <freejoint/>
        <site name="tool_tip_site_{i}" pos="0 0 0" size="0.0005"/>
        {geoms}
    </body>"""


def tool_xml(tool_name):
    """A rack of 24 tools (3 x 8), like a pipette tip box. One tool per hole."""
    h = TOOLS[tool_name]["handle_half"]
    hole = h + 0.0005  # 0.5 mm of play around the handle
    wall_x = RACK_PITCH_X - 2 * hole  # solid material between neighboring holes
    wall_y = RACK_PITCH_Y - 2 * hole
    height = TOOLS[tool_name]["rack_depth"]
    cx, cy = TOOL_RACK_XY
    half_x = RACK_ROWS * RACK_PITCH_X / 2
    half_y = RACK_COLS * RACK_PITCH_Y / 2
    walls = []
    for r in range(RACK_ROWS + 1):  # walls running along y
        x = cx - half_x + r * RACK_PITCH_X
        walls.append(f'<geom type="box" pos="{x:.4f} {cy:.4f} {height / 2}" size="{wall_x / 2:.4f} {half_y + wall_y / 2:.4f} {height / 2}" rgba="0.3 0.3 0.35 1"/>')
    for c in range(RACK_COLS + 1):  # walls running along x
        y = cy - half_y + c * RACK_PITCH_Y
        walls.append(f'<geom type="box" pos="{cx:.4f} {y:.4f} {height / 2}" size="{half_x + wall_x / 2:.4f} {wall_y / 2:.4f} {height / 2}" rgba="0.3 0.3 0.35 1"/>')
    rack = f"""
    <body name="tool_rack" pos="0 0 {BENCH_Z}">
        {chr(10).join("        " + w for w in walls)}
    </body>"""
    tools = "".join(one_tool_xml(tool_name, i) for i in range(RACK_ROWS * RACK_COLS))
    return rack + tools


def bin_xml():
    bx, by = BIN_XY
    return f"""
    <body name="waste_bin" pos="{bx} {by} {BENCH_Z}">
        <geom type="box" pos="0.032 0 0.025" size="0.002 0.034 0.025" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="-0.032 0 0.025" size="0.002 0.034 0.025" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="0 0.032 0.025" size="0.03 0.002 0.025" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="0 -0.032 0.025" size="0.03 0.002 0.025" rgba="0.75 0.2 0.2 1"/>
    </body>"""


PCR_ORIGIN = (0.555, -0.0315)  # first PCR tube; the 8-tube strip runs along +y, between the plates and the parked lids


def build_plate_scene(tool_name, seed, pcr=False, sectors=False, source_colonies=None):
    """Write the scene XML and return (path, truth). truth holds the hidden facts we grade against."""
    rng = np.random.default_rng(seed)
    colonies = random_colonies(rng)
    if source_colonies is not None:  # a plate grown from an earlier round (positions relative to the plate center)
        colonies = source_colonies
    agar_source = float(rng.uniform(AGAR_MIN, AGAR_MAX))
    agar_dest = float(rng.uniform(AGAR_MIN, AGAR_MAX))

    TOOL_LOCKS = "\n    ".join(
        f'<weld name="tool_lock_{i}" body1="hand" body2="tool_{i}" active="false" solref="0.002 1" solimp="0.99 0.999 0.0001"/>'
        for i in range(RACK_ROWS * RACK_COLS))
    xml = f"""<mujoco model="cyano plates">
  <include file="panda_lab.xml"/>
  <option timestep="0.001"/>
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.88 0.92" rgb2="0.55 0.6 0.65" width="512" height="3072"/>
    <texture type="2d" name="floor_tex" builtin="checker" rgb1="0.75 0.75 0.75" rgb2="0.68 0.68 0.68" width="300" height="300"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0.45 0 2" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <light pos="1.5 -1 1.5" dir="-1 0.7 -1" diffuse="0.25 0.25 0.25"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor_mat"/>
    <camera name="work_cam" pos="1.0 0 0.55" xyaxes="0 1 0 -0.633 0 0.774" fovy="45"/>
    <camera name="close_cam" pos="0.85 0 0.42" xyaxes="0 1 0 -0.63 0 0.777" fovy="38"/>
    <body name="bench" pos="0.5 0 0.05">
      <geom name="bench_top" type="box" size="0.25 0.35 0.05" rgba="0.25 0.27 0.3 1"/>
    </body>
    {plate_xml("source", SOURCE_XY, agar_source, colonies, rng)}
    {plate_xml("dest", DEST_XY, agar_dest, [], rng)}
    {sector_sticker_xml(DEST_XY, BENCH_Z + PLATE_FLOOR + agar_dest) if sectors == True else ''}
    {tool_xml(tool_name)}
    {bin_xml()}
    {__import__('pcr').pcr_strip_xml(PCR_ORIGIN, 'y') if pcr == True else ''}
  </worldbody>
  <equality>
    <!-- Tool adapter: switched on when the gripper closes on the tool, so the tool cannot slip in the fingers. -->
    {TOOL_LOCKS}
  </equality>
  <actuator>
    <!-- Lid lifting is modeled as a small suction pad at the fingertips (MuJoCo adhesion).
         A 90 mm lid is wider than the Panda gripper opens (80 mm), so real systems use suction too. -->
    <adhesion name="suction_left" body="left_finger" ctrlrange="0 1" gain="2"/>
    <adhesion name="suction_right" body="right_finger" ctrlrange="0 1" gain="2"/>
  </actuator>
</mujoco>
"""
    # One file per process, so several sims can run at the same time without overwriting each other.
    path = os.path.join(HERE, "models", f"_generated_plate_scene_{os.getpid()}.xml")
    with open(path, "w") as f:
        f.write(xml)

    truth = {
        "tool": tool_name,
        "seed": seed,
        "colonies": [{"x": SOURCE_XY[0] + c["x"], "y": SOURCE_XY[1] + c["y"], "r": c["r"]} for c in colonies],
        "agar_top": {
            "source": BENCH_Z + PLATE_FLOOR + agar_source,
            "dest": BENCH_Z + PLATE_FLOOR + agar_dest,
        },
    }
    return path, truth
