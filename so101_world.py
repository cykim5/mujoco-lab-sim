"""Builds the SO-101 ("Build" version) bench as a MuJoCo XML file.

The SO-101 is small, so everything sits 12 to 27 cm from its base (top view, robot at the origin):

    lid parking (-0.02, -0.20)                       lid parking (-0.02, 0.20)
              source plate (0.142, -0.111)    fresh plate (0.142, 0.111)
                      toothpick cup (0.20, 0.0), grab tab pointing at the robot
              waste cup (0.10, -0.225), beside the source plate
    calibration pins at (0.08, +-0.17) short and (0.25, +-0.08) tall

Toothpicks stand jumbled together in a cup with a printed grab tab. When no toothpick is free to pick,
the robot pinches the tab and gently shakes the cup until one separates. A low ring on the bench
keeps the cup in place.
"""

import math
import os

import numpy as np

import plate_world as pw

HERE = os.path.dirname(os.path.abspath(__file__))

BENCH_Z = pw.BENCH_Z
RISER_HEIGHT = 0.08

SOURCE_XY = (0.142, -0.111)
DEST_XY = (0.142, 0.111)
LID_PARK_XY = {"source": (-0.02, -0.20), "dest": (-0.02, 0.20)}
CUP_XY = (0.20, 0.0)
BIN_XY = (0.10, -0.225)  # tall waste cup beside the source plate, kept clear of where the lid hook hangs while dipping

N_TOOLS = 24
CUP_INNER_R = 0.016  # a narrow 3.2 cm holder: toothpicks lean on the rim and their tops fan out
CUP_DEPTH = 0.045  # deep enough that leaning toothpicks do not tip out over the rim
CUP_FLOOR = 0.003
CUP_FLOOR_TOP = BENCH_Z + CUP_FLOOR
CUP_RIM_Z = CUP_FLOOR_TOP + CUP_DEPTH - 0.0015
TAB_OUTER = 0.058  # the grab tab reaches 5.8 cm from the cup center, past where the toothpick tops fan out
TAB_GRIP_R = 0.052  # where the fingertips pinch the tab
TAB_HALF_THICK = 0.0015
TAB_HEIGHT = 0.03
NEST_CLEARANCE = 0.002  # the ring on the bench leaves the cup 2 mm of play

# Calibration pins: small posts at exactly known spots. The robot touches them to measure its own errors.
PIN_RADIUS = 0.004
# (x, y, height): two short pins near the robot and two tall ones far out, so touches happen at
# different reaches and heights. That lets the robot tell its shoulder, elbow and wrist errors apart.
CAL_PINS = [(0.08, -0.17, 0.016), (0.08, 0.17, 0.016), (0.25, -0.08, 0.05), (0.25, 0.08, 0.05)]

TOOTHPICK_LENGTH = 0.065
TOOTHPICK_RADIUS = 0.001
TOOTHPICK_TIP_RADIUS = 0.0004
TOOTHPICK_RGBA = "0.82 0.66 0.42 1"
CUP_CAMERA_HEIGHT = 0.30
CUP_CAMERA_FOVY = 16


def toothpick_xml(i, x, y, tilt_x, tilt_y):
    L = TOOTHPICK_LENGTH
    # Start almost upright, leaning a little; the cup wall and neighbors hold them up.
    qw = math.cos(math.hypot(tilt_x, tilt_y) / 2)
    norm = max(math.hypot(tilt_x, tilt_y), 1e-9)
    s = math.sin(math.hypot(tilt_x, tilt_y) / 2)
    quat = f"{qw:.6f} {s * tilt_x / norm:.6f} {s * tilt_y / norm:.6f} 0"
    return f"""
    <body name="tool_{i}" pos="{x:.5f} {y:.5f} {CUP_FLOOR_TOP + 0.0005:.5f}" quat="{quat}">
        <freejoint/>
        <site name="tool_tip_site_{i}" pos="0 0 0" size="0.0005"/>
        <geom name="tool{i}_shaft" type="capsule" fromto="0 0 {TOOTHPICK_RADIUS + 0.0035:.4f} 0 0 {L - TOOTHPICK_RADIUS - 0.0035:.4f}"
              size="{TOOTHPICK_RADIUS}" mass="0.0008" rgba="{TOOTHPICK_RGBA}" friction="0.6 0.005 0.0001"/>
        <geom name="tool{i}_tip" type="capsule" fromto="0 0 {TOOTHPICK_TIP_RADIUS:.4f} 0 0 0.004" size="{TOOTHPICK_TIP_RADIUS}"
              mass="0.0001" rgba="{TOOTHPICK_RGBA}" friction="0.7 0.005 0.0001"/>
        <geom name="tool{i}_top" type="capsule" fromto="0 0 {L - 0.004:.4f} 0 0 {L - TOOTHPICK_TIP_RADIUS:.4f}" size="{TOOTHPICK_TIP_RADIUS}"
              mass="0.0001" rgba="{TOOTHPICK_RGBA}" friction="0.6 0.005 0.0001"/>
    </body>"""


def cup_xml(rng):
    cx, cy = CUP_XY
    walls = pw.ring_walls(CUP_INNER_R + 0.001, CUP_FLOOR + CUP_DEPTH / 2 - 0.0015,
                          CUP_DEPTH / 2, "0.85 0.87 0.9 0.6", n_segments=20, extra='mass="0.001"')
    outer_r = CUP_INNER_R + 0.0015
    nest = pw.ring_walls(outer_r + NEST_CLEARANCE + 0.002, 0.00125, 0.00125, "0.3 0.3 0.33 1", n_segments=24)  # 2.5 mm low ring
    tab_len = TAB_OUTER - outer_r
    # 24 toothpicks dropped in together, leaning outward against each other and the wall.
    spots = []
    spacing = 0.0038
    for row in range(-5, 6):
        for col in range(-5, 6):
            x = (col + 0.5 * (row % 2)) * spacing
            y = row * spacing * 0.866
            if math.hypot(x, y) < CUP_INNER_R - 0.002:
                spots.append((x, y))
    spots.sort(key=lambda p: math.hypot(*p))
    spots = spots[:N_TOOLS]
    tools = []
    for i, (x, y) in enumerate(spots):
        x = x + rng.uniform(-0.0003, 0.0003)
        y = y + rng.uniform(-0.0003, 0.0003)
        # start almost upright (no overlaps); they lean and fall against each other as they settle
        tools.append(toothpick_xml(i, cx + x, cy + y, rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015)))
    return f"""
    <body name="cup_nest" pos="{cx} {cy} {BENCH_Z}">
        {nest}
    </body>
    <body name="cup" pos="{cx} {cy} {BENCH_Z + 0.0002}">
        <freejoint/>
        <geom name="cup_floor" type="cylinder" pos="0 0 {CUP_FLOOR / 2}" size="{outer_r} {CUP_FLOOR / 2}"
              rgba="0.85 0.87 0.9 1" mass="0.012" friction="0.5 0.005 0.0001"/>
        {walls}
        <!-- printed grab tab, pointing toward the robot (-x) -->
        <geom name="cup_tab" type="box" pos="{-(outer_r + tab_len / 2):.4f} 0 {0.005 + TAB_HEIGHT / 2:.4f}"
              size="{tab_len / 2:.4f} {TAB_HALF_THICK} {TAB_HEIGHT / 2}" rgba="0.2 0.55 0.9 1" mass="0.003" friction="1.2 0.01 0.001"/>
    </body>
    <camera name="cup_cam" pos="{cx} {cy} {BENCH_Z + CUP_CAMERA_HEIGHT}" xyaxes="1 0 0 0 1 0" fovy="{CUP_CAMERA_FOVY}"/>
    {"".join(tools)}"""


# Printed toothpick rack (the "rack" holder): two rows of 12 holes, one toothpick per hole.
RACK_ROW_X = (0.19, 0.215)  # rows 2.5 cm apart: the fingertip behind a toothpick and the lid hook clear the other row
RACK_PITCH = 0.007  # 7 mm between toothpicks along a row: room for the 6 mm wide fingertips
RACK_DEPTH = 0.025  # holes 2.5 cm deep, so 4 cm of each toothpick sticks up
RACK_HOLE_HALF = TOOTHPICK_RADIUS + 0.0005


def rack_slot_xy(i):
    row = i // 12
    col = i % 12
    return RACK_ROW_X[row], (col - 5.5) * RACK_PITCH


def rack_xml():
    walls = []
    h = RACK_DEPTH / 2
    half_len = 6 * RACK_PITCH + 0.003
    for x in RACK_ROW_X:
        for side in (-1, 1):  # long bars in front of and behind the holes
            walls.append(f'<geom type="box" pos="{x + side * (RACK_HOLE_HALF + 0.002):.4f} 0 {h}" size="0.002 {half_len:.4f} {h}" rgba="0.2 0.55 0.9 1"/>')
        for k in range(13):  # short bars between holes
            y = (k - 6) * RACK_PITCH
            walls.append(f'<geom type="box" pos="{x:.4f} {y:.4f} {h}" size="{RACK_HOLE_HALF} {(RACK_PITCH - 2 * RACK_HOLE_HALF) / 2:.4f} {h}" rgba="0.2 0.55 0.9 1"/>')
    # solid block between the two rows
    mid = (RACK_ROW_X[0] + RACK_ROW_X[1]) / 2
    gap = RACK_ROW_X[1] - RACK_ROW_X[0] - 2 * (RACK_HOLE_HALF + 0.004)
    walls.append(f'<geom type="box" pos="{mid:.4f} 0 {h}" size="{gap / 2:.4f} {half_len:.4f} {h}" rgba="0.2 0.55 0.9 1"/>')
    tools = []
    for i in range(N_TOOLS):
        x, y = rack_slot_xy(i)
        tools.append(toothpick_xml(i, x, y, 0.0, 0.0).replace(f'{CUP_FLOOR_TOP + 0.0005:.5f}', f'{BENCH_Z + 0.0002:.5f}'))
    return f"""
    <body name="toothpick_rack" pos="0 0 {BENCH_Z}">
        {chr(10).join("        " + w for w in walls)}
    </body>
    {"".join(tools)}"""


def bin_xml():
    bx, by = BIN_XY
    return f"""
    <body name="waste_bin" pos="{bx} {by} {BENCH_Z}">
        <!-- tall narrow waste cup, 5 cm inside, 8 cm tall: a dropped 6.5 cm toothpick cannot lean out -->
        <geom type="box" pos="0.026 0 0.04" size="0.002 0.028 0.04" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="-0.026 0 0.04" size="0.002 0.028 0.04" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="0 0.026 0.04" size="0.024 0.002 0.04" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="0 -0.026 0.04" size="0.024 0.002 0.04" rgba="0.75 0.2 0.2 1"/>
    </body>"""


PCR_ORIGIN = (0.168, -0.17)  # first PCR tube; the strip runs along +x between the source plate and the bin side


def build_so101_scene(tool_name, seed, holder="cup", pcr=False, sectors=False, source_colonies=None, liquid=False):
    if tool_name != "toothpick":
        raise ValueError("The SO-101 build version uses toothpicks only. A 12 cm loop held upright is too tall for "
                         "the SO-101 to lift out of a cup while pointing straight down.")
    rng = np.random.default_rng(seed)
    colonies = pw.random_colonies(rng)
    if source_colonies is not None:  # a plate grown from an earlier round (positions relative to the plate center)
        colonies = source_colonies
    agar_source = float(rng.uniform(pw.AGAR_MIN, pw.AGAR_MAX))
    agar_dest = float(rng.uniform(pw.AGAR_MIN, pw.AGAR_MAX))

    liquid_parts, liquid_welds, liquid_actuator = ("", "", "")
    if liquid == True:
        import liquid as lq
        liquid_parts, liquid_welds, liquid_actuator = lq.liquid_xml("so101", "gripper")
    xml = f"""<mujoco model="so101 cyano bench">
  <include file="so101_lab.xml"/>
  <option timestep="0.001"/>
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="160" elevation="-30" offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.88 0.92" rgb2="0.55 0.6 0.65" width="512" height="3072"/>
    <texture type="2d" name="floor_tex" builtin="checker" rgb1="0.75 0.75 0.75" rgb2="0.68 0.68 0.68" width="300" height="300"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0.12 0 2" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <light pos="1.2 -0.8 1.2" dir="-1 0.7 -1" diffuse="0.25 0.25 0.25"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floor_mat"/>
    <camera name="work_cam" pos="0.62 0 0.45" xyaxes="0 1 0 -0.6 0 0.8" fovy="45"/>
    <camera name="close_cam" pos="0.50 0 0.38" xyaxes="0 1 0 -0.66 0 0.75" fovy="42"/>
    <body name="bench" pos="0.12 0 0.05">
      <geom name="bench_top" type="box" size="0.25 0.32 0.05" rgba="0.25 0.27 0.3 1"/>
    </body>
    <body name="riser" pos="0 0 {BENCH_Z + RISER_HEIGHT / 2}">
      <geom type="box" size="0.055 0.055 {RISER_HEIGHT / 2}" rgba="0.2 0.55 0.9 1"/>
    </body>
    {pw.plate_xml("source", SOURCE_XY, agar_source, colonies, rng)}
    {pw.plate_xml("dest", DEST_XY, agar_dest, [], rng)}
    {pw.sector_sticker_xml(DEST_XY, BENCH_Z + pw.PLATE_FLOOR + agar_dest) if sectors == True else ''}
    {cup_xml(rng) if holder == "cup" else (rack_xml() if holder == "rack" else "")}
    {liquid_parts}
    {bin_xml()}
    {__import__('pcr').pcr_strip_xml(PCR_ORIGIN, 'x') if pcr == True else ''}
    {"".join(f'<geom name="cal_pin_{i}" type="cylinder" pos="{x} {y} {BENCH_Z + h / 2}" size="{PIN_RADIUS} {h / 2}" rgba="0.9 0.6 0.1 1"/>' for i, (x, y, h) in enumerate(CAL_PINS))}
  </worldbody>
  <equality>
    <!-- Grooved fingertip: once both fingertips touch a toothpick, the groove holds it (switched on in code) -->
    {"".join(f'<weld name="tool_lock_{i}" body1="gripper" body2="tool_{i}" active="false" solref="0.002 1" solimp="0.99 0.999 0.0001"/>' for i in range(N_TOOLS)) if holder != "none" else ""}
    {liquid_welds}
  </equality>
  <actuator>
    {liquid_actuator}
  </actuator>
</mujoco>
"""
    path = os.path.join(HERE, "models", f"_generated_so101_scene_{os.getpid()}.xml")
    with open(path, "w") as f:
        f.write(xml)

    truth = {
        "tool": tool_name,
        "seed": seed,
        "holder": holder,
        "colonies": [{"x": SOURCE_XY[0] + c["x"], "y": SOURCE_XY[1] + c["y"], "r": c["r"]} for c in colonies],
        "agar_top": {"source": BENCH_Z + pw.PLATE_FLOOR + agar_source, "dest": BENCH_Z + pw.PLATE_FLOOR + agar_dest},
    }
    return path, truth
