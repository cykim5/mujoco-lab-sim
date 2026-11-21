"""Builds the SO-101 ("Build" version) bench as a MuJoCo XML file.

The SO-101 is small, so everything sits 12 to 27 cm from its base (top view, robot at the origin):

    lid parking (-0.02, -0.20)                       lid parking (-0.02, 0.20)
              source plate (0.142, -0.111)    fresh plate (0.142, 0.111)
                      toothpick cup on shaker (0.20, 0.0)
                      waste bin (0.265, 0.0)

Toothpicks stand jumbled together in a cup on a small vibrating base (a cheap vibration motor in
real life). The robot shakes the cup and picks a toothpick whose top has separated from the rest.
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
BIN_XY = (0.265, 0.0)

N_TOOLS = 24
CUP_INNER_R = 0.016  # a narrow 3.2 cm holder: toothpicks lean on the rim and their tops fan out
CUP_DEPTH = 0.045  # deep enough that leaning toothpicks do not tip out over the rim
CUP_FLOOR = 0.003
PLATFORM_HEIGHT = 0.012  # shaker platform top above the bench
CUP_FLOOR_TOP = BENCH_Z + PLATFORM_HEIGHT + CUP_FLOOR

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
    walls = pw.ring_walls(CUP_INNER_R + 0.001, PLATFORM_HEIGHT + CUP_FLOOR + CUP_DEPTH / 2 - 0.0015,
                          CUP_DEPTH / 2, "0.85 0.87 0.9 0.6", n_segments=20)
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
    <body name="shaker" pos="{cx} {cy} {BENCH_Z}">
        <joint name="shake_x" type="slide" axis="1 0 0" range="-0.004 0.004" damping="2"/>
        <joint name="shake_y" type="slide" axis="0 1 0" range="-0.004 0.004" damping="2"/>
        <geom type="box" pos="0 0 {PLATFORM_HEIGHT / 2}" size="0.034 0.034 {PLATFORM_HEIGHT / 2}" rgba="0.2 0.2 0.22 1"
              contype="0" conaffinity="0" mass="0.2"/>
        <geom name="cup_floor" type="cylinder" pos="0 0 {PLATFORM_HEIGHT + CUP_FLOOR / 2}" size="{CUP_INNER_R + 0.0015} {CUP_FLOOR / 2}"
              rgba="0.85 0.87 0.9 1" mass="0.01"/>
        {walls}
    </body>
    <camera name="cup_cam" pos="{cx} {cy} {BENCH_Z + CUP_CAMERA_HEIGHT}" xyaxes="1 0 0 0 1 0" fovy="{CUP_CAMERA_FOVY}"/>
    {"".join(tools)}"""


def bin_xml():
    bx, by = BIN_XY
    return f"""
    <body name="waste_bin" pos="{bx} {by} {BENCH_Z}">
        <geom type="box" pos="0.025 0 0.02" size="0.002 0.027 0.02" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="-0.025 0 0.02" size="0.002 0.027 0.02" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="0 0.025 0.02" size="0.023 0.002 0.02" rgba="0.75 0.2 0.2 1"/>
        <geom type="box" pos="0 -0.025 0.02" size="0.023 0.002 0.02" rgba="0.75 0.2 0.2 1"/>
    </body>"""


def build_so101_scene(tool_name, seed):
    if tool_name != "toothpick":
        raise ValueError("The SO-101 build version uses toothpicks only. A 12 cm loop held upright is too tall for "
                         "the SO-101 to lift out of a cup while pointing straight down.")
    rng = np.random.default_rng(seed)
    colonies = pw.random_colonies(rng)
    agar_source = float(rng.uniform(pw.AGAR_MIN, pw.AGAR_MAX))
    agar_dest = float(rng.uniform(pw.AGAR_MIN, pw.AGAR_MAX))

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
    {cup_xml(rng)}
    {bin_xml()}
  </worldbody>
  <equality>
    <!-- Grooved fingertip: once both fingertips touch a toothpick, the groove holds it (switched on in code) -->
    {"".join(f'<weld name="tool_lock_{i}" body1="gripper" body2="tool_{i}" active="false" solref="0.002 1" solimp="0.99 0.999 0.0001"/>' for i in range(N_TOOLS))}
  </equality>
  <actuator>
    <!-- Vibration base under the cup -->
    <position name="shaker_x" joint="shake_x" kp="200" ctrlrange="-0.004 0.004"/>
    <position name="shaker_y" joint="shake_y" kp="200" ctrlrange="-0.004 0.004"/>
  </actuator>
</mujoco>
"""
    path = os.path.join(HERE, "models", f"_generated_so101_scene_{os.getpid()}.xml")
    with open(path, "w") as f:
        f.write(xml)

    truth = {
        "tool": tool_name,
        "seed": seed,
        "colonies": [{"x": SOURCE_XY[0] + c["x"], "y": SOURCE_XY[1] + c["y"], "r": c["r"]} for c in colonies],
        "agar_top": {"source": BENCH_Z + pw.PLATE_FLOOR + agar_source, "dest": BENCH_Z + pw.PLATE_FLOOR + agar_dest},
    }
    return path, truth
