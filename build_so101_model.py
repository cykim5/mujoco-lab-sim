"""Builds models/so101_lab.xml: the Menagerie SO-101 with our custom 3D-printed jaws.

Run once after changing the jaw design:  ~/.venvs/mujoco-lab/bin/python build_so101_model.py

The printed jaw has two working zones:
  1. Fingertips (fixed jaw + swinging jaw) for picking a single toothpick or loop.
  2. A lid hook on the swinging jaw. When the jaw is opened to LID_ANGLE, the hook sits exactly
     across the lid from the fixed finger, so closing a little pinches the lid by its rim.

All sizes are in meters, in the frame of the SO-101 "gripper" link
(jaws point along -z, the swinging jaw opens toward +x).
"""

import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "assets", "robotstudio_so101", "so101.xml")
OUTPUT = os.path.join(HERE, "models", "so101_lab.xml")

BASE_Z = 0.18  # the robot base sits on an 8 cm riser on the bench (bench top at 0.10)

# Swinging jaw hinge, in the gripper frame (from the Menagerie model)
PIVOT = (0.0202, 0.0188, -0.0234)

# Fingertips: 4 mm thick, 6 mm wide (narrower than the gap between toothpick tops), reaching down to z = -0.105
FINGER_HALF = (0.002, 0.003, 0.035)
FINGER_CENTER_Z = -0.07
FINGER_GAP = 0.003  # gap between the fingertips when the jaw joint is at 0

# Lid hook
LID_ANGLE = 1.70  # jaw angle (rad) at which the hook lines up with the far side of the lid.
# Set near the end of the jaw travel so that when the jaw is nearly closed (gripping a toothpick)
# the hook sits 2 cm ABOVE the fingertips and cannot land on the cup or other toothpicks.
LID_RIM_DIAMETER = 0.095  # outer diameter of the lid rim
HOOK_HALF = (0.002, 0.007, 0.005)  # each of the two V pads: 4 mm thick, 14 mm long, 10 mm tall
HOOK_V_ANGLE = math.radians(14)  # the two pads touch the lid rim 14 degrees either side of the jaw line
HOOK_CLEARANCE = 0.002  # pads sit 2 mm outside the rim at LID_ANGLE
FIXED_FACE = -FINGER_GAP / 2
HOOK_Z = -0.100
HOOK_ARM_Y = 0.012  # the hook arm runs beside the fixed finger, not through it

PRINT_RGBA = "0.2 0.55 0.9 1"  # our printed parts are blue


def gripper_to_jaw(point, angle):
    """Convert a point given in the gripper frame (with the jaw opened to `angle`) to the jaw's own frame."""
    gx, gy, gz = point
    dx = gx - PIVOT[0]
    dz = gz - PIVOT[2]
    # undo the jaw rotation (the jaw turns about the gripper's -y axis)
    ux = dx * math.cos(angle) + dz * math.sin(angle)
    uz = -dx * math.sin(angle) + dz * math.cos(angle)
    # jaw frame: x = gripper x, y = gripper z, z = -gripper y
    return (ux, uz, -(gy - PIVOT[1]))


def quat_about_z(angle):
    return f"{math.cos(angle / 2):.6f} 0 0 {math.sin(angle / 2):.6f}"


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def fmt(v):
    return " ".join(f"{x:.5f}" for x in v)


def fixed_jaw_xml():
    x = -(FINGER_GAP / 2 + FINGER_HALF[0])
    return f"""
                <!-- Printed fixed fingertip -->
                <geom name="fixed_finger" type="box" pos="{x:.5f} 0 {FINGER_CENTER_Z}" size="{fmt(FINGER_HALF)}"
                      rgba="{PRINT_RGBA}" friction="1.5 0.01 0.001" condim="6" solref="0.005 1" priority="1" mass="0.004"/>
                <site name="grip" pos="0 0 -0.100" size="0.002" rgba="1 0 0 0.5" group="4"/>"""


def moving_jaw_xml():
    # Fingertip, defined with the jaw at angle 0
    finger_center = (FINGER_GAP / 2 + FINGER_HALF[0], 0.0, FINGER_CENTER_Z)
    f = gripper_to_jaw(finger_center, 0.0)
    finger_half_local = (FINGER_HALF[0], FINGER_HALF[2], FINGER_HALF[1])
    finger_top = gripper_to_jaw((finger_center[0], 0.0, FINGER_CENTER_Z + FINGER_HALF[2]), 0.0)

    # V-shaped hook, defined with the jaw opened to LID_ANGLE and the lid pressed against the fixed fingertip.
    # Two angled pads touch the rim either side of the jaw line, so closing also centers the lid sideways.
    R = LID_RIM_DIAMETER / 2
    lid_center_x = FIXED_FACE + R
    hook_half_local = (HOOK_HALF[0], HOOK_HALF[2], HOOK_HALF[1])
    pads = []
    for sign in (1, -1):
        phi = sign * HOOK_V_ANGLE
        normal = (math.cos(phi), math.sin(phi))
        reach = R + HOOK_CLEARANCE + HOOK_HALF[0]
        center = (lid_center_x + reach * normal[0], reach * normal[1], HOOK_Z)
        local = gripper_to_jaw(center, LID_ANGLE)
        # orientation: turn about the gripper's vertical by phi, then undo the jaw opening
        q_turn = (math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0)  # gripper vertical = jaw-frame y
        q_jaw = (math.cos(LID_ANGLE / 2), 0.0, 0.0, -math.sin(LID_ANGLE / 2))
        q = quat_mul(q_jaw, q_turn)
        pads.append((local, q))
    hook_mid = gripper_to_jaw((lid_center_x + R + HOOK_CLEARANCE + HOOK_HALF[0], 0.0, HOOK_Z), LID_ANGLE)
    hook_side = gripper_to_jaw((lid_center_x + R + HOOK_CLEARANCE, HOOK_ARM_Y, HOOK_Z), LID_ANGLE)
    hinge_side = (0.0, 0.0, -(HOOK_ARM_Y - PIVOT[1]))
    pad_xml = ""
    for name, (local, q) in zip(("lid_hook_a", "lid_hook_b"), pads):
        pad_xml += f"""
                  <geom name="{name}" type="box" pos="{fmt(local)}" size="{fmt(hook_half_local)}" quat="{fmt(q)}"
                        rgba="{PRINT_RGBA}" friction="1.0 0.005 0.0001" condim="4" solref="0.005 1" priority="1" mass="0.003"/>
                  <geom type="capsule" fromto="{fmt(hook_side)} {fmt(local)}" size="0.003"
                        rgba="{PRINT_RGBA}" contype="0" conaffinity="0" mass="0.001"/>"""

    return f"""
                  <!-- Printed swinging fingertip -->
                  <geom name="moving_finger" type="box" pos="{fmt(f)}" size="{fmt(finger_half_local)}"
                        rgba="{PRINT_RGBA}" friction="1.5 0.01 0.001" condim="6" solref="0.005 1" priority="1" mass="0.004"/>
                  <geom name="finger_arm" type="capsule" fromto="0 0 {f[2]:.5f} {fmt(finger_top)}" size="0.004"
                        rgba="{PRINT_RGBA}" contype="0" conaffinity="0" mass="0.002"/>
                  <!-- Printed V-shaped lid hook: lines up with the far side of the lid when the jaw is open {LID_ANGLE} rad -->{pad_xml}
                  <geom name="hook_arm" type="capsule" fromto="{fmt(hinge_side)} {fmt(hook_side)}" size="0.003"
                        rgba="{PRINT_RGBA}" contype="0" conaffinity="0" mass="0.004"/>"""


def main():
    xml = open(SOURCE).read()
    xml = xml.replace('meshdir="assets"', 'meshdir="../assets/robotstudio_so101/assets"')
    xml = xml.replace('<option integrator="implicitfast" timestep="0.005" cone="elliptic" iterations="10" ls_iterations="20" impratio="10"/>',
                      '<option integrator="implicitfast" cone="elliptic" impratio="10"/>')
    xml = xml.replace('<body name="base" pos="0 0 0" quat="1 0 0 0" childclass="so101">',
                      f'<body name="base" pos="0 0 {BASE_Z}" quat="1 0 0 0" childclass="so101">')

    # Remove the stock jaw collision shapes and the stock swinging-jaw look. Our printed jaw replaces them.
    xml = re.sub(r'\s*<geom name="(fixed_jaw_\w+|moving_jaw_\w+)"[^>]*/>', "", xml, flags=re.S)
    xml = re.sub(r'\s*<geom type="mesh" class="collision_gripper_mesh"[^>]*/>', "", xml, flags=re.S)
    xml = re.sub(r'\s*<geom type="mesh" class="visual" pos="0 0 0.0189" quat="1 0 0 0" mesh="moving_jaw_so101_v1"\s*material="moving_jaw_so101_v1_material"/>',
                 "", xml, flags=re.S)

    # Remove the wrist-camera mount: we use fixed cameras above the bench, and the mount hits the
    # shoulder when the arm folds in to reach colonies near the base.
    xml = re.sub(r'\s*<!-- Camera mount -->\s*<body name="camera_mount">.*?</body>', "", xml, flags=re.S)

    # Add the printed parts.
    marker_fixed = '<!-- Frame gripperframe -->'
    assert marker_fixed in xml
    xml = xml.replace(marker_fixed, fixed_jaw_xml() + "\n                " + marker_fixed)
    marker_moving = '<joint axis="0 0 1" name="gripper" type="hinge" range="-0.174533 1.7453292" class="sts3215"/>'
    assert marker_moving in xml
    xml = xml.replace(marker_moving, marker_moving + moving_jaw_xml())

    header = ("<!-- Generated by build_so101_model.py from the MuJoCo Menagerie SO-101 (Apache 2.0).\n"
              "     Changes: sits on an 8 cm riser, wrist-camera mount removed, stock jaw collisions replaced by our printed fingertips and lid hook. -->\n")
    with open(OUTPUT, "w") as f:
        f.write(header + xml)
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
