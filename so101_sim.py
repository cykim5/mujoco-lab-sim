"""SO-101 "Build" version of the cyano colony pick and restreak.

Differences from the Panda "Reference" version:
  - A cheap 3D-printed 5-joint arm (SO-101) on an 8 cm riser, with our printed two-zone jaw.
  - Lids: the jaw opens wide and a printed hook pinches the lid by its rim. No suction.
  - Toothpicks stand jumbled in a cup. The robot shakes the cup, finds a toothpick whose top has
    separated from the rest (depth camera above the cup), and pulls just that one out by friction.
  - Realistic motors: no gravity compensation, and each joint has a small calibration error that the
    robot does not know about (default up to 0.5 degrees).
"""

import math
import os
from collections import deque

import mujoco
import numpy as np

import plate_world as pw
import so101_world as sw
from cyano_sim import CyanoSim
from lab_sim import LabSim

HOME = np.array([1.85, -1.6, 1.5, 1.0, 0.0])  # folded up and turned to the side, out of every camera's view
JAW_CLOSED = -0.15  # squeeze (the fingertips meet before this, so the servo presses)
JAW_TOOTHPICK_OPEN = 0.12  # replaced at startup by the angle that gives TOOTHPICK_GAP between fingertips
TOOTHPICK_GAP = 0.009  # open just 9 mm: room for one toothpick plus aiming error, not a bundle
AIM_MARGIN = 0.003  # aim so the toothpick passes 3 mm from the fixed finger (covers 1 to 2 mm of arm error)
FINGER_THICKNESS = 0.004
FINGER_HALF_WIDTH = 0.003
FIT_MARGIN = 0.001
JAW_RELEASE = 0.5
JAW_LID_OPEN = 1.745  # jaw fully open: hook swung out past the lid rim
LID_HOOK_ANGLE = 1.70  # from build_so101_model.py
LID_RIM_R = 0.0475
FIXED_FACE = -0.0015  # fixed fingertip inner face, gripper-frame x
LID_GRIP_OFFSET = LID_RIM_R - FIXED_FACE + 0.0015 - 0.0015  # near rim lands 1.5 mm from the fixed fingertip face
LID_HELD_OFFSET = LID_RIM_R + FIXED_FACE  # once squeezed, the lid presses on the fixed fingertip
TOOL_YAW = 0.0  # gripper rotation used whenever a toothpick is held (keeps the measured tip offset valid)
SAFE_Z = pw.BENCH_Z + 0.11  # travel height with nothing in the fingers: above every toothpick top
SAFE_Z_TOOL = pw.BENCH_Z + 0.135  # travel height holding a toothpick: its tip also clears the other tops
SHAKE_FREQUENCY = 4.0


def quat_for_yaw(yaw):
    """Gripper pointing straight down, jaw opening toward the given compass direction."""
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def label_blobs(mask, min_pixels=3):
    """Group touching True pixels into blobs. Returns a list of (N, 2) arrays of (row, col)."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    blobs = []
    for start in zip(*np.nonzero(mask)):
        if seen[start] == True:
            continue
        queue = deque([start])
        seen[start] = True
        pixels = []
        while len(queue) > 0:
            y, x = queue.popleft()
            pixels.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == True and seen[ny, nx] == False:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(pixels) >= min_pixels:
            blobs.append(np.array(pixels))
    return blobs


class PickFailed(Exception):
    pass


class So101CyanoSim(CyanoSim):
    def __init__(self, tool_name="toothpick", seed=0, record_video=True, calibration_error_deg=0.5, holder="cup", pcr=False, sectors=False, source_colonies=None):
        self.holder = holder
        scene_path, self.truth = sw.build_so101_scene(tool_name, seed, holder=holder, pcr=pcr, sectors=sectors, source_colonies=source_colonies)
        self.tool_name = tool_name
        self.tool = {"length": sw.TOOTHPICK_LENGTH, "tip_radius": sw.TOOTHPICK_TIP_RADIUS}
        self.source_xy = sw.SOURCE_XY
        self.dest_xy = sw.DEST_XY
        self.lid_park_xy = sw.LID_PARK_XY
        self.bin_xy = sw.BIN_XY
        self.park_arm = HOME
        self.gripper_open = JAW_RELEASE
        self.gripper_closed = JAW_CLOSED
        self.check_rack_bumps = False
        self.n_arm = 5
        self.gripper_index = 5
        self.calibration_offset = np.zeros(5)
        LabSim.__init__(self, record_video=record_video, video_camera="close_cam", video_size=(320, 240),
                        video_fps=4, scene_path=scene_path)
        os.remove(scene_path)

        m = self.model
        self.agar_geom = {"source": {m.geom("source_agar").id}, "dest": {m.geom("dest_agar").id}}
        self.wall_geoms = set()
        for name in ["source", "dest"]:
            plate_body = m.body("plate_" + name).id
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == plate_body and g not in self.agar_geom[name]:
                    self.wall_geoms.add(g)
        self.lid_geoms = {}
        for name in ["source", "dest"]:
            lid_body = m.body("lid_" + name).id
            self.lid_geoms[name] = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == lid_body)
        self.fixed_finger = {m.geom("fixed_finger").id}
        self.moving_finger = {m.geom("moving_finger").id}
        self.hook = {m.geom("lid_hook_a").id, m.geom("lid_hook_b").id}
        self.finger_geoms = self.fixed_finger | self.moving_finger
        self.tool_body_ids = [m.body(f"tool_{i}").id for i in range(sw.N_TOOLS)]
        self.robot_body_ids = set(range(m.body("base").id, m.body("moving_jaw_so101_v1").id + 1))
        self.target_body = None
        self.locked_body = None
        self.tool_geoms = set()  # filled in once a toothpick is picked
        self.tool_body_name = None
        self.tool_tip_site = None
        self.tip_offset = None
        self.hit_wall = False
        self.bumped = set()
        self.log = []
        self.pick_info = {}
        self.slot = (seed * 7) % sw.N_TOOLS if holder == "rack" else -1  # which rack hole to take first
        self.track_bumps = False
        self.bumped = set()
        self.squeeze = 0.003  # hold about 2.9 N*m at the jaw once clamped
        self.shake_style = "swirl"  # tested side-to-side, tap and swirl: swirl left the most pickable, spilled none

        # Motor calibration error: each joint really sits a little off from what its encoder reports.
        rng = np.random.default_rng(seed + 1000)
        self.calibration_offset = np.radians(rng.uniform(-calibration_error_deg, calibration_error_deg, 5))
        self.calibration_estimate = np.zeros(5)
        self.apply_joint_error()
        self.holding_tool = False
        self.shakes = 0
        self.pin_geoms = set(m.geom(f"cal_pin_{i}").id for i in range(len(sw.CAL_PINS)))

        self.set_yaw(TOOL_YAW)
        self.measure_jaw()
        self.wait(2.0)  # let the toothpicks settle in the cup

    # ---------- robot basics ----------

    def reset_robot(self):
        self.data.qpos[:5] = HOME
        self.data.ctrl[:5] = HOME
        self.data.ctrl[5] = JAW_TOOTHPICK_OPEN

    def solve_ik(self, target_pos, start_arm=None, max_iters=300, tolerance=1e-4, target_quat=None):
        """Same as the shared IK, but if it gets stuck (a 5-joint arm has awkward local minima),
        retry from a few other starting poses and keep the best answer closest to the current pose."""
        q, err = super().solve_ik(target_pos, start_arm, max_iters, tolerance, target_quat)
        if err < 1e-3:
            return q, err
        here = self.data.ctrl[:5].copy() if start_arm is None else np.array(start_arm)
        rng = np.random.default_rng(int(abs(target_pos[0] * 1e4 + target_pos[1] * 1e3)) % 10000)
        lo, hi = self.model.jnt_range[:5, 0], self.model.jnt_range[:5, 1]
        best_q, best_err = q, err
        for _ in range(12):
            start = rng.uniform(lo, hi)
            start[0] = math.atan2(target_pos[1], target_pos[0])  # face the target
            cand, cand_err = super().solve_ik(target_pos, start, 400, tolerance, target_quat)
            if cand_err < best_err - 1e-4 or (cand_err < 1e-3 and np.linalg.norm(cand - here) < np.linalg.norm(best_q - here)):
                best_q, best_err = cand, cand_err
        return best_q, best_err

    def apply_joint_error(self):
        """Each joint really sits (true error - what calibration measured) away from what its encoder says.
        Real STS3215 servos can store a measured offset, which is what applying the calibration means."""
        self.joint_error = self.calibration_offset - self.calibration_estimate
        for i in range(5):
            kp = self.model.actuator_gainprm[i, 0]
            self.model.actuator_biasprm[i, 0] = kp * self.joint_error[i]

    def set_yaw(self, yaw):
        self.down_quat = quat_for_yaw(yaw)

    def step(self):
        super().step()
        if self.track_bumps == True:
            m, d = self.model, self.data
            held = getattr(self, "locked_body", None)
            for i in range(d.ncon):
                c = d.contact[i]
                for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if m.geom_bodyid[a] in self.robot_body_ids:
                        body = int(m.geom_bodyid[b])
                        if body in self.tool_body_ids and body != held and body != self.target_body:
                            self.bumped.add(body)

    # ---------- safe travel: never sweep low across the bench ----------

    def safe_height(self):
        return SAFE_Z_TOOL if self.holding_tool == True else SAFE_Z

    def travel(self, xy, final_z, gripper, duration=1.5):
        """Go up to a safe height, move across in short straight steps, then come down."""
        here = self.get_grip_pos()
        safe = self.safe_height()
        if here[2] < safe - 0.003:
            self.move_grip_to([here[0], here[1], safe], gripper, duration=0.6)
        here = self.get_grip_pos()
        # Arc around the robot base (interpolate distance and angle), keeping at least 13 cm out, where the
        # SO-101 can still reach the safe height.
        r0, a0 = math.hypot(here[0], here[1]), math.atan2(here[1], here[0])
        r1, a1 = math.hypot(xy[0], xy[1]), math.atan2(xy[1], xy[0])
        distance = float(np.hypot(xy[0] - here[0], xy[1] - here[1]))
        n = max(1, int(distance / 0.04))
        for k in range(1, n + 1):
            f = k / n
            r = max(r0 + (r1 - r0) * f, 0.13) if k < n else r1
            a = a0 + (a1 - a0) * f
            p = [r * math.cos(a), r * math.sin(a)] if k < n else xy
            self.move_grip_to([p[0], p[1], safe], gripper, duration=max(0.25, duration / n))
        self.move_grip_to([xy[0], xy[1], final_z], gripper, duration=0.8)

    def travel_tip(self, xy, tip_z, duration=1.5):
        grip_xy = np.array(xy) - self.tip_offset[:2]
        self.travel(grip_xy, tip_z - self.tip_offset[2], self.data.ctrl[5], duration)

    def move_arm(self, target_arm, gripper, duration=1.5, settle=0.3):
        # Before folding back to the parked pose, rise straight up first so the swing stays high.
        already_parked = np.linalg.norm(self.data.ctrl[:5] - self.park_arm) < 0.05
        if np.array_equal(target_arm, self.park_arm) and hasattr(self, "pin_geoms") and already_parked == False:
            here = self.get_grip_pos()
            if here[2] < self.safe_height() - 0.003:
                super().move_grip_to([here[0], here[1], self.safe_height()], gripper, duration=0.6)
        super().move_arm(target_arm, gripper, duration, settle)

    # ---------- calibration by touching pins ----------

    def calibrate(self):
        """Touch 4 pins at known spots (top, and from all four sides) and solve for each joint's error."""
        self.set_yaw(TOOL_YAW)
        jaw = 0.0  # fingertips 3 mm apart: the closed fingertip block is 11 mm along x and 6 mm along y
        extent = {"x": 0.0055, "y": 0.003}
        records = []

        def touching():
            force, _ = self.contact(self.finger_geoms, self.pin_geoms)
            return force > 0.15

        for (px, py, height) in sw.CAL_PINS:
            pin_top = pw.BENCH_Z + height
            # top touch
            self.travel([px, py], pin_top + 0.005 + 0.006, jaw)
            for k in range(40):
                self.move_grip_to([px, py, pin_top + 0.005 + 0.006 - k * 0.0003], jaw, duration=0.02)
                self.wait(0.005)
                if touching() == True:
                    records.append((self.data.qpos[:5] - self.joint_error, 2, pin_top + 0.005))
                    break
            # side touches
            for axis, sign in (("x", 1), ("x", -1), ("y", 1), ("y", -1)):
                a = 0 if axis == "x" else 1
                d = np.zeros(2)
                d[a] = sign
                start = np.array([px, py]) - d * (sw.PIN_RADIUS + extent[axis] + 0.003)
                z = pin_top - 0.001
                self.move_grip_to([start[0], start[1], pin_top + 0.012], jaw, duration=0.4)
                self.move_grip_to([start[0], start[1], z], jaw, duration=0.3)
                for k in range(40):
                    p = start + d * k * 0.0002
                    self.move_grip_to([p[0], p[1], z], jaw, duration=0.02)
                    self.wait(0.005)
                    if touching() == True:
                        expected = [px, py][a] - sign * (sw.PIN_RADIUS + extent[axis])
                        records.append((self.data.qpos[:5] - self.joint_error, a, expected))
                        break
                self.move_grip_to([start[0], start[1], pin_top + 0.012], jaw, duration=0.3)

        # Least squares: find joint offsets that make the robot's kinematics agree with the known pin positions.
        def predict(q):
            d = self.ik_data
            d.qpos[:] = self.data.qpos
            d.qpos[:5] = q
            mujoco.mj_kinematics(self.model, d)
            return d.site_xpos[self.grip_site].copy()

        # The wrist roll cannot be measured this way (the fingertip point sits on its axis), so only the
        # first four joints are estimated. A small pull toward zero keeps the fit from running away.
        estimate = self.calibration_estimate.copy()
        joints = 4
        for _ in range(8):
            residuals = []
            J = []
            for q, a, expected in records:
                base = predict(q + estimate)[a] - expected
                residuals.append(base)
                row = []
                for j in range(joints):
                    dq = np.zeros(5)
                    dq[j] = 1e-5
                    row.append((predict(q + estimate + dq)[a] - predict(q + estimate)[a]) / 1e-5)
                J.append(row)
            J = np.array(J)
            r = np.array(residuals)
            lam = 1e-4  # regularization (meters per radian scale)
            A = J.T @ J + lam * np.eye(joints)
            step = np.zeros(5)
            step[:joints] = np.linalg.solve(A, -J.T @ r - lam * estimate[:joints])
            estimate = estimate + step
            if np.linalg.norm(step) < 1e-6:
                break
        before = float(np.sqrt(np.mean(np.array([predict(q)[a] - e for q, a, e in records]) ** 2)) * 1000)
        after = float(np.sqrt(np.mean(np.array([predict(q + estimate)[a] - e for q, a, e in records]) ** 2)) * 1000)
        self.calibration_estimate = estimate
        self.apply_joint_error()
        self.calibration_info = {"cal_touches": len(records), "cal_error_before_mm": round(before, 2),
                                 "cal_error_after_mm": round(after, 2),
                                 "joint_error_left_deg": [round(float(math.degrees(v)), 3) for v in self.joint_error]}
        self.say(f"calibrated on {len(records)} pin touches: fit error {before:.2f} mm -> {after:.2f} mm")
        self.move_arm(self.park_arm, JAW_TOOTHPICK_OPEN, duration=1.5)

    def get_grip_pos(self):
        """Where the robot BELIEVES its gripper is, from its encoders and whatever calibration it has done."""
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[:5] = self.data.qpos[:5] - self.joint_error
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.grip_site].copy()

    def tip_pos(self):
        if self.tool_tip_site is None:
            return self.data.site_xpos[self.grip_site].copy()
        return self.data.site_xpos[self.tool_tip_site].copy()

    def set_jaw(self, angle, duration=0.4):
        self.move_arm(self.data.ctrl[:5].copy(), angle, duration, settle=0.1)

    def jaw_faces(self, angle):
        """Inner faces of the two fingertips (gripper-frame x, at the fingertip) for a jaw angle."""
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[5] = angle
        mujoco.mj_kinematics(self.model, d)
        g = self.model.body("gripper").id
        R = d.xmat[g].reshape(3, 3)
        moving = R.T @ (d.geom_xpos[self.model.geom("moving_finger").id] - d.xpos[g])
        fixed = R.T @ (d.geom_xpos[self.model.geom("fixed_finger").id] - d.xpos[g])
        return fixed[0] + FINGER_THICKNESS / 2, moving[0] - FINGER_THICKNESS / 2 * math.cos(angle)

    def measure_jaw(self):
        """Find the jaw angle that opens the fingertips TOOTHPICK_GAP apart, and where to aim the toothpick."""
        global JAW_TOOTHPICK_OPEN
        for angle in np.arange(0.0, 0.6, 0.002):
            fixed_face, moving_face = self.jaw_faces(angle)
            if moving_face - fixed_face >= TOOTHPICK_GAP:
                JAW_TOOTHPICK_OPEN = float(angle)
                break
        self.fixed_face, self.moving_face = self.jaw_faces(JAW_TOOTHPICK_OPEN)
        # where the toothpick center should sit in the gripper frame
        self.toothpick_aim = self.fixed_face + AIM_MARGIN + sw.TOOTHPICK_RADIUS

    def fingers_fit(self, candidate, yaw):
        """Would the open fingertips come down around this toothpick, at this wrist rotation, without touching others?"""
        dx = candidate["others_x"] - candidate["x"]
        dy = candidate["others_y"] - candidate["y"]
        u = dx * math.cos(yaw) + dy * math.sin(yaw)  # along the jaw opening direction
        v = -dx * math.sin(yaw) + dy * math.cos(yaw)
        u_low = self.fixed_face - FINGER_THICKNESS - self.toothpick_aim - FIT_MARGIN
        u_high = self.moving_face + FINGER_THICKNESS - self.toothpick_aim + FIT_MARGIN
        inside = (u > u_low) & (u < u_high) & (np.abs(v) < FINGER_HALF_WIDTH + FIT_MARGIN)
        return int(inside.sum()) == 0

    # ---------- lids: printed hook ----------

    def lift_lid(self, plate_name, from_xy, to_xy, put_back):
        lid_geoms = self.lid_geoms[plate_name]
        on_plate = put_back == False
        lid_z = pw.BENCH_Z + (pw.PLATE_WALL_TOP if on_plate == True else pw.LID_WALL_HEIGHT)
        grip_z = max(lid_z - 0.004, pw.BENCH_Z + 0.0065)  # middle of the rim, but fingertips stay off the bench

        yaw = math.atan2(from_xy[1], from_xy[0])  # jaw opens straight away from the robot
        direction = np.array([math.cos(yaw), math.sin(yaw)])
        # Come down with the fixed fingertip 4 mm outside the rim (room for arm error), then slide in until
        # the fingertip feels the rim (servo load). This finds the lid by touch, whatever the calibration error.
        grip_xy = np.array(from_xy) - (LID_GRIP_OFFSET + 0.004) * direction
        self.set_yaw(yaw)
        self.travel(grip_xy, grip_z + 0.04, JAW_LID_OPEN)
        self.move_grip_to([grip_xy[0], grip_xy[1], grip_z], JAW_LID_OPEN, duration=0.8)
        for _ in range(24):
            finger_force, _ = self.contact(self.fixed_finger, lid_geoms)
            if finger_force > 0.2:
                break
            grip_xy = grip_xy + 0.0005 * direction
            self.move_grip_to([grip_xy[0], grip_xy[1], grip_z], JAW_LID_OPEN, duration=0.05)
            self.wait(0.02)

        # Close slowly until the hook feels the lid (the servo reports load), then squeeze a little.
        angle = JAW_LID_OPEN
        touched = False
        while angle > LID_HOOK_ANGLE - 0.12:
            angle = angle - 0.01
            self.data.ctrl[5] = angle
            self.wait(0.03)
            hook_force, _ = self.contact(self.hook, lid_geoms)
            finger_force, _ = self.contact(self.fixed_finger, lid_geoms)
            if hook_force > 0.3 and finger_force > 0.3:
                touched = True
                break
        if touched == False:
            self.say(f"hook never closed on the {plate_name} lid")
            self.set_jaw(JAW_LID_OPEN)
            self.move_grip_to([grip_xy[0], grip_xy[1], grip_z + 0.04], JAW_LID_OPEN, duration=0.6)
            return False
        self.data.ctrl[5] = angle - 0.01
        self.wait(0.3)

        self.move_grip_to([grip_xy[0], grip_xy[1], grip_z + 0.03], self.data.ctrl[5], duration=0.8)
        # The lid is pressed against the fixed finger, so its center sits a known distance from the gripper.
        to_yaw = math.atan2(to_xy[1], to_xy[0])
        to_dir = np.array([math.cos(to_yaw), math.sin(to_yaw)])
        place_xy = np.array(to_xy) - LID_HELD_OFFSET * to_dir
        place_lid_z = pw.BENCH_Z + (pw.PLATE_WALL_TOP if put_back == True else pw.LID_WALL_HEIGHT) + 0.0005
        held_grip_z = self.get_grip_pos()[2] - (self.get_body_pos("lid_" + plate_name)[2] - 0.0)  # grip height above the lid
        self.set_yaw(to_yaw)
        self.travel(place_xy, place_lid_z + held_grip_z + 0.03, self.data.ctrl[5], duration=1.8)
        self.move_grip_to([place_xy[0], place_xy[1], place_lid_z + held_grip_z], self.data.ctrl[5], duration=0.8)
        if put_back == True:
            place_xy = self.spiral_onto_plate(place_xy, place_lid_z + held_grip_z)
        # Let go gently: open just enough to free the lid, lift straight up, then swing the hook fully open.
        self.set_jaw(LID_HOOK_ANGLE + 0.03, duration=0.4)
        self.move_grip_to([place_xy[0], place_xy[1], place_lid_z + held_grip_z + 0.015], self.data.ctrl[5], duration=0.6)
        self.set_jaw(JAW_LID_OPEN, duration=0.4)
        self.move_grip_to([place_xy[0], place_xy[1], place_lid_z + held_grip_z + 0.04], JAW_LID_OPEN, duration=0.6)
        self.set_yaw(TOOL_YAW)
        return True

    def spiral_onto_plate(self, xy, seated_grip_z):
        """If the lid stops high (its rim resting on the plate wall), shift it in small circles until it drops
        into place. The robot feels this through its encoders: the gripper cannot reach the seated height."""
        def blocked():
            return self.get_grip_pos()[2] - seated_grip_z > 0.001
        if blocked() == False:
            return xy
        above = seated_grip_z + 0.009  # clear the rim (a blocked lid rests about 5 mm high) before each new spot
        for radius in [0.001, 0.002, 0.003, 0.004, 0.005]:
            for k in range(8):
                a = 2 * math.pi * k / 8
                trial = xy + radius * np.array([math.cos(a), math.sin(a)])
                self.move_grip_to([trial[0], trial[1], above], self.data.ctrl[5], duration=0.15)
                self.move_grip_to([trial[0], trial[1], seated_grip_z - 0.0005], self.data.ctrl[5], duration=0.15)
                if blocked() == False:
                    return trial
        return xy

    # ---------- toothpick cup ----------

    def shake_cup(self, round_number, style=None):
        """Pinch the cup's grab tab and shake gently. Each round is a little stronger (1.5 mm up to 3.5 mm)."""
        cup = self.data.body("cup")
        R = cup.xmat.reshape(3, 3)
        tab_dir = -R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9)
        tab_point = cup.xpos[:2] + tab_dir * sw.TAB_GRIP_R
        across = R[:2, 1]
        yaw = math.atan2(across[1], across[0])  # jaw closes across the tab's thickness
        direction = np.array([math.cos(yaw), math.sin(yaw)])
        aim = self.fixed_face + AIM_MARGIN + sw.TAB_HALF_THICK
        grip_xy = tab_point - aim * direction
        grip_z = pw.BENCH_Z + 0.017  # fingertips 12 mm above the bench, on the tab (tab spans 5 to 35 mm)
        self.set_yaw(yaw)
        self.travel(grip_xy, grip_z, JAW_TOOTHPICK_OPEN)
        tab = {self.model.geom("cup_tab").id}
        angle = self.data.ctrl[5]
        while angle > JAW_CLOSED:
            angle = angle - 0.004
            self.data.ctrl[5] = angle
            self.wait(0.02)
            f1, _ = self.contact(self.fixed_finger, tab)
            f2, _ = self.contact(self.moving_finger, tab)
            if f1 > 0.2 and f2 > 0.2:
                break
        self.data.ctrl[5] = self.data.qpos[5] - 0.002
        amplitude = min(0.0015 + 0.0005 * (round_number - 1), 0.0035)
        center = np.array([grip_xy[0], grip_xy[1], grip_z])
        arm = self.data.ctrl[:5].copy()
        t0 = self.data.time
        style = self.shake_style if style is None else style
        while self.data.time - t0 < 1.2:
            t = self.data.time - t0
            w = 2 * math.pi * SHAKE_FREQUENCY * t
            if style == "tap":  # lift the cup a few mm and set it down again
                offset = np.array([0.0, 0.0, (amplitude + 0.0015) * (0.5 - 0.5 * math.cos(w))])
            elif style == "swirl":  # small circles
                offset = amplitude * np.array([math.cos(w), math.sin(w), 0.0])
            else:  # side to side
                offset = amplitude * np.array([math.sin(w), 0.6 * math.sin(1.3 * w), 0.0])
            arm, _ = LabSim.solve_ik(self, center + offset, start_arm=arm, max_iters=30)
            self.data.ctrl[:5] = arm
            for _ in range(10):
                self.step()
        self.move_grip_to(center, self.data.ctrl[5], duration=0.3)
        self.set_jaw(JAW_TOOTHPICK_OPEN)
        self.move_grip_to([grip_xy[0], grip_xy[1], SAFE_Z], JAW_TOOTHPICK_OPEN, duration=0.8)
        self.set_yaw(TOOL_YAW)
        self.shakes = self.shakes + 1
        self.say(f"shook the cup gently (round {round_number}, {amplitude * 1000:.1f} mm)")

    def find_separated_toothpick(self, size=480):
        """Depth camera above the cup. A toothpick top is a local high point: nothing within 5 mm rises above it.
        Because every bottom rests on the cup floor, the height of a top tells how far that stick leans."""
        renderer = mujoco.Renderer(self.model, height=size, width=size)
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera="cup_cam")
        depth = renderer.render().copy()
        renderer.close()

        cam_z = pw.BENCH_Z + sw.CUP_CAMERA_HEIGHT
        height = cam_z - depth
        scale = 2 * math.tan(math.radians(sw.CUP_CAMERA_FOVY / 2)) / size  # meters per pixel per meter of depth
        cols, rows = np.meshgrid(np.arange(size), np.arange(size))
        cup_x = self.data.body("cup").xpos[0]
        cup_y = self.data.body("cup").xpos[1]
        world_x = sw.CUP_XY[0] + (cols + 0.5 - size / 2) * depth * scale
        world_y = sw.CUP_XY[1] - (rows + 0.5 - size / 2) * depth * scale

        rim_top = sw.CUP_RIM_Z
        near_cup = np.hypot(world_x - cup_x, world_y - cup_y) < sw.CUP_INNER_R + 0.035
        valid = near_cup & (height > rim_top + 0.002) & (height < rim_top + sw.TOOTHPICK_LENGTH)

        # Local maxima on a 4x coarser grid (about 0.7 mm per cell).
        step = 4
        h = np.where(valid, height, -1.0)[::step, ::step]
        radius = int(round(0.005 / (0.084 / size * step)))  # 5 mm in coarse cells
        padded = np.pad(h, radius, constant_values=-1.0)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
        local_max = windows.max(axis=(2, 3))
        peaks = np.argwhere((h > rim_top + 0.004) & (h >= local_max - 1e-6))

        candidates = []
        for r, c in peaks:
            row, col = r * step, c * step
            top = np.array([world_x[row, col], world_y[row, col]])
            top_z = float(height[row, col])
            lean = math.acos(min(1.0, (top_z - sw.CUP_FLOOR_TOP) / sw.TOOTHPICK_LENGTH))
            toward_center = np.array([cup_x, cup_y]) - top
            toward_center = toward_center / max(np.linalg.norm(toward_center), 1e-6)
            drop = 0.007  # grab 7 mm below the top
            if top_z - drop - 0.005 < rim_top + 0.004:
                continue  # the fingertips would come down onto the cup rim
            grasp = top + toward_center * drop * math.tan(lean)
            candidates.append({"x": float(grasp[0]), "y": float(grasp[1]), "z": top_z - drop, "top_z": top_z,
                               "top": top, "toward_center": toward_center, "lean": lean})

        # For the finger-room check: every other stick tall enough to meet the descending fingertips.
        for cand in candidates:
            tall = valid & (height > cand["z"] - 0.006)
            px = world_x[tall]
            py = world_y[tall]
            # This stick's own direction, measured from the pixels just below its top (not assumed).
            a = cand["top"]
            near_top = np.hypot(px - a[0], py - a[1]) < 0.006
            if near_top.sum() >= 6:
                pts = np.vstack([px[near_top] - a[0], py[near_top] - a[1]])
                w, v = np.linalg.eigh(pts @ pts.T)
                direction = v[:, -1]
                if direction @ pts.mean(axis=1) < 0:
                    direction = -direction  # point from the top down along the stick
                cand["stick_dir"] = direction
            else:
                cand["stick_dir"] = cand["toward_center"]
            # remove this stick's own pixels (the segment from its top 15 mm down along the stick)
            b = a + cand["stick_dir"] * 0.015 * math.tan(cand["lean"]) + 1e-6
            ab = b - a
            t = np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / (ab @ ab), 0.0, 1.0)
            dist = np.hypot(px - (a[0] + t * ab[0]), py - (a[1] + t * ab[1]))
            own = dist < 0.0016
            cand["others_x"] = px[own == False]
            cand["others_y"] = py[own == False]
            cand["clearance"] = float(np.min(np.hypot(cand["others_x"] - cand["x"], cand["others_y"] - cand["y"]))) \
                if len(cand["others_x"]) > 0 else 1.0
        candidates.sort(key=lambda cand: -cand["clearance"])
        return candidates

    def toothpick_bottoms(self):
        return np.array([self.data.xpos[b][2] for b in self.tool_body_ids])

    def gripped_toothpick(self):
        """Which toothpick (if any) touches both fingertips."""
        touching_fixed = set()
        touching_moving = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                body = int(self.model.geom_bodyid[b])
                if body in self.tool_body_ids:
                    if a in self.fixed_finger:
                        touching_fixed.add(body)
                    if a in self.moving_finger:
                        touching_moving.add(body)
        both = touching_fixed & touching_moving
        return list(both)

    def pick_from_rack(self, max_attempts=3):
        """Printed rack: every toothpick stands in its own hole at a known spot. Take the next one."""
        self.set_yaw(TOOL_YAW)  # jaw opens away from the robot; the fingertip behind and the hook clear the other row
        top_z = pw.BENCH_Z + 0.0002 + sw.TOOTHPICK_LENGTH
        for attempt in range(1, max_attempts + 1):
            index = (self.slot + attempt - 1) % sw.N_TOOLS
            body = self.tool_body_ids[index]
            x, y = sw.rack_slot_xy(index)
            grasp_z = top_z - 0.007
            grip_xy = np.array([x - self.toothpick_aim, y])
            self.target_body = body
            self.track_bumps = True
            bottoms_before = self.toothpick_bottoms()
            self.travel(grip_xy, grasp_z + 0.03, JAW_TOOTHPICK_OPEN)
            self.move_grip_to([grip_xy[0], grip_xy[1], grasp_z], JAW_TOOTHPICK_OPEN, duration=0.8)
            if self.close_gently() == False or self.locked_body != body:
                self.say(f"rack hole {index + 1}: fingers did not close on that toothpick, trying the next hole")
                self.unlock_all()
                self.set_jaw(JAW_TOOTHPICK_OPEN)
                self.move_grip_to([grip_xy[0], grip_xy[1], grasp_z + 0.03], JAW_TOOTHPICK_OPEN, duration=0.6)
                continue
            # pull straight up until the bottom clears the rack (holes are 2.5 cm deep)
            self.move_grip_to([grip_xy[0], grip_xy[1], grasp_z + sw.RACK_DEPTH + 0.01], self.data.ctrl[5], duration=1.5)
            bottoms_after = self.toothpick_bottoms()
            extra = [i for i in range(sw.N_TOOLS) if i != index and bottoms_after[i] - bottoms_before[i] > 0.005]
            self.holding_tool = True
            self.move_grip_to([grip_xy[0], grip_xy[1], SAFE_Z_TOOL], self.data.ctrl[5], duration=0.8)
            self.track_bumps = False
            if len(extra) > 0:
                self.say(f"rack hole {index + 1}: pulled {len(extra)} neighbor(s) along, putting back")
            self.tool_body_name = f"tool_{index}"
            self.tool_geoms = set(g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body)
            self.tool_tip_site = self.model.site(f"tool_tip_site_{index}").id
            self.wait(0.3)
            self.tip_offset = self.tip_pos() - self.get_grip_pos()
            axis = self.data.xmat[body].reshape(3, 3)[:, 2]
            tilt = math.degrees(math.acos(min(1.0, abs(axis[2]))))
            self.pick_info = {"pick_attempts": attempt, "rack_hole": index + 1, "rack_neighbors_bumped": len(self.bumped),
                              "toothpick_tilt_deg": round(tilt, 1)}
            self.say(f"took the toothpick from rack hole {index + 1} (tilt {tilt:.1f} deg, neighbors bumped: {len(self.bumped)})")
            return
        self.track_bumps = False
        self.pick_info = {"pick_attempts": max_attempts, "rack_hole": None, "rack_neighbors_bumped": len(self.bumped),
                          "toothpick_tilt_deg": None}
        raise PickFailed("could not take a toothpick from the rack")

    def pick_up_tool(self, max_attempts=14):
        if self.holder == "rack":
            return self.pick_from_rack()
        self.set_yaw(TOOL_YAW)
        shake_round = 0
        for attempt in range(1, max_attempts + 1):
            self.move_arm(self.park_arm, JAW_TOOTHPICK_OPEN, duration=1.2)
            self.wait(0.3)
            candidates = self.find_separated_toothpick()
            # Try every candidate with 12 wrist rotations; take the first where the fingertips fit.
            choice = None
            for c in candidates:
                # Turn the wrist so the jaw opens toward the cup center. Then the jaw closes along the stick's lean,
                # and the lid hook (which hangs 5.7 cm behind the fingertips) stays outside the cup.
                center_dir = math.atan2(c["toward_center"][1], c["toward_center"][0])
                yaws = [center_dir + math.radians(k) for k in (0, 15, -15, 30, -30, 45, -45)]
                for yaw in yaws:
                    if self.fingers_fit(c, yaw) == False:
                        continue
                    direction = np.array([math.cos(yaw), math.sin(yaw)])
                    grip_xy = np.array([c["x"], c["y"]]) - self.toothpick_aim * direction
                    _, err = self.solve_ik([grip_xy[0], grip_xy[1], c["z"]], target_quat=quat_for_yaw(yaw))
                    if err < 1e-3:
                        choice = (c, yaw, grip_xy)
                        break
                if choice is not None:
                    break
            if choice is None:
                self.say(f"attempt {attempt}: {len(candidates)} toothpick tops seen, none with room for the fingers")
                shake_round = shake_round + 1
                self.shake_cup(shake_round)
                continue
            target, yaw, grip_xy = choice
            self.say(f"attempt {attempt}: found a toothpick the fingers fit around (wrist turned {math.degrees(yaw):.0f} deg)")

            bottoms_before = self.toothpick_bottoms()
            self.set_yaw(yaw)
            target = {"x": grip_xy[0], "y": grip_xy[1], "z": target["z"]}
            self.travel([target["x"], target["y"]], target["z"] + 0.03, JAW_TOOTHPICK_OPEN)
            self.move_grip_to([target["x"], target["y"], target["z"]], JAW_TOOTHPICK_OPEN, duration=0.8)
            self.close_gently()
            held = self.gripped_toothpick()
            if len(held) == 0:
                self.say(f"attempt {attempt}: fingers closed on nothing, retrying")
                self.set_jaw(JAW_TOOTHPICK_OPEN)
                self.set_yaw(TOOL_YAW)
                self.move_grip_to([target["x"], target["y"], target["z"] + 0.04], JAW_TOOTHPICK_OPEN, duration=0.8)
                continue

            # Pull straight up, slowly, so neighbors are not dragged along.
            self.move_grip_to([target["x"], target["y"], target["z"] + 0.055], self.data.ctrl[5], duration=2.0)
            self.wait(0.3)
            held = [self.locked_body]
            bottoms_after = self.toothpick_bottoms()
            lifted = [i for i in range(sw.N_TOOLS) if bottoms_after[i] - bottoms_before[i] > 0.02]
            held_index = [self.tool_body_ids.index(b) for b in held]
            extra = [i for i in lifted if i not in held_index]
            if len(held) != 1 or len(extra) > 0:
                self.say(f"attempt {attempt}: pulled {len(held)} held + {len(extra)} extra toothpicks, putting back")
                self.unlock_all()
                self.set_jaw(JAW_TOOTHPICK_OPEN)
                self.set_yaw(TOOL_YAW)
                self.wait(0.5)
                continue

            # Rise to the toothpick travel height, then turn the wrist back to the standard rotation.
            self.holding_tool = True
            self.move_grip_to([target["x"], target["y"], SAFE_Z_TOOL], self.data.ctrl[5], duration=1.0)
            self.set_yaw(TOOL_YAW)
            self.move_grip_to([target["x"], target["y"], SAFE_Z_TOOL], self.data.ctrl[5], duration=1.0)
            self.wait(0.3)

            index = held_index[0]
            self.tool_body_name = f"tool_{index}"
            body = self.tool_body_ids[index]
            self.tool_geoms = set(g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body)
            self.tool_tip_site = self.model.site(f"tool_tip_site_{index}").id
            # A fixed "tip camera" measures where the tip really is, relative to where the robot thinks it is.
            self.tip_offset = self.tip_pos() - self.get_grip_pos()
            axis = self.data.xmat[body].reshape(3, 3)[:, 2]
            tilt = math.degrees(math.acos(min(1.0, abs(axis[2]))))
            self.pick_info = {"pick_attempts": attempt, "cup_shakes": self.shakes, "toothpick_tilt_deg": round(tilt, 1)}
            self.say(f"picked one toothpick on attempt {attempt} (tilt in fingers {tilt:.1f} deg)")
            return
        self.pick_info = {"pick_attempts": max_attempts, "cup_shakes": self.shakes, "toothpick_tilt_deg": None}
        raise PickFailed("could not pick a single toothpick")

    def close_gently(self):
        """Close slowly. The moment both fingertips touch the same toothpick, stop and let the fingertip groove
        hold it (modeled as a lock, like the Panda's tool adapter). Squeezing harder would push a tilted stick out."""
        angle = self.data.ctrl[5]
        while angle > JAW_CLOSED:
            angle = angle - 0.003
            self.data.ctrl[5] = angle
            self.wait(0.02)
            held = self.gripped_toothpick()
            if len(held) > 0:
                self.data.ctrl[5] = self.data.qpos[5] - 0.001
                self.lock_toothpick(held[0], True)
                self.wait(0.1)
                return True
        return False

    def lock_toothpick(self, body, on):
        index = self.tool_body_ids.index(body)
        weld = self.model.equality(f"tool_lock_{index}").id
        if on == True:
            g = self.data.body("gripper")
            t = self.data.xpos[body]
            R = g.xmat.reshape(3, 3)
            conj = np.zeros(4)
            mujoco.mju_negQuat(conj, g.xquat)
            rel_quat = np.zeros(4)
            mujoco.mju_mulQuat(rel_quat, conj, self.data.xquat[body])
            self.model.eq_data[weld, 0:3] = 0.0
            self.model.eq_data[weld, 3:6] = R.T @ (t - g.xpos)
            self.model.eq_data[weld, 6:10] = rel_quat
            self.model.eq_data[weld, 10] = 1.0
            self.locked_body = body
        self.data.eq_active[weld] = 1 if on == True else 0

    def unlock_all(self):
        for b in self.tool_body_ids:
            self.lock_toothpick(b, False)

    def drop_tool_in_bin(self):
        bx, by = self.bin_xy
        self.travel_tip([bx, by], pw.BENCH_Z + 0.03)  # lower the toothpick well into the 8 cm waste cup before letting go
        self.unlock_all()
        self.holding_tool = False
        self.set_jaw(JAW_RELEASE)
        self.wait(0.6)
        pos = self.tip_pos()
        in_bin = abs(pos[0] - bx) < 0.026 and abs(pos[1] - by) < 0.026  # inside the walls (stick radius included)
        self.say("dropped toothpick in waste bin" if in_bin == True else "toothpick missed the waste bin")

    def tool_slip_mm(self):
        """How far the toothpick has slipped in the fingers since it was picked (truth vs. what the robot assumes)."""
        if self.tip_offset is None:
            return None
        return float(np.linalg.norm(self.tip_pos() - (self.get_grip_pos() + self.tip_offset)) * 1000)

    # ---------- look before touching ----------

    def see_tip_near(self, xy, agar_guess, size=640):
        """Plate camera (with depth): find the lowest point of the toothpick near a spot. Returns its xy or None."""
        renderer = mujoco.Renderer(self.model, height=size, width=size)
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera="source_cam")
        depth = renderer.render().copy()
        renderer.close()
        cam_z = pw.BENCH_Z + pw.CAMERA_HEIGHT
        height = cam_z - depth
        scale = 2 * math.tan(math.radians(pw.CAMERA_FOVY / 2)) / size
        cols, rows = np.meshgrid(np.arange(size), np.arange(size))
        wx = self.source_xy[0] + (cols + 0.5 - size / 2) * depth * scale
        wy = self.source_xy[1] - (rows + 0.5 - size / 2) * depth * scale
        near = np.hypot(wx - xy[0], wy - xy[1]) < 0.012
        agar_pixels = height[near & (np.abs(height - agar_guess) < 0.003)]
        agar = float(np.median(agar_pixels)) if len(agar_pixels) > 20 else agar_guess  # agar hidden: keep the guess
        stick = near & (height > agar + 0.0008) & (height < agar + 0.015)
        if stick.sum() < 5:
            return None, agar
        lowest = height[stick].min()
        tip = stick & (height < lowest + 0.0006)
        return np.array([wx[tip].mean(), wy[tip].mean()]), agar

    def pick_colony(self, colony):
        start_z = pw.BENCH_Z + pw.PLATE_WALL_TOP + 0.002
        target = np.array([colony["x"], colony["y"]])
        agar_guess = pw.BENCH_Z + pw.PLATE_FLOOR + (pw.AGAR_MIN + pw.AGAR_MAX) / 2
        self.travel_tip(target, start_z + 0.03)
        aim = target.copy()
        self.camera_corrections = []
        for _ in range(2):
            self.move_tip_to([aim[0], aim[1], agar_guess + 0.0045], duration=0.8)
            seen, agar_guess = self.see_tip_near(target, agar_guess)
            if seen is None:
                break
            correction = target - seen
            self.camera_corrections.append(round(float(np.linalg.norm(correction)) * 1000, 2))
            if np.linalg.norm(correction) < 0.0002:
                break
            aim = aim + correction
        surface = self.touch_down((aim[0], aim[1]), agar_guess + 0.0045,
                                  lambda: self.contact(self.tool_geoms, self.agar_geom["source"]), use_tip=True)
        if surface is None:
            return None, None
        touch_xy = self.tip_pos()[:2]
        self.wait(0.3)  # dwell so cells stick to the toothpick
        self.move_tip_to([aim[0], aim[1], start_z + 0.03], duration=0.8)
        return touch_xy, surface

    # ---------- whole run ----------

    def run(self):
        self.calibrate()
        try:
            report, detections, chosen, samples, image = super().run()
        except PickFailed:
            report = {"tool": self.tool_name, "seed": self.truth["seed"], "success": False, "picked": False}
            return {**report, **self.pick_info, **self.calibration_info}, [], None, [], None
        report.pop("rack_slot", None)
        report.update(self.pick_info)
        report["camera_corrections_mm"] = getattr(self, "camera_corrections", None)
        report["calibration_error_deg"] = [round(float(math.degrees(v)), 2) for v in self.calibration_offset]
        report.update(self.calibration_info)
        return report, detections, chosen, samples, image

    def streak(self, surface_guess):
        first = self.streak_path()[0][0]
        self.travel_tip(first, pw.BENCH_Z + pw.PLATE_WALL_TOP + 0.012)
        samples, surface = super().streak(surface_guess)
        self.slip_after_streak_mm = self.tool_slip_mm()
        return samples, surface
