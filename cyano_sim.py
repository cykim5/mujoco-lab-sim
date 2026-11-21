"""Cyanobacteria colony picking and quadrant restreaking, plate to plate.

Workflow the robot runs:
  1. Take the lids off both plates (suction at the fingertips) and park them.
  2. Photograph the source plate from above and find colonies (green dots).
  3. Choose a well-isolated colony.
  4. Pick up the next tool (plastic loop or wooden toothpick) from a 24-slot rack.
  5. Lower the tool until it touches the agar (this also finds the agar height) on the chosen colony.
  6. Quadrant streak onto the fresh plate, pressing lightly so it never digs in.
  7. Drop the tool in the waste bin and put both lids back.

What the sim can tell you: whether the motion, vision and contact control work, and how long it takes.
What it cannot: whether cells actually transfer and grow. That needs real plates.
"""

import math
import os
from collections import deque

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import plate_world as pw
from lab_sim import GRIPPER_CLOSED, GRIPPER_OPEN, OUTPUT_DIR, PARK_ARM, LabSim

MAX_SINGLE_COLONY_R = 0.0011  # a blob bigger than this (2.2 mm across) is probably several colonies touching
TOUCH_FORCE = 0.003  # newtons. Anything above this counts as "touching".
PRESS_DEPTH = 0.0003  # while streaking, aim 0.3 mm below the agar surface for light, steady contact
GOUGE_DEPTH = 0.001  # pressing more than 1 mm into the agar counts as digging in
STREAK_SPEED = 0.02  # 2 cm/s, about the speed a person streaks
STREAK_RADIUS = 0.036  # stay this far inside the plate center when streaking
STROKES_PER_QUADRANT = 6


class CyanoSim(LabSim):
    def __init__(self, tool_name, seed=0, record_video=True, slot=None, pcr=False, sectors=False):
        scene_path, self.truth = pw.build_plate_scene(tool_name, seed, pcr=pcr, sectors=sectors)
        self.set_panda_layout()
        # Which of the 24 rack slots to take the tool from. Different seeds use different slots.
        self.slot = (seed * 7) % (pw.RACK_ROWS * pw.RACK_COLS) if slot is None else slot
        self.tool_body_name = f"tool_{self.slot}"
        self.tool_name = tool_name
        self.tool = pw.TOOLS[tool_name]
        # Video: one frame every 0.25 s of sim time, played back at 2x speed.
        super().__init__(record_video=record_video, video_camera="close_cam", video_size=(320, 240),
                         video_fps=4, scene_path=scene_path)
        os.remove(scene_path)  # the generated file is only needed while loading

        m = self.model
        self.tool_tip_site = m.site(f"tool_tip_site_{self.slot}").id
        tool_body = m.body(self.tool_body_name).id
        self.tool_geoms = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == tool_body)
        finger_bodies = [m.body("left_finger").id, m.body("right_finger").id]
        self.finger_geoms = set(g for g in range(m.ngeom) if m.geom_bodyid[g] in finger_bodies)
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

        # To check the robot never bumps a neighbor in the rack: any touch between the robot (or the
        # tool it holds) and another tool counts.
        robot_bodies = set(m.body(name).id for name in ["link6", "link7", "hand", "left_finger", "right_finger"])
        self.robot_geoms = set(g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies) | self.tool_geoms
        self.other_tool_geoms = {}
        for i in range(pw.RACK_ROWS * pw.RACK_COLS):
            if i != self.slot:
                body = m.body(f"tool_{i}").id
                self.other_tool_geoms[i] = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == body)
        self.all_other_tool_geoms = set().union(*self.other_tool_geoms.values())
        self.bumped = set()

        self.tip_offset = None  # tool tip position relative to the grip point, measured after grasping
        self.hit_wall = False
        self.log = []

    def select_slot(self, slot):
        """Switch to another rack slot (for runs that use several tools, like colony PCR)."""
        m = self.model
        self.slot = slot
        self.tool_body_name = f"tool_{slot}"
        self.tool_tip_site = m.site(f"tool_tip_site_{slot}").id
        body = m.body(self.tool_body_name).id
        self.tool_geoms = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == body)
        robot_bodies = set(m.body(name).id for name in ["link6", "link7", "hand", "left_finger", "right_finger"])
        self.robot_geoms = set(g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies) | self.tool_geoms
        self.other_tool_geoms = {}
        for i in range(pw.RACK_ROWS * pw.RACK_COLS):
            if i != slot:
                b = m.body(f"tool_{i}").id
                self.other_tool_geoms[i] = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == b)
        self.all_other_tool_geoms = set().union(*self.other_tool_geoms.values())
        self.tip_offset = None

    def set_panda_layout(self):
        self.source_xy = pw.SOURCE_XY
        self.dest_xy = pw.DEST_XY
        self.lid_park_xy = pw.LID_PARK_XY
        self.bin_xy = pw.BIN_XY
        self.park_arm = PARK_ARM
        self.gripper_open = GRIPPER_OPEN
        self.gripper_closed = GRIPPER_CLOSED
        self.check_rack_bumps = True

    # ---------- sensing ----------

    def robot_tip_estimate(self):
        """Where the robot believes the tool tip is: its own kinematics plus the measured tip offset."""
        return self.get_grip_pos() + self.tip_offset

    def contact(self, geoms_a, geoms_b):
        """Total normal force (N) and deepest penetration (m) between two groups of geoms."""
        force = 0.0
        depth = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair_ab = c.geom1 in geoms_a and c.geom2 in geoms_b
            pair_ba = c.geom2 in geoms_a and c.geom1 in geoms_b
            if pair_ab == True or pair_ba == True:
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                force = force + abs(f6[0])
                depth = max(depth, -c.dist)
        return force, depth

    def tip_pos(self):
        return self.data.site_xpos[self.tool_tip_site].copy()

    def step(self):
        super().step()
        for i in range(self.data.ncon if self.check_rack_bumps == True else 0):
            c = self.data.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.robot_geoms and b in self.all_other_tool_geoms:
                    self.bumped.add(int(self.model.geom_bodyid[b]))
        if self.hit_wall == False:
            wall_force, _ = self.contact(self.tool_geoms, self.wall_geoms)
            if wall_force > 0.01:
                self.hit_wall = True

    def say(self, text):
        print(f"  [{self.data.time:6.1f} s] {text}")
        self.log.append(text)

    # ---------- small motions ----------

    def wait(self, seconds):
        for _ in range(int(seconds / self.model.opt.timestep)):
            self.step()

    def set_suction(self, on):
        value = 1.0 if on == True else 0.0
        self.data.ctrl[8] = value
        self.data.ctrl[9] = value

    def move_tip_to(self, target, duration=1.0):
        target = np.array(target)
        self.move_grip_to(target - self.tip_offset, self.data.ctrl[self.gripper_index], duration)
        # One correction: the robot reads its joint encoders, sees how far off it landed, and nudges.
        error = target - (self.get_grip_pos() + self.tip_offset)
        if np.linalg.norm(error) > 0.0001 and duration > 0.1:
            self.move_grip_to(target - self.tip_offset + error, self.data.ctrl[self.gripper_index], duration=0.2)

    def touch_down(self, xy, start_z, contact_check, use_tip, step=0.0003, max_steps=80):
        """Lower in small steps until contact_check() reports a force. Returns the contact height or None."""
        for i in range(max_steps):
            z = start_z - i * step
            if use_tip == True:
                self.move_tip_to([xy[0], xy[1], z], duration=0.03)
            else:
                self.move_grip_to([xy[0], xy[1], z], self.data.ctrl[self.gripper_index], duration=0.03)
            self.wait(0.02)
            force, _ = contact_check()
            if force > TOUCH_FORCE:
                if use_tip == True:
                    return self.robot_tip_estimate()[2]
                return z
        return None

    # ---------- lids ----------

    def lift_lid(self, plate_name, from_xy, to_xy, put_back):
        lid_body = "lid_" + plate_name
        lid_top = self.get_body_pos(lid_body)[2] + pw.LID_TOP_THICKNESS
        self.move_grip_to([from_xy[0], from_xy[1], lid_top + 0.04], GRIPPER_CLOSED, duration=1.2)
        z = self.touch_down(from_xy, lid_top + 0.015, lambda: self.contact(self.finger_geoms, self.lid_geoms[plate_name]),
                            use_tip=False, step=0.0005)
        if z is None:
            self.say(f"could not find the {plate_name} lid")
            return False
        self.move_grip_to([from_xy[0], from_xy[1], z - 0.001], GRIPPER_CLOSED, duration=0.2)
        self.set_suction(True)
        self.wait(0.3)
        grip_minus_lid = self.get_grip_pos()[2] - self.get_body_pos(lid_body)[2]
        self.move_grip_to([from_xy[0], from_xy[1], z + 0.05], GRIPPER_CLOSED, duration=0.8)
        if put_back == True:
            lid_z = pw.BENCH_Z + pw.PLATE_WALL_TOP + 0.001  # resting on the plate wall
        else:
            lid_z = pw.BENCH_Z + pw.LID_WALL_HEIGHT + 0.001  # resting on the bench
        self.move_grip_to([to_xy[0], to_xy[1], lid_z + grip_minus_lid + 0.04], GRIPPER_CLOSED, duration=1.2)
        self.move_grip_to([to_xy[0], to_xy[1], lid_z + grip_minus_lid], GRIPPER_CLOSED, duration=0.6)
        self.set_suction(False)
        self.wait(0.3)
        self.move_grip_to([to_xy[0], to_xy[1], lid_z + grip_minus_lid + 0.05], GRIPPER_CLOSED, duration=0.6)
        return True

    # ---------- vision ----------

    def find_colonies(self, size=640):
        """Photograph the source plate and find green dots. Returns detections in world coordinates."""
        renderer = mujoco.Renderer(self.model, height=size, width=size)
        renderer.update_scene(self.data, camera="source_cam")
        image = renderer.render().copy()
        renderer.close()

        rgb = image.astype(np.int16)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mask = (g - r > 25) & (g - b > 25)

        # Group touching green pixels into blobs (flood fill).
        seen = np.zeros_like(mask)
        blobs = []
        rows, cols = np.nonzero(mask)
        for start in zip(rows, cols):
            if seen[start] == True:
                continue
            queue = deque([start])
            seen[start] = True
            pixels = []
            while len(queue) > 0:
                y, x = queue.popleft()
                pixels.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < size and 0 <= nx < size and mask[ny, nx] == True and seen[ny, nx] == False:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) >= 4:
                blobs.append(np.array(pixels))

        # Pixel -> meters. The camera looks straight down from a known height.
        agar_guess = pw.BENCH_Z + pw.PLATE_FLOOR + (pw.AGAR_MIN + pw.AGAR_MAX) / 2
        distance = pw.BENCH_Z + pw.CAMERA_HEIGHT - agar_guess
        meters_per_pixel = 2 * distance * math.tan(math.radians(pw.CAMERA_FOVY / 2)) / size
        cx, cy = self.source_xy
        detections = []
        for pixels in blobs:
            row, col = pixels.mean(axis=0)
            x = cx + (col + 0.5 - size / 2) * meters_per_pixel
            y = cy - (row + 0.5 - size / 2) * meters_per_pixel
            radius = math.sqrt(len(pixels) / math.pi) * meters_per_pixel
            # Merged colonies show up as one blob that is stretched or too big. Mark round, single-sized blobs.
            spread = np.linalg.eigvalsh(np.cov(pixels.T.astype(float))) if len(pixels) >= 5 else np.array([1.0, 1.0])
            stretch = math.sqrt(max(spread[-1], 1e-9) / max(spread[0], 1e-9))
            single = stretch < 1.3 and radius < MAX_SINGLE_COLONY_R
            detections.append({"x": x, "y": y, "r": radius, "px": (col, row), "r_px": radius / meters_per_pixel,
                               "single": bool(single)})
        return detections, image, meters_per_pixel

    def choose_colony(self, detections):
        """Pick the colony with the most empty space around it (a clean, single colony)."""
        best = None
        best_gap = -1.0
        for d in detections:
            from_center = math.hypot(d["x"] - self.source_xy[0], d["y"] - self.source_xy[1])
            if d["r"] < 0.0004 or from_center > 0.032 or d.get("single", True) == False:
                continue
            gap = 1.0
            for other in detections:
                if other is d:
                    continue
                edge_to_edge = math.hypot(d["x"] - other["x"], d["y"] - other["y"]) - d["r"] - other["r"]
                gap = min(gap, edge_to_edge)
            d["gap"] = gap
            if gap > best_gap:
                best_gap = gap
                best = d
        return best

    def save_detection_image(self, image, detections, chosen, filename):
        picture = Image.fromarray(image)
        draw = ImageDraw.Draw(picture)
        for d in detections:
            col, row = d["px"]
            rad = d["r_px"] + 4
            color = (40, 90, 230) if d is not chosen else (230, 40, 40)
            width = 2 if d is not chosen else 4
            draw.ellipse([col - rad, row - rad, col + rad, row + rad], outline=color, width=width)
        path = f"{OUTPUT_DIR}/{filename}"
        picture.save(path)
        return path

    # ---------- tool ----------

    def lock_tool(self, on):
        """Tool adapter. Freezes the tool where it sits in the fingers right now (or releases it)."""
        weld = self.model.equality(f"tool_lock_{self.slot}").id
        if on == True:
            hand = self.data.body("hand")
            tool = self.data.body(self.tool_body_name)
            hand_rot = hand.xmat.reshape(3, 3)
            rel_pos = hand_rot.T @ (tool.xpos - hand.xpos)
            hand_conj = np.zeros(4)
            mujoco.mju_negQuat(hand_conj, hand.xquat)
            rel_quat = np.zeros(4)
            mujoco.mju_mulQuat(rel_quat, hand_conj, tool.xquat)
            self.model.eq_data[weld, 0:3] = 0.0  # anchor
            self.model.eq_data[weld, 3:6] = rel_pos
            self.model.eq_data[weld, 6:10] = rel_quat
            self.model.eq_data[weld, 10] = 1.0  # torque scale
        self.data.eq_active[weld] = 1 if on == True else 0

    def pick_up_tool(self):
        hx, hy = pw.rack_slot_xy(self.slot)
        top = self.get_body_pos(self.tool_body_name)[2] + self.tool["length"]
        # Open only 3 mm wider than the handle on each side, so the fingers fit between neighbors in the rack.
        narrow_open = int((self.tool["handle_half"] + 0.003) / 0.04 * 255)
        self.move_grip_to([hx, hy, top + 0.06], narrow_open, duration=1.5)
        self.move_grip_to([hx, hy, top - 0.012], narrow_open, duration=0.8)
        # Close the fingers just to the handle's width (255 = 4 cm per finger), so they hold it without fighting the lock.
        hold = int(self.tool["handle_half"] / 0.04 * 255)
        self.set_gripper(hold)
        self.wait(0.2)
        self.lock_tool(True)
        # Lift straight up by the whole tool length (plus 3 cm) so the tip clears the neighbors before moving sideways.
        self.move_grip_to([hx, hy, top + self.tool["length"] + 0.03], hold, duration=1.5)
        self.wait(0.3)
        # Measure where the tip sits relative to the gripper, once, in free air.
        # (A real cell would do this with a tip-finding camera or a touch plate.)
        self.tip_offset = self.tip_pos() - self.get_grip_pos()
        tilt = math.degrees(math.atan2(math.hypot(self.tip_offset[0], self.tip_offset[1]), -self.tip_offset[2]))
        self.say(f"picked up the {self.tool_name} from rack slot {self.slot + 1} of 24 (tilt {tilt:.1f} deg, "
                 f"neighbors bumped: {self.neighbors_bumped()})")

    def neighbors_bumped(self):
        """How many other tools in the rack the robot (or the tool it holds) touched."""
        return len(self.bumped)

    def drop_tool_in_bin(self):
        bx, by = self.bin_xy
        self.move_tip_to([bx, by, pw.BENCH_Z + 0.09], duration=1.5)
        self.lock_tool(False)
        self.set_gripper(GRIPPER_OPEN)
        self.wait(0.5)
        in_bin = abs(self.tip_pos()[0] - bx) < 0.03 and abs(self.tip_pos()[1] - by) < 0.03
        self.say("dropped tool in waste bin" if in_bin == True else "tool missed the waste bin")

    # ---------- pick and streak ----------

    def pick_colony(self, colony):
        start_z = pw.BENCH_Z + pw.PLATE_WALL_TOP + 0.002
        # Travel level, high above the rack, until over the colony, then go down.
        self.move_tip_to([colony["x"], colony["y"], self.tip_pos()[2]], duration=1.5)
        self.move_tip_to([colony["x"], colony["y"], start_z + 0.03], duration=1.0)
        surface = self.touch_down((colony["x"], colony["y"]), start_z,
                                  lambda: self.contact(self.tool_geoms, self.agar_geom["source"]), use_tip=True)
        if surface is None:
            return None, None
        touch_xy = self.tip_pos()[:2]
        self.wait(0.3)  # dwell so cells stick to the tool
        self.move_tip_to([colony["x"], colony["y"], start_z + 0.03], duration=0.8)
        return touch_xy, surface

    def streak_path(self):
        """Quadrant streak: in each quadrant, zigzag from the rim in toward the center.
        Each new quadrant starts inside the previous one (near the rim) and drags cells across."""
        cx, cy = self.dest_xy
        quadrants = []
        for k in range(4):
            angle = math.radians(45 + 90 * k)
            u = np.array([math.cos(angle), math.sin(angle)])  # outward
            v = np.array([-u[1], u[0]])  # sideways
            points = []
            if k > 0:
                prev = math.radians(45 + 90 * (k - 1) + 20)
                points.append(np.array([cx, cy]) + 0.030 * np.array([math.cos(prev), math.sin(prev)]))
            radii = np.linspace(0.034, 0.010, STROKES_PER_QUADRANT)  # rim first, then inward
            for i, rad in enumerate(radii):
                half = min(rad * 0.85, math.sqrt(max(STREAK_RADIUS ** 2 - rad ** 2, 0.0))) - 0.002
                side = 1 if i % 2 == 0 else -1
                points.append(np.array([cx, cy]) + rad * u - side * half * v)
                points.append(np.array([cx, cy]) + rad * u + side * half * v)
            # resample every 1 mm
            dense = [points[0]]
            for a, b in zip(points[:-1], points[1:]):
                n = max(1, int(np.linalg.norm(b - a) / 0.001))
                for j in range(1, n + 1):
                    dense.append(a + (b - a) * j / n)
            quadrants.append(dense)
        return quadrants

    def streak(self, surface_guess):
        start_z = pw.BENCH_Z + pw.PLATE_WALL_TOP + 0.002
        quadrants = self.streak_path()
        samples = []
        surface = None
        steps_per_point = int(0.001 / STREAK_SPEED / self.model.opt.timestep)
        for k, path in enumerate(quadrants):
            x0, y0 = path[0]
            self.move_tip_to([x0, y0, start_z + 0.01], duration=0.8)
            found = self.touch_down((x0, y0), start_z, lambda: self.contact(self.tool_geoms, self.agar_geom["dest"]),
                                    use_tip=True)
            if found is None:
                self.say(f"quadrant {k + 1}: never touched the agar")
                continue
            if surface is None:
                surface = found
                self.say(f"fresh plate agar found at {(surface - pw.BENCH_Z) * 1000:.1f} mm above the bench")
            arm = self.data.ctrl[:self.n_arm].copy()
            for p in path:
                target = np.array([p[0], p[1], surface - PRESS_DEPTH]) - self.tip_offset
                arm, _ = self.solve_ik(target, start_arm=arm, max_iters=30)
                self.data.ctrl[:self.n_arm] = arm
                force_max = 0.0
                depth_max = 0.0
                for _ in range(steps_per_point):
                    self.step()
                force, _ = self.contact(self.tool_geoms, self.agar_geom["dest"])
                force_max = max(force_max, force)
                # How far the tip went below the real agar surface (graded against the hidden truth).
                depth = self.truth["agar_top"]["dest"] - self.tip_pos()[2]
                depth_max = max(depth_max, depth)
                samples.append({"quadrant": k, "pos": self.tip_pos(), "force": force_max, "depth": depth_max})
            self.move_tip_to([path[-1][0], path[-1][1], start_z + 0.01], duration=0.5)
        return samples, surface

    # ---------- whole run ----------

    def run(self):
        report = {"tool": self.tool_name, "seed": self.truth["seed"], "rack_slot": self.slot + 1}
        self.hit_wall = False

        self.say("taking lids off")
        ok_source = self.lift_lid("source", self.source_xy, self.lid_park_xy["source"], put_back=False)
        ok_dest = self.lift_lid("dest", self.dest_xy, self.lid_park_xy["dest"], put_back=False)
        report["lids_off"] = ok_source == True and ok_dest == True
        self.move_arm(self.park_arm, self.gripper_closed, duration=1.5)

        detections, image, _ = self.find_colonies()
        chosen = self.choose_colony(detections)
        report["detection"] = self.grade_detection(detections)
        self.say(f"found {len(detections)} colonies ({len(self.truth['colonies'])} on the plate)")
        if chosen is None:
            self.say("no isolated colony to pick")
            report["success"] = False
            return report, detections, chosen, [], image
        self.say(f"chose a {chosen['r'] * 2000:.1f} mm colony with {chosen['gap'] * 1000:.1f} mm clear around it")

        self.pick_up_tool()
        touch_xy, source_surface = self.pick_colony(chosen)
        report.update(self.grade_pick(touch_xy, chosen))
        if source_surface is not None:
            report["source_agar_error_mm"] = round((source_surface - self.truth["agar_top"]["source"]) * 1000, 2)
        self.say(f"touched colony: {'yes' if report['picked'] == True else 'NO'}, "
                 f"{report['miss_mm']} mm from its center, clean single colony: {report['clean_pick']}")

        samples, dest_surface = self.streak(source_surface)
        report.update(self.grade_streak(samples))
        if dest_surface is not None:
            report["dest_agar_error_mm"] = round((dest_surface - self.truth["agar_top"]["dest"]) * 1000, 2)

        self.drop_tool_in_bin()
        self.say("putting lids back")
        back_dest = self.lift_lid("dest", self.lid_park_xy["dest"], self.dest_xy, put_back=True)
        back_source = self.lift_lid("source", self.lid_park_xy["source"], self.source_xy, put_back=True)
        report["lids_back"] = self.lids_seated() and back_dest == True and back_source == True
        self.move_arm(self.park_arm, self.gripper_open, duration=1.5)

        report["hit_plate_wall"] = self.hit_wall
        report["neighbors_bumped"] = self.neighbors_bumped()
        report["cycle_time_s"] = round(self.data.time, 1)
        report["success"] = (report["lids_off"] == True and report["picked"] == True and report["clean_pick"] == True
                             and report["streak_ok"] == True and report["lids_back"] == True
                             and report["hit_plate_wall"] == False and report["neighbors_bumped"] == 0)
        return report, detections, chosen, samples, image

    # ---------- grading against the hidden truth ----------

    def grade_detection(self, detections):
        found = 0
        errors = []
        for c in self.truth["colonies"]:
            nearest = min([math.hypot(d["x"] - c["x"], d["y"] - c["y"]) for d in detections], default=1.0)
            if nearest < max(c["r"], 0.0005):
                found = found + 1
                errors.append(nearest)
        mean_error = float(np.mean(errors)) * 1000 if len(errors) > 0 else None
        return {"found": found, "total": len(self.truth["colonies"]), "detections": len(detections),
                "mean_error_mm": None if mean_error is None else round(mean_error, 2)}

    def grade_pick(self, touch_xy, chosen):
        if touch_xy is None:
            return {"picked": False, "clean_pick": False, "miss_mm": None}
        colonies = self.truth["colonies"]
        target = min(colonies, key=lambda c: math.hypot(c["x"] - chosen["x"], c["y"] - chosen["y"]))
        tip_r = self.tool["tip_radius"]
        miss = math.hypot(touch_xy[0] - target["x"], touch_xy[1] - target["y"])
        picked = miss < target["r"] + tip_r
        touched_others = 0
        for c in colonies:
            if c is target:
                continue
            if math.hypot(touch_xy[0] - c["x"], touch_xy[1] - c["y"]) < c["r"] + tip_r:
                touched_others = touched_others + 1
        return {"picked": bool(picked), "clean_pick": bool(picked) and touched_others == 0,
                "miss_mm": round(miss * 1000, 2), "touched_other_colonies": touched_others}

    def grade_streak(self, samples):
        if len(samples) == 0:
            return {"streak_ok": False, "contact_pct": 0.0}
        in_contact = [s["force"] > 0.0005 for s in samples]
        contact_pct = 100.0 * sum(in_contact) / len(samples)
        max_depth = max(s["depth"] for s in samples)
        max_force = max(s["force"] for s in samples)
        per_quadrant = []
        for k in range(4):
            q = [s for s in samples if s["quadrant"] == k]
            touching = [s for s in q if s["force"] > 0.0005]
            per_quadrant.append(round(len(touching) / 1000, 3))  # meters of streak in contact (samples are 1 mm apart)
        streak_ok = contact_pct >= 90 and max_depth <= GOUGE_DEPTH and min(per_quadrant) > 0.1
        return {"streak_ok": bool(streak_ok), "contact_pct": round(contact_pct, 1),
                "max_depth_mm": round(max_depth * 1000, 2), "max_force_N": round(max_force, 3),
                "streak_length_m_per_quadrant": per_quadrant}

    def lids_seated(self):
        ok = True
        for name, xy in [("source", self.source_xy), ("dest", self.dest_xy)]:
            pos = self.get_body_pos("lid_" + name)
            off = math.hypot(pos[0] - xy[0], pos[1] - xy[1])
            if off > 0.002 or abs(pos[2] - (pw.BENCH_Z + pw.PLATE_WALL_TOP)) > 0.002:
                ok = False
        return ok

    # ---------- pictures ----------

    def save_video(self, filename):
        if len(self.frames) == 0:
            return None
        path = f"{OUTPUT_DIR}/{filename}"
        images = [Image.fromarray(f) for f in self.frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=125, loop=0)  # 2x speed
        return path

    def save_streak_image(self, samples, filename, size=800):
        renderer = mujoco.Renderer(self.model, height=size, width=size, max_geom=20000)
        renderer.update_scene(self.data, camera="dest_cam")
        scene = renderer.scene
        quadrant_sizes = [sum(1 for s in samples if s["quadrant"] == k) for k in range(4)]
        seen_in_quadrant = [0, 0, 0, 0]
        for s in samples:
            if scene.ngeom >= scene.maxgeom:
                break
            k = s["quadrant"]
            progress = seen_in_quadrant[k] / max(quadrant_sizes[k] - 1, 1)  # 0 at the start, 1 at the end
            seen_in_quadrant[k] = seen_in_quadrant[k] + 1
            touching = s["force"] > 0.0005
            if touching == True:
                rgba = np.array([0.55 - 0.5 * progress, 0.9 - 0.55 * progress, 0.55 - 0.45 * progress, 1.0])
            else:
                rgba = np.array([0.9, 0.1, 0.1, 1.0])
            pos = s["pos"].copy()
            pos[2] = pos[2] + 0.0004
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.0006, 0, 0]), pos, np.eye(3).flatten(), rgba.astype(np.float32))
            scene.ngeom = scene.ngeom + 1
        picture = Image.fromarray(renderer.render())
        renderer.close()

        # Label where each quadrant starts.
        distance = pw.CAMERA_HEIGHT - (self.truth["agar_top"]["dest"] - pw.BENCH_Z)
        meters_per_pixel = 2 * distance * math.tan(math.radians(pw.CAMERA_FOVY / 2)) / size
        draw = ImageDraw.Draw(picture)
        for k in range(4):
            first = next((s for s in samples if s["quadrant"] == k), None)
            if first is None:
                continue
            col = (first["pos"][0] - self.dest_xy[0]) / meters_per_pixel + size / 2
            row = size / 2 - (first["pos"][1] - self.dest_xy[1]) / meters_per_pixel
            draw.ellipse([col - 16, row - 16, col + 16, row + 16], fill=(255, 255, 255), outline=(30, 30, 30), width=2)
            draw.text((col, row), str(k + 1), fill=(30, 30, 30), anchor="mm", font_size=22)
        path = f"{OUTPUT_DIR}/{filename}"
        picture.save(path)
        return path
