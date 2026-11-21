"""Helpers for the vial-rack lab simulation.

Everything the other scripts need lives here:
  - loading the scene
  - putting the robot in its home pose
  - inverse kinematics (IK): "what joint angles put the gripper at this spot?"
  - moving the arm smoothly and recording video frames
  - checking whether the vials ended up in the rack
"""

import os

import mujoco
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(HERE, "models", "lab_scene.xml")
OUTPUT_DIR = os.path.join(HERE, "outputs")

# Robot home pose (7 arm joints). Gripper points straight down.
HOME_ARM = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
# Parked pose: same as home but swung to the side so cameras can see the bench.
PARK_ARM = np.array([-1.6, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Gripper command: the 8th actuator goes from 0 (closed) to 255 (fully open, 4 cm per finger).
GRIPPER_OPEN = 255
GRIPPER_CLOSED = 0
GRIPPER_RELEASE = 120  # opens about 1.9 cm per finger, enough to let go without hitting neighbor vials

# Slot centers in the rack (x, y) in meters. See models/lab_scene.xml.
RACK_SLOTS = {
    "A1": (0.48, 0.11), "A2": (0.48, 0.15), "A3": (0.48, 0.19),
    "B1": (0.52, 0.11), "B2": (0.52, 0.15), "B3": (0.52, 0.19),
}
RACK_FLOOR_Z = 0.106  # top of the rack base plate

VIAL_HALF_HEIGHT = 0.035
GRASP_ABOVE_CENTER = 0.015  # grab the vial a little above its middle, under the cap
SAFE_Z = 0.30  # height to travel at so we never hit anything


class LabSim:
    def __init__(self, record_video=True, video_camera="lab_cam", video_size=(320, 240), video_fps=10, scene_path=SCENE_PATH):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)  # scratch copy so IK never disturbs the real sim

        self.grip_site = self.model.site("grip").id
        self.record_video = record_video
        self.video_camera = video_camera
        self.video_fps = video_fps
        self.frames = []
        self.video_renderer = None
        if self.record_video == True:
            self.video_renderer = mujoco.Renderer(self.model, height=video_size[1], width=video_size[0])
        self.last_frame_time = -1.0

        self.reset()
        # Orientation we want the gripper to hold the whole time: the "pointing down" orientation from home.
        self.down_quat = self.get_grip_quat(self.data)

    # ---------- basic state ----------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = HOME_ARM
        self.data.qpos[7:9] = 0.04  # fingers open
        self.data.ctrl[:7] = HOME_ARM
        self.data.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)
        # Let the vials settle onto the bench.
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

    def get_grip_quat(self, data):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self.grip_site])
        return quat

    def get_grip_pos(self):
        return self.data.site_xpos[self.grip_site].copy()

    def get_body_pos(self, body_name):
        return self.data.body(body_name).xpos.copy()

    def get_body_up_axis(self, body_name):
        # The body's local z axis expressed in world frame. (0, 0, 1) means standing upright.
        rotation = self.data.body(body_name).xmat.reshape(3, 3)
        return rotation[:, 2].copy()

    # ---------- inverse kinematics ----------

    def solve_ik(self, target_pos, start_arm=None, max_iters=300, tolerance=1e-4):
        """Find 7 arm joint angles that put the grip site at target_pos, pointing down.

        Uses damped least squares: repeatedly nudge the joints in the direction
        that shrinks the position and orientation error.
        """
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        if start_arm is None:
            d.qpos[:7] = self.data.qpos[:7]
        else:
            d.qpos[:7] = start_arm

        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        damping = 1e-3
        low = self.model.jnt_range[:7, 0]
        high = self.model.jnt_range[:7, 1]

        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)

            pos_error = np.array(target_pos) - d.site_xpos[self.grip_site]

            # orientation error as a rotation vector
            current_quat = self.get_grip_quat(d)
            current_conj = np.zeros(4)
            mujoco.mju_negQuat(current_conj, current_quat)
            error_quat = np.zeros(4)
            mujoco.mju_mulQuat(error_quat, self.down_quat, current_conj)
            rot_error = np.zeros(3)
            mujoco.mju_quat2Vel(rot_error, error_quat, 1.0)

            error = np.concatenate([pos_error, rot_error])
            if np.linalg.norm(error) < tolerance:
                break

            mujoco.mj_jacSite(self.model, d, jac_pos, jac_rot, self.grip_site)
            J = np.vstack([jac_pos[:, :7], jac_rot[:, :7]])
            step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), error)
            d.qpos[:7] = np.clip(d.qpos[:7] + step, low, high)

        return d.qpos[:7].copy(), np.linalg.norm(error)

    # ---------- motion ----------

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.record_video == True:
            if self.data.time - self.last_frame_time >= 1.0 / self.video_fps:
                self.video_renderer.update_scene(self.data, camera=self.video_camera)
                self.frames.append(self.video_renderer.render().copy())
                self.last_frame_time = self.data.time

    def move_arm(self, target_arm, gripper, duration=1.5, settle=0.3):
        """Smoothly move the joint targets from where they are now to target_arm."""
        start_arm = self.data.ctrl[:7].copy()
        start_gripper = self.data.ctrl[7]
        n_steps = int(duration / self.model.opt.timestep)
        for i in range(n_steps):
            t = (i + 1) / n_steps
            s = t * t * (3 - 2 * t)  # smoothstep: slow start, slow stop
            self.data.ctrl[:7] = start_arm + s * (target_arm - start_arm)
            self.data.ctrl[7] = start_gripper + s * (gripper - start_gripper)
            self.step()
        for _ in range(int(settle / self.model.opt.timestep)):
            self.step()

    def move_grip_to(self, target_pos, gripper, duration=1.5):
        target_arm, error = self.solve_ik(target_pos, start_arm=self.data.ctrl[:7])
        if error > 1e-2:
            print(f"  warning: IK could not reach {np.round(target_pos, 3)} (error {error:.4f})")
        self.move_arm(target_arm, gripper, duration)

    def set_gripper(self, gripper, duration=0.6):
        self.move_arm(self.data.ctrl[:7].copy(), gripper, duration, settle=0.2)

    # ---------- the task ----------

    def pick_and_place(self, vial_name, slot_name):
        vial = self.get_body_pos(vial_name)
        slot_x, slot_y = RACK_SLOTS[slot_name]
        grasp_z = vial[2] + GRASP_ABOVE_CENTER
        # When released, the vial bottom should hang about 8 mm above the rack floor.
        place_z = RACK_FLOOR_Z + 0.008 + VIAL_HALF_HEIGHT + GRASP_ABOVE_CENTER

        print(f"Moving {vial_name} -> slot {slot_name}")
        self.move_grip_to([vial[0], vial[1], SAFE_Z], GRIPPER_OPEN)
        self.move_grip_to([vial[0], vial[1], grasp_z], GRIPPER_OPEN, duration=1.2)
        self.set_gripper(GRIPPER_CLOSED)
        self.move_grip_to([vial[0], vial[1], SAFE_Z], GRIPPER_CLOSED, duration=1.2)
        self.move_grip_to([slot_x, slot_y, SAFE_Z], GRIPPER_CLOSED)
        self.move_grip_to([slot_x, slot_y, place_z], GRIPPER_CLOSED, duration=1.2)
        self.set_gripper(GRIPPER_RELEASE)
        self.move_grip_to([slot_x, slot_y, SAFE_Z], GRIPPER_RELEASE, duration=1.2)

    def check_vial_in_slot(self, vial_name, slot_name):
        """A vial counts as placed if it is upright, centered in the slot, and down inside the rack."""
        pos = self.get_body_pos(vial_name)
        up = self.get_body_up_axis(vial_name)
        slot_x, slot_y = RACK_SLOTS[slot_name]
        xy_offset = np.hypot(pos[0] - slot_x, pos[1] - slot_y)
        bottom_z = pos[2] - VIAL_HALF_HEIGHT

        upright = up[2] > 0.95
        centered = xy_offset < 0.012
        seated = bottom_z < RACK_FLOOR_Z + 0.01
        success = upright == True and centered == True and seated == True
        return success, {"xy_offset_mm": round(float(xy_offset) * 1000, 1), "upright": bool(upright), "seated": bool(seated)}

    # ---------- images ----------

    def save_image(self, filename, camera="lab_cam", width=1280, height=960):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera)
        path = os.path.join(OUTPUT_DIR, filename)
        Image.fromarray(renderer.render()).save(path)
        renderer.close()
        return path

    def save_video(self, filename):
        if len(self.frames) == 0:
            return None
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        images = [Image.fromarray(f) for f in self.frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / self.video_fps), loop=0)
        return path
