"""Scripted baseline: move three vials from the bench into the rack.

This is the "hand-written" version. Later, Gemini will look at the camera
image and write this kind of plan itself, and we will compare success rates.

Run:  ~/.venvs/mujoco-lab/bin/python pick_and_place.py
"""

from lab_sim import GRIPPER_OPEN, PARK_ARM, LabSim

# Which vial goes in which slot
PLAN = [
    ("vial_red", "A1"),
    ("vial_blue", "A2"),
    ("vial_green", "A3"),
]


def main():
    sim = LabSim(record_video=True)

    print("Saved", sim.save_image("before_lab_cam.png", camera="lab_cam"))
    print("Saved", sim.save_image("before_top_cam.png", camera="top_cam"))

    for vial_name, slot_name in PLAN:
        sim.pick_and_place(vial_name, slot_name)

    sim.move_arm(PARK_ARM, GRIPPER_OPEN)  # swing out of the way so the cameras can see

    print("Saved", sim.save_image("after_lab_cam.png", camera="lab_cam"))
    print("Saved", sim.save_image("after_top_cam.png", camera="top_cam"))
    print("Saved", sim.save_video("pick_and_place.gif"))

    print()
    print("Results")
    placed = 0
    for vial_name, slot_name in PLAN:
        success, details = sim.check_vial_in_slot(vial_name, slot_name)
        if success == True:
            placed = placed + 1
        print(f"  {vial_name} in {slot_name}: {'OK' if success == True else 'FAILED'} {details}")
    print(f"{placed}/{len(PLAN)} vials placed, sim time {sim.data.time:.1f} s")


if __name__ == "__main__":
    main()
