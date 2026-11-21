"""Run one cyano colony pick + quadrant restreak, plate to plate.

Run:
  ~/.venvs/mujoco-lab/bin/python streak_plate.py                 (Panda, asks which tool)
  ~/.venvs/mujoco-lab/bin/python streak_plate.py --tool loop
  ~/.venvs/mujoco-lab/bin/python streak_plate.py --tool toothpick --seed 3
  ~/.venvs/mujoco-lab/bin/python streak_plate.py --robot so101    (SO-101 build version, toothpicks from a cup)
"""

import argparse
import json

from cyano_sim import CyanoSim

TOOL_MENU = {"1": "loop", "2": "toothpick"}


def ask_for_tool():
    print("Which tool should the robot use?")
    print("  1) plastic loop")
    print("  2) wooden toothpick")
    while True:
        answer = input("Enter 1 or 2: ").strip()
        if answer in TOOL_MENU:
            return TOOL_MENU[answer]


def main():
    parser = argparse.ArgumentParser(description="Simulate cyano colony picking and quadrant restreaking.")
    parser.add_argument("--tool", choices=["loop", "toothpick"], help="plastic loop or wooden toothpick")
    parser.add_argument("--seed", type=int, default=0, help="changes the colony layout and agar thickness")
    parser.add_argument("--no-video", action="store_true", help="skip the GIF (faster)")
    parser.add_argument("--robot", choices=["panda", "so101"], default="panda",
                        help="panda = lab-grade Reference version, so101 = cheap 3D-printed Build version")
    args = parser.parse_args()

    if args.robot == "so101":
        from so101_sim import So101CyanoSim
        if args.tool == "loop":
            print("The SO-101 version uses toothpicks only (a 12 cm loop is too tall for it to lift out of a cup).")
        tool = "toothpick"
        print(f"\nRunning the SO-101 build version with a toothpick, seed {args.seed}")
        sim = So101CyanoSim(tool, seed=args.seed, record_video=args.no_video == False)
        name = "so101_toothpick"
    else:
        tool = args.tool
        if tool is None:
            tool = ask_for_tool()
        print(f"\nRunning the Panda reference version with the {tool}, seed {args.seed}")
        sim = CyanoSim(tool, seed=args.seed, record_video=args.no_video == False)
        name = f"cyano_{tool}"
    report, detections, chosen, samples, image = sim.run()
    if image is None:
        print("\nReport")
        print(json.dumps(report, indent=2, default=str))
        print("\nOVERALL: FAILED (could not pick a single toothpick)")
        return

    print("Saved", sim.save_detection_image(image, detections, chosen, f"{name}_colonies.png"))
    print("Saved", sim.save_streak_image(samples, f"{name}_streak.png"))
    if args.no_video == False:
        print("Saved", sim.save_video(f"{name}.gif"))

    print("\nReport")
    print(json.dumps(report, indent=2, default=str))
    print("\nOVERALL:", "SUCCESS" if report["success"] == True else "FAILED")


if __name__ == "__main__":
    main()
