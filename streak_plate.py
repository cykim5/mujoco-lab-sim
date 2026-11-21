"""Run one cyano colony pick + quadrant restreak, plate to plate.

Run:
  ~/.venvs/mujoco-lab/bin/python streak_plate.py                 (asks which tool)
  ~/.venvs/mujoco-lab/bin/python streak_plate.py --tool loop
  ~/.venvs/mujoco-lab/bin/python streak_plate.py --tool toothpick --seed 3
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
    args = parser.parse_args()

    tool = args.tool
    if tool is None:
        tool = ask_for_tool()

    print(f"\nRunning with the {tool}, seed {args.seed}")
    sim = CyanoSim(tool, seed=args.seed, record_video=args.no_video == False)
    report, detections, chosen, samples, image = sim.run()

    name = f"cyano_{tool}"
    print("Saved", sim.save_detection_image(image, detections, chosen, f"{name}_colonies.png"))
    print("Saved", sim.save_streak_image(samples, f"{name}_streak.png"))
    if args.no_video == False:
        print("Saved", sim.save_video(f"{name}.gif"))

    print("\nReport")
    print(json.dumps(report, indent=2))
    print("\nOVERALL:", "SUCCESS" if report["success"] == True else "FAILED")


if __name__ == "__main__":
    main()
