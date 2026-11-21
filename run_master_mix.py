"""Master-mix preparation: combine 2x premix, primers and water, mix, and aliquot 20 uL into the 8-tube PCR strip.

Run:
  ~/.venvs/mujoco-lab/bin/python run_master_mix.py --robot so101
  ~/.venvs/mujoco-lab/bin/python run_master_mix.py --robot panda
  add --video to save a GIF, --seed to change the scene
"""

import argparse
import json
import os

import liquid
import pcr
from lab_sim import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "so101"], default="so101")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    if args.robot == "so101":
        import so101_world as sw
        from so101_sim import So101CyanoSim
        sim = So101CyanoSim("toothpick", seed=args.seed, record_video=args.video, holder="none", pcr=True, liquid=True)
        centers = pcr.tube_centers(sw.PCR_ORIGIN, "x")
    else:
        import plate_world as pw
        from cyano_sim import CyanoSim
        sim = CyanoSim("toothpick", seed=args.seed, record_video=args.video, pcr=True, liquid=True)
        centers = pcr.tube_centers(pw.PCR_ORIGIN, "y")
    sim.say = lambda text: print(f"[{args.robot}] {sim.data.time:6.0f} s  {text}", flush=True)
    lq, summary = liquid.run_master_mix(sim, args.robot, centers)
    summary["error_budget"] = [liquid.monte_carlo(args.robot, tolerance=t) for t in (0.10, 0.05, 0.02)]
    summary["robot"] = args.robot
    summary["events"] = lq.events
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = f"master_mix_{args.robot}"
    with open(os.path.join(OUTPUT_DIR, f"{name}_results.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    if args.video == True:
        sim.save_video(f"{name}.gif")
    sim.save_image(f"{name}_strip.png", camera="pcr_cam", width=800, height=480)
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=1, default=str))


if __name__ == "__main__":
    main()
