"""Segregation rounds: weekly restreak and colony PCR until each transformant lineage is fully segregated.

Run:
  ~/.venvs/mujoco-lab/bin/python run_segregation.py --robot so101 --seed 0
  ~/.venvs/mujoco-lab/bin/python run_segregation.py --robot panda --seed 0 --rounds 5
"""

import argparse
import json
import os

import pcr
import segregation as seg
from lab_sim import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "so101"], default="so101")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    def make_sim(scene_seed, source_colonies):
        if args.robot == "so101":
            import so101_world as sw
            from so101_sim import So101CyanoSim
            sim = So101CyanoSim("toothpick", seed=scene_seed, record_video=False, holder="rack", pcr=True, sectors=True,
                                source_colonies=source_colonies)
            return sim, pcr.tube_centers(sw.PCR_ORIGIN, "x")
        import plate_world as pw
        from cyano_sim import CyanoSim
        sim = CyanoSim("toothpick", seed=scene_seed, record_video=False, pcr=True, sectors=True, source_colonies=source_colonies)
        return sim, pcr.tube_centers(pw.PCR_ORIGIN, "y")

    summary, rounds = seg.run_rounds(make_sim, args.robot, args.seed, max_rounds=args.rounds,
                                     log=lambda t: print(t, flush=True))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seg.draw_gel(rounds, f"{OUTPUT_DIR}/segregation_{args.robot}_gel.png")
    with open(f"{OUTPUT_DIR}/segregation_{args.robot}_results.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"\nSegregation ({args.robot}): fully segregated by week {summary['segregated_week']} (None = not yet)")
    print("Saved", f"{OUTPUT_DIR}/segregation_{args.robot}_gel.png")


if __name__ == "__main__":
    main()
