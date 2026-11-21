"""Run many random plates with each tool and summarize the success rate.

Every seed gives a different colony layout and different agar thickness on both plates.

Run:  ~/.venvs/mujoco-lab/bin/python run_trials.py --trials 10
"""

import argparse
import csv
import os

from cyano_sim import CyanoSim
from lab_sim import OUTPUT_DIR

COLUMNS = ["tool", "seed", "success", "lids_off", "picked", "clean_pick", "miss_mm", "streak_ok",
           "contact_pct", "max_depth_mm", "max_force_N", "lids_back", "hit_plate_wall",
           "colonies_found", "colonies_total", "source_agar_error_mm", "dest_agar_error_mm", "cycle_time_s"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--tools", nargs="+", default=["loop", "toothpick"], choices=["loop", "toothpick"])
    args = parser.parse_args()

    rows = []
    for tool in args.tools:
        for seed in range(args.trials):
            print(f"\n=== {tool}, seed {seed} ===")
            sim = CyanoSim(tool, seed=seed, record_video=False)
            report = sim.run()[0]
            detection = report.get("detection", {})
            row = {key: report.get(key) for key in COLUMNS}
            row["colonies_found"] = detection.get("found")
            row["colonies_total"] = detection.get("total")
            rows.append(row)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "trials.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n\nSummary")
    for tool in args.tools:
        mine = [r for r in rows if r["tool"] == tool]
        n = len(mine)

        def count(key):
            return sum(1 for r in mine if r[key] == True)

        times = [r["cycle_time_s"] for r in mine if r["cycle_time_s"] is not None]
        misses = [r["miss_mm"] for r in mine if r["miss_mm"] is not None]
        print(f"  {tool:9s}  full success {count('success')}/{n}   clean pick {count('clean_pick')}/{n}   "
              f"streak ok {count('streak_ok')}/{n}   lids ok {count('lids_back')}/{n}   "
              f"worst miss {max(misses) if misses else '-'} mm   avg cycle {sum(times) / max(len(times), 1):.0f} s")
    print("Saved", path)


if __name__ == "__main__":
    main()
