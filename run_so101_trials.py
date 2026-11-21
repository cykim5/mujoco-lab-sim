"""Run many random plates with the SO-101 build version and summarize each step's success rate.

Every seed changes the colony layout, agar thickness, how the toothpicks fall in the cup,
and the arm's hidden calibration error (up to 0.5 degrees per joint).

Run:  ~/.venvs/mujoco-lab/bin/python run_so101_trials.py --trials 8
"""

import argparse
import csv
import os
from multiprocessing import Pool

from lab_sim import OUTPUT_DIR

COLUMNS = ["seed", "success", "lids_off", "pick_attempts", "toothpick_tilt_deg", "picked", "clean_pick", "miss_mm",
           "camera_corrections_mm", "streak_ok", "contact_pct", "max_depth_mm", "lids_back", "hit_plate_wall",
           "cycle_time_s", "calibration_error_deg"]


def one_run(seed):
    from so101_sim import So101CyanoSim
    sim = So101CyanoSim("toothpick", seed=seed, record_video=False)
    report = sim.run()[0]
    return {key: report.get(key) for key in COLUMNS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with Pool(args.workers) as pool:
        rows = pool.map(one_run, range(args.trials))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "so101_trials.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)

    def count(key):
        return sum(1 for r in rows if r.get(key) == True)

    got_toothpick = sum(1 for r in rows if r.get("toothpick_tilt_deg") is not None)
    print("\nSO-101 summary")
    print(f"  lids off              {count('lids_off')}/{n}")
    print(f"  single toothpick      {got_toothpick}/{n}  (shakes needed: {[r.get('pick_attempts') for r in rows]})")
    print(f"  clean colony touch    {count('clean_pick')}/{n}  (misses mm: {[r.get('miss_mm') for r in rows]})")
    print(f"  streak ok             {count('streak_ok')}/{n}")
    print(f"  lids back on          {count('lids_back')}/{n}")
    print(f"  full success          {count('success')}/{n}")
    print("Saved", path)


if __name__ == "__main__":
    main()
