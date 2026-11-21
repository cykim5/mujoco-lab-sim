"""Connected lab version: 8 colonies, each picked once, streaked in its sector, and resuspended in its PCR tube.

Run:
  ~/.venvs/mujoco-lab/bin/python run_connected_lab.py --robot so101 --seeds 0 1 2 3
  ~/.venvs/mujoco-lab/bin/python run_connected_lab.py --robot panda --seeds 0 1 2 3
  add --video to save a GIF and pictures of the first plate
"""

import argparse
import json
import os
from multiprocessing import Pool

import connected_lab as cl
import pcr
from lab_sim import OUTPUT_DIR


def one(job):
    robot, seed, video = job
    if robot == "so101":
        import so101_world as sw
        from so101_sim import So101CyanoSim
        sim = So101CyanoSim("toothpick", seed=seed, record_video=video, holder="rack", pcr=True, sectors=True)
        centers = pcr.tube_centers(sw.PCR_ORIGIN, "x")
    else:
        import plate_world as pw
        from cyano_sim import CyanoSim
        sim = CyanoSim("toothpick", seed=seed, record_video=video, pcr=True, sectors=True)
        centers = pcr.tube_centers(pw.PCR_ORIGIN, "y")
    sim.say = lambda text: print(f"[{robot} plate {seed}] {sim.data.time:6.0f} s  {text}", flush=True)
    report, detections, colonies, image, samples = cl.run_connected(sim, centers)
    report.update({"robot": robot, "seed": seed})
    name = f"connected_{robot}"
    if video == True:
        sim.save_video(f"{name}.gif")
    if seed == 0:
        cl.save_pictures(sim, samples, name)
    print(f"[{robot} plate {seed}] DONE: {report['colonies_ok']}/{report['colonies_tried']} colonies fully right, "
          f"lids back {report['lids_back']}, {report['time_s']:.0f} s", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "so101"], default="so101")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [(args.robot, seed, args.video and i == 0) for i, seed in enumerate(args.seeds)]
    if len(jobs) > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            reports = pool.map(one, jobs)
    else:
        reports = [one(jobs[0])]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"connected_{args.robot}_results.json")
    with open(path, "w") as f:
        json.dump(reports, f, indent=1, default=str)
    rows = [r for rep in reports for r in rep["rows"]]
    colors = {}
    for r in rows:
        colors[r.get("tube_color", "no toothpick")] = colors.get(r.get("tube_color", "no toothpick"), 0) + 1
    print(f"\nConnected lab ({args.robot}): {sum(1 for r in rows if r.get('ok') == True)}/{len(rows)} colonies fully right")
    print("  tube colors:", colors)
    print("Saved", path)


if __name__ == "__main__":
    main()
