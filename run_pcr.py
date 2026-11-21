"""Colony PCR run: 8 colonies -> 8 patches on the fresh plate -> 8 PCR tubes.

Run:
  ~/.venvs/mujoco-lab/bin/python run_pcr.py --robot panda --tool toothpick
  ~/.venvs/mujoco-lab/bin/python run_pcr.py --robot so101             (SO-101 with the printed toothpick rack)
  add --seeds 0 1 2 3 to test several plates, --video to save a GIF of the first one
"""

import argparse
import json
import os
from multiprocessing import Pool

import pcr
from lab_sim import OUTPUT_DIR


def one(robot, tool, seed, video):
    if robot == "so101":
        import so101_world as sw
        from so101_sim import So101CyanoSim
        sim = So101CyanoSim("toothpick", seed=seed, record_video=video, holder="rack", pcr=True)
        centers = pcr.tube_centers(sw.PCR_ORIGIN, "x")
    else:
        import plate_world as pw
        from cyano_sim import CyanoSim
        sim = CyanoSim(tool, seed=seed, record_video=video, pcr=True)
        centers = pcr.tube_centers(pw.PCR_ORIGIN, "y")
    sim.say = lambda text: print(f"[{robot} plate {seed}] {sim.data.time:6.0f} s  {text}", flush=True)
    report, detections, colonies, image = pcr.run_pcr(sim, centers)
    report.update({"robot": robot, "tool": tool if robot == "panda" else "toothpick", "seed": seed})
    if video == True:
        name = f"pcr_{robot}"
        sim.save_video(f"{name}.gif")
        sim.save_image(f"{name}_plate.png", camera="dest_cam", width=800, height=800)
        if image is not None:
            sim.save_detection_image(image, detections, colonies[0] if colonies else None, f"{name}_colonies.png")
    return report


def one_job(job):
    robot, tool, seed, video = job
    r = one(robot, tool, seed, video)
    print(f"[{robot} plate {seed}] DONE: {r['colonies_done']}/{r['colonies_tried']} colonies fully done, "
          f"lids back {r['lids_back']}, {r['time_s']:.0f} s", flush=True)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "so101"], default="panda")
    parser.add_argument("--tool", choices=["loop", "toothpick"], default="toothpick", help="Panda only")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = [(args.robot, args.tool, seed, args.video and i == 0) for i, seed in enumerate(args.seeds)]
    if len(jobs) > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            reports = pool.map(one_job, jobs)
    else:
        reports = [one_job(jobs[0])]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"pcr_{args.robot}_results.json")
    with open(path, "w") as f:
        json.dump(reports, f, indent=1, default=str)
    done = sum(r["colonies_done"] for r in reports)
    tried = sum(r["colonies_tried"] for r in reports)
    print(f"\nColony PCR ({args.robot}): {done}/{tried} colonies picked, patched and dipped correctly")
    print("Saved", path)


if __name__ == "__main__":
    main()
