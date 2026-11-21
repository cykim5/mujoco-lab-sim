"""Segregation rounds: restreak each transformant lineage weekly until colony PCR shows full segregation.

Workflow simulated (one round = one week of growth):
  Round 1  Transformation plate: pick 8 isolated transformant colonies. Streak each into sector k of a fresh
           selective plate and resuspend the remaining cells in PCR tube k.
  Round n  The sector plate from the previous round has grown. For each lineage not yet segregated, pick the
           isolated colony nearest the center of its sector (the sparse, inner end of the streak), streak it
           into sector k of a new plate, and run colony PCR again.
  Stop     A lineage is fully segregated when the PCR shows only the insert band.

Biology is a stated model, not simulated cells:
  - Cyanobacteria are polyploid. Each colony carries a fraction f of chromosome copies with the insert.
  - Transformant colonies start with f between 0.05 and 0.6.
  - Under selection, each colony grown from a restreak has f' = 1 - (1 - f) * r, with r drawn per colony
    between 0.1 and 0.7, so colonies on the same plate differ.
  - When fewer than 3% of copies are wild type, the colony is fully segregated (f = 1).
  - Colony PCR with primers flanking the insertion site gives an insert band (proportional to f) and a
    wild-type band (proportional to 1 - f). The wild-type band is undetectable below 5% of a normal band.
  - Each sector streak grows a dense region near the rim (merged colonies, not pickable) and isolated single
    colonies toward the center.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

import connected_lab as cl
import pcr
from lab_sim import OUTPUT_DIR

N_LINEAGES = 8
SEGREGATED_WT = 0.03  # wild-type copy fraction below which a colony counts as fully segregated
WT_DETECTION = 0.05  # wild-type band intensity (relative to a full band) below which the gel shows nothing
NORMAL_CELLS = 0.115  # cells released into the tube by the standard single-pick protocol (connected_lab model)


def grow_sector_colonies(dest_xy, k, parent_f, rng):
    """Colonies that grow from the streak in sector k after a week, positions relative to the plate center."""
    path = [np.array(p) - np.array(dest_xy) for p in cl.sector_path(dest_xy, k)]
    n = len(path)
    colonies = []
    # dense region: first 65% of the streak, colonies merge into a lawn
    for i in range(0, int(0.65 * n), 1):
        p = path[i] + rng.normal(0, 0.0003, 2)
        colonies.append({"x": float(p[0]), "y": float(p[1]), "r": float(rng.uniform(0.0005, 0.0008)), "f": None})
    # sparse region: isolated single colonies, 4 to 7 mm apart, toward the center
    i = int(0.65 * n)
    while i < n:
        p = path[i] + rng.normal(0, 0.0005, 2)
        f = 1 - (1 - parent_f) * rng.uniform(0.1, 0.7)
        if 1 - f < SEGREGATED_WT:
            f = 1.0
        colonies.append({"x": float(p[0]), "y": float(p[1]), "r": float(rng.uniform(0.00045, 0.0009)), "f": float(f)})
        i = i + int(rng.integers(4, 8))
    return colonies


def choose_inner_colonies(sim, detections, active):
    """For each active sector, the isolated single colony nearest the plate center (inner end of the streak)."""
    chosen = []
    for k in range(N_LINEAGES):
        if active[k] == False:
            chosen.append(None)
            continue
        best = None
        for d in detections:
            if d.get("single", True) == False:
                continue
            dx, dy = d["x"] - sim.source_xy[0], d["y"] - sim.source_xy[1]
            r = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx)) % 360
            if not (45 * k + 2 < angle < 45 * (k + 1) - 2) or r < 0.004 or r > 0.032:
                continue
            gap = min([math.hypot(d["x"] - o["x"], d["y"] - o["y"]) - d["r"] - o["r"] for o in detections if o is not d],
                      default=1.0)
            if gap < 0.0015:
                continue
            if best is None or r < best[0]:
                best = (r, d)
        chosen.append(None if best is None else best[1])
    return chosen


def pcr_bands(f, cells):
    """Relative band intensities (1.0 = a normal band). None means no cells, so an empty lane."""
    if f is None or cells <= 0:
        return {"insert": 0.0, "wt": 0.0}
    scale = cells / NORMAL_CELLS
    return {"insert": round(f * scale, 3), "wt": round((1 - f) * scale, 3)}


def draw_gel(rounds, path):
    """One gel row per week: a ladder lane, then lanes 1 to 8. Insert band (2.4 kb) above the wild-type band (1.1 kb)."""
    lane_w, row_h, margin = 46, 120, 70
    width = margin + lane_w * (N_LINEAGES + 1) + 20
    height = 40 + row_h * len(rounds)
    img = Image.new("RGB", (width, height), (18, 18, 24))
    draw = ImageDraw.Draw(img)
    draw.text((margin, 10), "L    1    2    3    4    5    6    7    8", fill=(200, 200, 200))
    for row, rnd in enumerate(rounds):
        top = 40 + row * row_h
        draw.text((8, top + 45), f"Week {rnd['week']}", fill=(200, 200, 200))
        draw.rectangle([margin - 4, top, width - 16, top + row_h - 12], outline=(60, 60, 70))
        # ladder
        for y in (20, 38, 55, 70, 84, 96):
            draw.rectangle([margin + 6, top + y, margin + lane_w - 10, top + y + 3], fill=(150, 150, 150))
        for k in range(N_LINEAGES):
            x0 = margin + lane_w * (k + 1) + 6
            x1 = x0 + lane_w - 16
            lane = rnd["lanes"][k]
            if lane is None:
                continue
            for name, y in (("insert", 30), ("wt", 70)):
                level = min(1.0, lane[name])
                if level < WT_DETECTION:
                    continue
                shade = int(40 + 215 * level)
                draw.rectangle([x0, top + y, x1, top + y + 5], fill=(shade, shade, shade))
    draw.text((width - 120, height - 14), "upper: insert   lower: WT", fill=(150, 150, 150))
    img.save(path)


def run_rounds(make_sim, robot, seed, max_rounds=4, centers=None, log=print):
    rng = np.random.default_rng(seed + 12345)
    lineage_f = [None] * N_LINEAGES
    segregated_week = [None] * N_LINEAGES
    active = [True] * N_LINEAGES
    source_colonies = None
    source_f = None
    rounds = []
    for week in range(1, max_rounds + 1):
        sim, centers = make_sim(seed * 10 + week, source_colonies)
        sim.say = lambda text, w=week: log(f"[{robot} week {w}] {sim.data.time:6.0f} s  {text}")
        if source_f is None:  # transformation plate: give each colony a starting insert fraction
            source_f = [float(rng.uniform(0.05, 0.6)) for _ in sim.truth["colonies"]]
        truth = sim.truth["colonies"]
        picked_f = [None] * N_LINEAGES

        def on_pick(k, colony, touch_xy):
            if touch_xy is None:
                return
            i = int(np.argmin([math.hypot(t["x"] - touch_xy[0], t["y"] - touch_xy[1]) for t in truth]))
            if math.hypot(truth[i]["x"] - touch_xy[0], truth[i]["y"] - touch_xy[1]) < truth[i]["r"] + 0.0005:
                picked_f[k] = source_f[i]

        if week == 1:
            choose = None
        else:
            choose = lambda s, dets: choose_inner_colonies(s, dets, active)
        report, detections, colonies, image, samples = cl.run_connected(sim, centers, choose=choose, on_pick=on_pick)

        lanes = []
        next_colonies = []
        next_f = []
        for k in range(N_LINEAGES):
            row = report["rows"][k] if k < len(report["rows"]) else {"skipped": True}
            if row.get("skipped") == True or "cells_in_tube" not in row:
                lanes.append(None)
                continue
            f = picked_f[k]
            bands = pcr_bands(f, row["cells_in_tube"])
            lanes.append(bands)
            lineage_f[k] = f
            if f is not None and bands["insert"] > WT_DETECTION and bands["wt"] < WT_DETECTION and segregated_week[k] is None:
                segregated_week[k] = week
                active[k] = False
            # the streak grows into next week's colonies
            if f is not None and row.get("streak_contact_pct", 0) > 50:
                grown = grow_sector_colonies(sim.dest_xy, k, f, rng)
                next_colonies.extend({"x": c["x"], "y": c["y"], "r": c["r"]} for c in grown)
                next_f.extend(c["f"] if c["f"] is not None else f for c in grown)
        rounds.append({"week": week, "lanes": lanes, "lineage_f": [None if f is None else round(f, 3) for f in lineage_f],
                       "segregated": [w is not None for w in segregated_week], "report": report})
        calls = []
        for k, lane in enumerate(lanes):
            if lane is None:
                calls.append(f"{k + 1}: -")
            elif lane["wt"] < WT_DETECTION:
                calls.append(f"{k + 1}: insert only")
            else:
                calls.append(f"{k + 1}: WT {lane['wt']:.2f}")
        log(f"[{robot} week {week}] colony PCR  " + "   ".join(calls))
        if week == 2:
            save_plate_with_choices(image, detections, colonies, f"{OUTPUT_DIR}/segregation_{robot}_week2_plate.png")
        if all(a == False for a in active) or len(next_colonies) == 0:
            break
        source_colonies = next_colonies
        source_f = next_f
    return {"robot": robot, "seed": seed, "segregated_week": segregated_week, "rounds": [
        {k: v for k, v in r.items() if k != "report"} | {"colonies_ok": r["report"]["colonies_ok"],
                                                         "colonies_tried": r["report"]["colonies_tried"],
                                                         "time_s": r["report"]["time_s"],
                                                         "lids_back": r["report"]["lids_back"]} for r in rounds]}, rounds


def save_plate_with_choices(image, detections, chosen, path):
    if image is None:
        return
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    size = picture.size[0]
    for d in detections:
        col, row = d["px"]
        rad = d["r_px"] + 3
        color = (230, 40, 40) if any(c is d for c in chosen if c is not None) else ((40, 90, 230) if d.get("single", True) else (150, 150, 150))
        draw.ellipse([col - rad, row - rad, col + rad, row + rad], outline=color, width=3 if color[0] == 230 else 1)
    picture.save(path)
