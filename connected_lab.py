"""Connected lab version: the lab's real colony-PCR protocol, adapted so no toothpick flip is needed.

The lab protocol pokes a colony with both ends of one toothpick (heavy, flip, light). Neither robot can
flip a toothpick in its hand at bench height (see README), so this version uses ONE end for everything:

For colony k (k = 1..8), with one fresh toothpick:
  1. poke the colony once,
  2. zigzag inside pie slice k of the fresh plate, from the rim toward the center,
  3. swirl the same end in PCR tube k, releasing whatever cells are left over,
  4. discard.

Cell amounts are a simple model, not simulated biology:
  - a clean poke loads 1.0 (relative units) onto the tip,
  - streaking deposits cells as it goes: the tip keeps exp(-contact length / STREAK_DECAY) of its load,
  - swirling 3 circles in the master mix releases 1 - 0.5**3 = 87.5% of what is left,
  - the master mix turns green in proportion to the cells released. The lab says it should be VERY slightly
    green, so the target band is GREEN_OK_LOW to GREEN_OK_HIGH. Below means maybe too few cells for the PCR,
    above means too many (cells inhibit PCR).
STREAK_DECAY is an assumption. A photo of real tubes after this protocol would let you measure it.
"""

import math

import mujoco
import numpy as np
from PIL import Image

import pcr
import plate_world as pw
from lab_sim import OUTPUT_DIR

N_COLONIES = 8
STREAK_DECAY = 0.06  # meters of streak contact over which the tip's load drops to 37% (assumed)
SWIRL_RELEASE = 1 - 0.5 ** 3
GREEN_OK_LOW = 0.05
GREEN_OK_HIGH = 0.30
SECTOR_R_OUT = 0.035  # start strokes 3.5 mm inside the agar edge
SECTOR_R_IN = 0.006


def sector_path(dest_xy, k):
    """Zigzag inside pie slice k (between the sticker lines at 45k and 45(k+1) degrees), rim toward center."""
    bisector = math.radians(45 * k + 22.5)
    u = np.array([math.cos(bisector), math.sin(bisector)])  # outward along the middle of the slice
    v = np.array([-u[1], u[0]])  # across the slice
    center = np.array(dest_xy)
    points = []
    radii = np.linspace(SECTOR_R_OUT, SECTOR_R_IN, 7)
    for i, r in enumerate(radii):
        half = max(r * math.tan(math.radians(22.5)) - 0.0015, 0.0006)  # stay 1.5 mm inside the sticker lines
        side = 1 if i % 2 == 0 else -1
        points.append(center + r * u - side * half * v)
        points.append(center + r * u + side * half * v)
    dense = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.linalg.norm(b - a) / 0.001))
        for j in range(1, n + 1):
            dense.append(a + (b - a) * j / n)
    return dense


def sector_streak(sim, k):
    """Touch down at the rim end of slice k and zigzag to the center at about 2 cm/s, pressing lightly."""
    path = sector_path(sim.dest_xy, k)
    start_z = pw.BENCH_Z + pw.PLATE_WALL_TOP + 0.002
    pcr.go_tip(sim, path[0], start_z + 0.01)
    surface = sim.touch_down(path[0], start_z, lambda: sim.contact(sim.tool_geoms, sim.agar_geom["dest"]), use_tip=True)
    if surface is None:
        return [], 0.0
    samples = []
    arm = sim.data.ctrl[:sim.n_arm].copy()
    for p in path:
        target = np.array([p[0], p[1], surface - 0.0003]) - sim.tip_offset
        arm, _ = sim.solve_ik(target, start_arm=arm, max_iters=30)
        sim.data.ctrl[:sim.n_arm] = arm
        for _ in range(50):
            sim.step()
        force, _ = sim.contact(sim.tool_geoms, sim.agar_geom["dest"])
        tip = sim.tip_pos()
        samples.append({"sector": k, "pos": tip, "force": force,
                        "depth": sim.truth["agar_top"]["dest"] - tip[2]})
    sim.move_tip_to([path[-1][0], path[-1][1], start_z + 0.01], duration=0.5)
    contact_length = sum(1 for s in samples if s["force"] > 0.0005) * 0.001
    return samples, contact_length


def tint_tube(sim, k, amount):
    """Master mix color: clear blue, turning green with more cells."""
    g = sim.model.geom(f"tube{k}_mix").id
    t = min(1.0, amount / 0.6)
    base = np.array([0.35, 0.55, 0.95, 0.6])
    green = np.array([0.25, 0.85, 0.20, 0.95])
    sim.model.geom_rgba[g] = base + (green - base) * t


def green_label(amount):
    if amount < GREEN_OK_LOW:
        return "too little"
    if amount > GREEN_OK_HIGH:
        return "too green"
    return "slight green"


def run_connected(sim, centers):
    m = sim.model
    tube_geoms = {}
    for k in range(pcr.N_TUBES):
        body = m.body(f"pcr_tube_{k}").id
        tube_geoms[k] = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == body)

    if hasattr(sim, "calibrate"):
        sim.calibrate()
    sim.say("taking lids off")
    lids_off = sim.lift_lid("source", sim.source_xy, sim.lid_park_xy["source"], put_back=False) and \
        sim.lift_lid("dest", sim.dest_xy, sim.lid_park_xy["dest"], put_back=False)
    sim.move_arm(sim.park_arm, sim.gripper_closed, duration=1.5)
    detections, image, _ = sim.find_colonies()
    colonies = pcr.choose_colonies(sim, detections, N_COLONIES)
    sim.say(f"found {len(detections)} colonies, {len(colonies)} isolated single colonies to test")

    rows = []
    all_samples = []
    for k, colony in enumerate(colonies):
        row = {"colony": k + 1, "sector": k + 1, "tube": k + 1}
        try:
            if hasattr(sim, "pick_from_rack"):
                sim.slot = (sim.slot + (1 if k > 0 else 0)) % 24
                sim.pick_up_tool()
                sim.slot = sim.tool_body_ids.index(sim.locked_body)
            else:
                sim.select_slot(k)
                sim.pick_up_tool()
        except Exception as e:
            row.update({"ok": False, "error": str(e)})
            rows.append(row)
            continue
        # 1. poke once
        touch_xy, _ = sim.pick_colony(colony)
        grade = sim.grade_pick(touch_xy, colony)
        load = 1.0 if grade["picked"] == True else 0.0
        row.update({"picked": grade["picked"], "clean_pick": grade["clean_pick"], "miss_mm": grade["miss_mm"]})
        # 2. zigzag in slice k, rim to center
        samples, contact_length = sector_streak(sim, k)
        all_samples.extend(samples)
        contact_pct = 100.0 * sum(1 for s in samples if s["force"] > 0.0005) / max(len(samples), 1)
        max_depth = max([s["depth"] for s in samples], default=0.0)
        inside = all(abs(math.degrees(math.atan2(s["pos"][1] - sim.dest_xy[1], s["pos"][0] - sim.dest_xy[0])) % 360
                         - (45 * k + 22.5)) < 22.5 for s in samples)
        leftover = load * math.exp(-contact_length / STREAK_DECAY)
        row.update({"streak_contact_pct": round(contact_pct, 1), "streak_length_mm": round(contact_length * 1000),
                    "streak_max_depth_mm": round(max_depth * 1000, 2), "stayed_in_slice": bool(inside),
                    "leftover_on_tip": round(leftover, 3)})
        # 3. swirl the leftover in tube k
        d = pcr.dip(sim, k, centers, tube_geoms)
        released = leftover * SWIRL_RELEASE if d["dipped"] == True else 0.0
        tint_tube(sim, k, released)
        row.update({"dipped": d["dipped"], "touched_other_tubes": d["touched_other_tubes"],
                    "cells_in_tube": round(released, 3), "tube_color": green_label(released)})
        # 4. discard
        sim.drop_tool_in_bin()
        row["ok"] = (row["clean_pick"] == True and contact_pct >= 90 and max_depth <= 0.001 and inside == True
                     and d["dipped"] == True and len(d["touched_other_tubes"]) == 0
                     and row["tube_color"] == "slight green")
        sim.say(f"colony {k + 1}: clean poke {row['clean_pick']} ({row['miss_mm']} mm), slice {k + 1} zigzag "
                f"{contact_pct:.0f}% contact, {row['streak_length_mm']} mm long, leftover {leftover:.2f}, "
                f"tube {k + 1}: {row['tube_color']} ({released:.2f})")
        rows.append(row)

    sim.say("putting lids back")
    back = sim.lift_lid("dest", sim.lid_park_xy["dest"], sim.dest_xy, put_back=True) and \
        sim.lift_lid("source", sim.lid_park_xy["source"], sim.source_xy, put_back=True)
    sim.move_arm(sim.park_arm, sim.gripper_open, duration=1.5)
    report = {"colonies_ok": sum(1 for r in rows if r.get("ok") == True), "colonies_tried": len(rows),
              "lids_off": bool(lids_off), "lids_back": bool(back and sim.lids_seated()),
              "time_s": round(sim.data.time, 1), "streak_decay_assumed_m": STREAK_DECAY, "rows": rows}
    return report, detections, colonies, image, all_samples


def save_pictures(sim, samples, name):
    """Fresh plate with the 8 slice zigzags drawn on it, and the PCR strip with each tube's color."""
    renderer = mujoco.Renderer(sim.model, height=800, width=800, max_geom=20000)
    renderer.update_scene(sim.data, camera="dest_cam")
    scene = renderer.scene
    for s in samples:
        if scene.ngeom >= scene.maxgeom:
            break
        touching = s["force"] > 0.0005
        rgba = np.array([0.1, 0.5, 0.15, 1.0]) if touching == True else np.array([0.9, 0.1, 0.1, 1.0])
        pos = s["pos"].copy()
        pos[2] = pos[2] + 0.0004
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0006, 0, 0]), pos,
                            np.eye(3).flatten(), rgba.astype(np.float32))
        scene.ngeom = scene.ngeom + 1
    Image.fromarray(renderer.render()).save(f"{OUTPUT_DIR}/{name}_plate.png")
    renderer.close()
    renderer = mujoco.Renderer(sim.model, height=480, width=800)
    renderer.update_scene(sim.data, camera="pcr_cam")
    Image.fromarray(renderer.render()).save(f"{OUTPUT_DIR}/{name}_tubes.png")
    renderer.close()
