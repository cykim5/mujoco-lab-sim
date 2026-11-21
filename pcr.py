"""Colony PCR: pick several colonies, patch each onto the fresh plate, and dip each into its own PCR tube.

For colony k the robot:
  1. takes a fresh toothpick,
  2. touches colony k (a different isolated colony each time),
  3. patches it onto spot k on the fresh plate (so a living copy of that clone is kept),
  4. dips it into PCR tube k of an 8-tube strip and swirls in the master mix,
  5. drops the toothpick in the waste bin.
Colony k -> patch k -> tube k, so every PCR result traces back to a clone you still have.

The master mix itself is not simulated (no liquids in MuJoCo). "Dipped" means the tip went at least
1 mm below the liquid level inside the right tube, and never touched any other tube.

Works with both robots: the Panda (CyanoSim) and the SO-101 with the printed rack (So101CyanoSim).
"""

import math

import numpy as np

import plate_world as pw

N_TUBES = 8
TUBE_PITCH = 0.009  # standard 9 mm strip spacing
TUBE_INNER_R = 0.0025  # 0.2 mL PCR tube, about 5 mm across inside
TUBE_DEPTH = 0.015
TUBE_FLOOR_Z = 0.004  # tube floor above the bench
LIQUID_HEIGHT = 0.004  # about 20 uL of master mix fills the bottom 4 mm
PATCH_RADIUS = 0.022  # patches sit on a circle 2.2 cm from the fresh plate's center


def tube_centers(origin_xy, along="x"):
    """Centers of the 8 tubes. origin_xy is the first tube; the strip runs along +x or +y."""
    ox, oy = origin_xy
    if along == "x":
        return [(ox + k * TUBE_PITCH, oy) for k in range(N_TUBES)]
    return [(ox, oy + k * TUBE_PITCH) for k in range(N_TUBES)]


def pcr_strip_xml(origin_xy, along="x"):
    """An 8-tube PCR strip standing in a printed holder. Tube walls are thin boxes, master mix is drawn only."""
    lines = []
    for k, (x, y) in enumerate(tube_centers(origin_xy, along)):
        walls = pw.ring_walls(TUBE_INNER_R + 0.0003, TUBE_FLOOR_Z + TUBE_DEPTH / 2, TUBE_DEPTH / 2,
                              "0.95 0.95 0.9 0.5", n_segments=16)
        walls = walls.replace("<geom ", f'<geom name="tube{k}_wall_SEG" ')
        # give each wall segment a unique name
        count = 0
        while "_SEG" in walls:
            walls = walls.replace("_SEG", f"{count}", 1)
            count = count + 1
        lines.append(f"""
    <body name="pcr_tube_{k}" pos="{x:.4f} {y:.4f} {pw.BENCH_Z}">
        <geom name="tube{k}_floor" type="cylinder" pos="0 0 {TUBE_FLOOR_Z - 0.0005}" size="{TUBE_INNER_R + 0.0006} 0.0005" rgba="0.95 0.95 0.9 0.6"/>
        {walls}
        <geom name="tube{k}_mix" type="cylinder" pos="0 0 {TUBE_FLOOR_Z + LIQUID_HEIGHT / 2}" size="{TUBE_INNER_R - 0.0002} {LIQUID_HEIGHT / 2}"
              rgba="0.3 0.55 0.95 0.6" contype="0" conaffinity="0"/>
    </body>""")
    # printed holder block around the tubes' lower part
    centers = tube_centers(origin_xy, along)
    cx = (centers[0][0] + centers[-1][0]) / 2
    cy = (centers[0][1] + centers[-1][1]) / 2
    length = (N_TUBES - 1) * TUBE_PITCH / 2 + 0.006
    size = f"{length:.4f} 0.006 0.0015" if along == "x" else f"0.006 {length:.4f} 0.0015"
    lines.append(f'<geom name="pcr_holder" type="box" pos="{cx:.4f} {cy:.4f} {pw.BENCH_Z + 0.0015}" size="{size}" rgba="0.2 0.55 0.9 1"/>')
    return "\n".join(lines)


def patch_spot(dest_xy, k):
    a = math.radians(22.5 + 45 * k)
    return (dest_xy[0] + PATCH_RADIUS * math.cos(a), dest_xy[1] + PATCH_RADIUS * math.sin(a))


# ---------------------------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------------------------

def go_tip(sim, xy, z):
    """Move the tool tip somewhere safely: the SO-101 travels high, the Panda rises before moving across."""
    if hasattr(sim, "travel_tip"):
        sim.travel_tip(xy, z)
    else:
        here = sim.tip_pos()
        sim.move_tip_to([here[0], here[1], max(here[2], pw.BENCH_Z + 0.06)], duration=0.6)
        sim.move_tip_to([xy[0], xy[1], max(here[2], pw.BENCH_Z + 0.06)], duration=1.2)
        sim.move_tip_to([xy[0], xy[1], z], duration=0.8)


def choose_colonies(sim, detections, n):
    """Pick up to n well-isolated colonies, each at least 4 mm (edge to edge) from every other colony."""
    chosen = []
    candidates = []
    for d in detections:
        from_center = math.hypot(d["x"] - sim.source_xy[0], d["y"] - sim.source_xy[1])
        if d["r"] < 0.0004 or from_center > 0.032 or d.get("single", True) == False:
            continue
        gap = min([math.hypot(d["x"] - o["x"], d["y"] - o["y"]) - d["r"] - o["r"] for o in detections if o is not d],
                  default=1.0)
        d["gap"] = gap
        if gap > 0.002:
            candidates.append(d)
    candidates.sort(key=lambda d: -d["gap"])
    for d in candidates[:n]:
        chosen.append(d)
    return chosen


def tube_contacts(sim, tube_geoms):
    """Which tubes the held toothpick is touching right now."""
    touched = set()
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if a in sim.tool_geoms:
                for k, geoms in tube_geoms.items():
                    if b in geoms:
                        touched.add(k)
    return touched


def patch(sim, k):
    """Touch down on patch spot k of the fresh plate and make three short strokes."""
    x, y = patch_spot(sim.dest_xy, k)
    start_z = pw.BENCH_Z + pw.PLATE_WALL_TOP + 0.002
    go_tip(sim, (x, y), start_z + 0.01)
    surface = sim.touch_down((x, y), start_z, lambda: sim.contact(sim.tool_geoms, sim.agar_geom["dest"]), use_tip=True)
    if surface is None:
        return 0.0
    in_contact = 0
    samples = 0
    arm = sim.data.ctrl[:sim.n_arm].copy()
    for stroke in range(3):
        offset = (stroke - 1) * 0.0012
        for t in np.linspace(-0.0025, 0.0025, 6) * (1 if stroke % 2 == 0 else -1):
            target = np.array([x + t, y + offset, surface - 0.0003]) - sim.tip_offset
            arm, _ = sim.solve_ik(target, start_arm=arm, max_iters=30)
            sim.data.ctrl[:sim.n_arm] = arm
            for _ in range(50):
                sim.step()
            force, _ = sim.contact(sim.tool_geoms, sim.agar_geom["dest"])
            samples = samples + 1
            if force > 0.0005:
                in_contact = in_contact + 1
    sim.move_tip_to([x, y, start_z + 0.01], duration=0.5)
    return in_contact / max(samples, 1)


def dip(sim, k, centers, tube_geoms):
    """Lower the tip into tube k, below the master-mix level, swirl three small circles, and lift out."""
    x, y = centers[k]
    top = pw.BENCH_Z + TUBE_FLOOR_Z + TUBE_DEPTH
    liquid_top = pw.BENCH_Z + TUBE_FLOOR_Z + LIQUID_HEIGHT
    go_tip(sim, (x, y), top + 0.012)
    sim.move_tip_to([x, y, top + 0.003], duration=0.5)
    touched_other = set()
    lowest = 1.0
    inside = True
    arm = sim.data.ctrl[:sim.n_arm].copy()
    # straight down to 1.5 mm above the tube floor, in small steps slow enough for the motors to follow
    for z in np.linspace(top + 0.003, pw.BENCH_Z + TUBE_FLOOR_Z + 0.0015, 30):
        arm, _ = sim.solve_ik(np.array([x, y, z]) - sim.tip_offset, start_arm=arm, max_iters=30)
        sim.data.ctrl[:sim.n_arm] = arm
        for _ in range(60):
            sim.step()
        tip = sim.tip_pos()
        lowest = min(lowest, tip[2])
        if tip[2] < top and math.hypot(tip[0] - x, tip[1] - y) > TUBE_INNER_R:
            inside = False
        touched_other |= tube_contacts(sim, tube_geoms) - {k}
    # let the arm settle at the bottom, then swirl three circles of 0.8 mm radius in the master mix
    for _ in range(300):
        sim.step()
    lowest = min(lowest, sim.tip_pos()[2])
    z = pw.BENCH_Z + TUBE_FLOOR_Z + 0.0015
    for t in np.linspace(0, 3 * 2 * math.pi, 36):
        p = np.array([x + 0.0008 * math.cos(t), y + 0.0008 * math.sin(t), z])
        arm, _ = sim.solve_ik(p - sim.tip_offset, start_arm=arm, max_iters=30)
        sim.data.ctrl[:sim.n_arm] = arm
        for _ in range(50):
            sim.step()
        lowest = min(lowest, sim.tip_pos()[2])
        touched_other |= tube_contacts(sim, tube_geoms) - {k}
    sim.move_tip_to([x, y, top + 0.012], duration=0.8)
    dipped = inside == True and lowest < liquid_top - 0.001
    return {"dipped": bool(dipped), "depth_below_mix_mm": round((liquid_top - lowest) * 1000, 2),
            "touched_other_tubes": sorted(touched_other)}


def run_pcr(sim, centers, n=N_TUBES):
    """Whole colony-PCR run. Returns a report with one row per colony."""
    m = sim.model
    tube_geoms = {}
    for k in range(N_TUBES):
        body = m.body(f"pcr_tube_{k}").id
        tube_geoms[k] = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == body)

    if hasattr(sim, "calibrate"):
        sim.calibrate()
    sim.say("taking lids off")
    lids_off = sim.lift_lid("source", sim.source_xy, sim.lid_park_xy["source"], put_back=False) and \
        sim.lift_lid("dest", sim.dest_xy, sim.lid_park_xy["dest"], put_back=False)
    sim.move_arm(sim.park_arm, sim.gripper_closed, duration=1.5)
    detections, image, _ = sim.find_colonies()
    colonies = choose_colonies(sim, detections, n)
    sim.say(f"found {len(detections)} colonies, {len(colonies)} isolated enough for PCR")

    rows = []
    for k, colony in enumerate(colonies):
        row = {"colony": k + 1}
        try:
            if hasattr(sim, "pick_from_rack"):
                sim.slot = (sim.slot + (1 if k > 0 else 0)) % 24
                sim.pick_up_tool()
                sim.slot = sim.tool_body_ids.index(sim.locked_body)
            else:
                sim.select_slot(k)
                sim.pick_up_tool()
        except Exception as e:  # could not get a toothpick
            row.update({"picked": False, "error": str(e)})
            rows.append(row)
            continue
        touch_xy, surface = sim.pick_colony(colony)
        grade = sim.grade_pick(touch_xy, colony)
        row.update({"picked": grade["picked"], "clean_pick": grade["clean_pick"], "miss_mm": grade["miss_mm"]})
        row["patch_contact"] = round(patch(sim, k), 2)
        row.update(dip(sim, k, centers, tube_geoms))
        sim.drop_tool_in_bin()
        row["ok"] = (row["clean_pick"] == True and row["patch_contact"] >= 0.8 and row["dipped"] == True
                     and len(row["touched_other_tubes"]) == 0)
        sim.say(f"colony {k + 1}: clean pick {row['clean_pick']} ({row['miss_mm']} mm), patch {row['patch_contact']:.0%}, "
                f"dipped {row['dipped']} ({row['depth_below_mix_mm']} mm into the mix), other tubes touched {row['touched_other_tubes']}")
        rows.append(row)

    sim.say("putting lids back")
    back = sim.lift_lid("dest", sim.lid_park_xy["dest"], sim.dest_xy, put_back=True) and \
        sim.lift_lid("source", sim.lid_park_xy["source"], sim.source_xy, put_back=True)
    sim.move_arm(sim.park_arm, sim.gripper_open, duration=1.5)
    report = {"colonies_done": sum(1 for r in rows if r.get("ok") == True), "colonies_tried": len(rows),
              "lids_off": bool(lids_off), "lids_back": bool(back and sim.lids_seated()),
              "time_s": round(sim.data.time, 1), "rows": rows}
    return report, detections, colonies, image
