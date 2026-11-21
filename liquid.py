"""Liquid handling and master-mix preparation.

Hardware (printable build):
  - A syringe pump stays on the bench: stepper, lead screw, 250 uL glass syringe, water-filled line. Only the
    small printed nozzle is on the arm (held by the fingertips), so the arm carries almost no extra weight.
    The tubing is not modeled physically.
  - Pipette tips (200 uL) hang by their collars in a tip box. The nozzle is pressed into a tip to seal it.
  - A printed stripper plate with a slot sits on the waste cup. The arm slides the nozzle into the slot with the
    tip collar under the plate and lifts; the plate pulls the tip off and it drops into the cup.

Liquids are tracked as volumes (MuJoCo has no fluids):
  - Each tube's liquid level follows from its volume; the tip must stay below the surface while aspirating.
  - Each aspirate/dispense has an error: dispensed = target * (1 + N(0, cv)) + N(0, abs_sd).
  - A tip seals only if the nozzle is in the collar and the arm can press down with at least SEAT_FORCE at that
    pose; it strips only if the flange is caught under the plate and the arm can lift with at least STRIP_FORCE.
    The arm's force capacity comes from its motor torque limits minus what it needs to hold itself up.
    SEAT_FORCE and STRIP_FORCE are assumptions to test against real tips.

Master mix for 8 colony PCRs (+10% extra), 20 uL reactions: 2x premix 10 uL, primer mix 1 uL, water 9 uL each.
"""

import math

import mujoco
import numpy as np

import pcr
import plate_world as pw

TIP_LENGTH = 0.045
# Note: tips are modeled at about 2 g (real: ~0.3 g) with stiff collar contacts, so press forces are not lost to
# simulation softness on a very light body.
TIP_COLLAR_R = 0.0035
TIP_SEGMENTS = [(0.0005, 0.0, 0.010), (0.0010, 0.010, 0.025), (0.0018, 0.025, 0.040)]  # (radius, from, to) above the end
NOZZLE_LENGTH = 0.045  # printed nozzle body: gripped near its top, cone at the bottom
NOZZLE_R = 0.0020
TUBE_R = 0.003  # 0.5 mL tubes, inner radius
TUBE_DEPTH = 0.030
UL_HEIGHT = 1e-9 / (math.pi * TUBE_R ** 2)  # meters of liquid per uL in a reagent tube
PCR_UL_HEIGHT = 1e-9 / (math.pi * pcr.TUBE_INNER_R ** 2)
SEAT_FORCE = 4.0  # N needed to seal a tip (assumed)
STRIP_FORCE = 3.0  # N needed to pull a tip off (assumed)
N_TIPS = 8

N_RXN = 8
EXTRA = 1.1
RECIPE = [("water", 9.0), ("premix", 10.0), ("primers", 1.0)]  # uL per reaction, in pipetting order
REAGENTS = {"premix": 200.0, "primers": 40.0, "water": 300.0, "mix": 0.0}  # starting volumes, uL
REACTION_VOLUME = 20.0
TOLERANCE = 0.10

PUMP_ERROR = {"so101": {"cv": 0.015, "abs_sd": 0.15}, "panda": {"cv": 0.006, "abs_sd": 0.05}}


def layout(robot):
    if robot == "so101":
        return {"nozzle": (0.20, 0.07), "tipbox": (0.205, 0.0), "reagents_x": 0.245, "reagents_y0": -0.03,
                "stripper": (0.10, -0.225), "stripper_z": pw.BENCH_Z + 0.083, "pump": (-0.03, 0.12)}
    return {"nozzle": (0.33, 0.20), "tipbox": (0.40, 0.22), "reagents_x": 0.61, "reagents_y0": -0.03,
            "stripper": (0.30, -0.25), "stripper_z": pw.BENCH_Z + 0.053, "pump": (0.25, 0.30)}


def tip_slots(lay):
    cx, cy = lay["tipbox"]
    return [(cx + (c - 0.5) * 0.009, cy + (r - 1.5) * 0.009) for c in range(2) for r in range(4)]


def reagent_xy(lay, name):
    order = ["premix", "primers", "water", "mix"]
    return (lay["reagents_x"], lay["reagents_y0"] + 0.02 * order.index(name))


def liquid_xml(robot, gripper_body):
    lay = layout(robot)
    parts = []
    # nozzle in its stand
    nx, ny = lay["nozzle"]
    parts.append(f"""
    <body name="nozzle_stand" pos="{nx} {ny} {pw.BENCH_Z}">
        <geom type="box" pos="0.006 0 0.012" size="0.002 0.008 0.012" rgba="0.2 0.55 0.9 1"/>
        <geom type="box" pos="-0.006 0 0.012" size="0.002 0.008 0.012" rgba="0.2 0.55 0.9 1"/>
        <geom type="box" pos="0 0.006 0.012" size="0.004 0.002 0.012" rgba="0.2 0.55 0.9 1"/>
        <geom type="box" pos="0 -0.006 0.012" size="0.004 0.002 0.012" rgba="0.2 0.55 0.9 1"/>
    </body>
    <body name="nozzle" pos="{nx} {ny} {pw.BENCH_Z + 0.0002}">
        <freejoint/>
        <site name="nozzle_end" pos="0 0 0" size="0.0005"/>
        <geom name="nozzle_body" type="cylinder" pos="0 0 {0.009 + (NOZZLE_LENGTH - 0.009) / 2:.4f}" size="{NOZZLE_R} {(NOZZLE_LENGTH - 0.009) / 2:.4f}"
              rgba="0.2 0.55 0.9 1" mass="0.004" friction="1.2 0.01 0.001"/>
        <geom name="nozzle_cone1" type="cylinder" pos="0 0 0.0015" size="0.0014 0.0015" rgba="0.2 0.55 0.9 1" mass="0.0002" solref="0.002 1" solimp="0.95 0.99 0.0005"/>
        <geom name="nozzle_cone2" type="cylinder" pos="0 0 0.0045" size="0.0017 0.0015" rgba="0.2 0.55 0.9 1" mass="0.0002" solref="0.002 1" solimp="0.95 0.99 0.0005"/>
        <geom name="nozzle_cone3" type="cylinder" pos="0 0 0.0075" size="0.0019 0.0015" rgba="0.2 0.55 0.9 1" mass="0.0002" solref="0.002 1" solimp="0.95 0.99 0.0005"/>
    </body>""")
    # tip box: tips hang by their collars on a perforated top plate
    cx, cy = lay["tipbox"]
    box_h = TIP_LENGTH - 0.004
    walls = []
    hole = 0.0022
    for x in (cx - 0.009, cx, cx + 0.009):
        walls.append(f'<geom type="box" pos="{x:.4f} {cy} {box_h:.4f}" size="{(0.009 - 2 * hole) / 2:.4f} 0.022 0.001" rgba="0.85 0.6 0.2 1" solref="0.002 1" solimp="0.95 0.99 0.0005"/>')
    for k in range(5):
        y = cy + (k - 2) * 0.009
        walls.append(f'<geom type="box" pos="{cx} {y:.4f} {box_h:.4f}" size="0.0135 {(0.009 - 2 * hole) / 2:.4f} 0.001" rgba="0.85 0.6 0.2 1" solref="0.002 1" solimp="0.95 0.99 0.0005"/>')
    walls.append(f'<geom type="box" pos="{cx} {cy} 0.001" size="0.0135 0.022 0.001" rgba="0.85 0.6 0.2 0.4"/>')
    parts.append(f'<body name="tipbox" pos="0 0 {pw.BENCH_Z}">{"".join(walls)}</body>')
    for i, (x, y) in enumerate(tip_slots(lay)):
        segs = "".join(f'<geom name="tip{i}_s{j}" type="capsule" fromto="0 0 {a + r:.4f} 0 0 {b - r:.4f}" size="{r}" '
                       f'rgba="0.95 0.95 0.95 0.7" mass="0.0002"/>' for j, (r, a, b) in enumerate(TIP_SEGMENTS))
        # collar: an inner ring the nozzle cone wedges into (1.6 mm inner radius) and an outer flange
        # (3.5 mm radius) that hangs on the tip box and catches on the stripper plate
        collar = pw.ring_walls(0.0021, TIP_LENGTH - 0.0025, 0.0025, "0.95 0.95 0.95 0.8", n_segments=12,
                               extra='mass="0.001" solref="0.002 1" solimp="0.95 0.99 0.0005"')
        collar += pw.ring_walls(TIP_COLLAR_R - 0.0005, TIP_LENGTH - 0.00075, 0.00075, "0.95 0.95 0.95 0.8", n_segments=16,
                                extra='mass="0.0005" solref="0.002 1" solimp="0.95 0.99 0.0005"')
        collar = collar.replace("<geom ", f'<geom name="tip{i}_collar_SEG" ')
        count = 0
        while "_SEG" in collar:
            collar = collar.replace("_SEG", f"{count}", 1)
            count = count + 1
        parts.append(f"""
    <body name="tip_{i}" pos="{x:.4f} {y:.4f} {pw.BENCH_Z + box_h + 0.001 - (TIP_LENGTH - 0.005):.4f}">
        <freejoint/>
        <site name="tip_end_{i}" pos="0 0 0" size="0.0004"/>
        {segs}
        {collar}
    </body>""")
    # reagent rack: 0.5 mL tubes with drawn liquid
    for name in ["premix", "primers", "water", "mix"]:
        x, y = reagent_xy(lay, name)
        walls_r = pw.ring_walls(TUBE_R + 0.0004, TUBE_DEPTH / 2, TUBE_DEPTH / 2, "0.95 0.95 0.9 0.3", n_segments=16)
        color = {"premix": "0.95 0.75 0.3 0.8", "primers": "0.8 0.4 0.8 0.8", "water": "0.5 0.75 1.0 0.6", "mix": "0.9 0.6 0.5 0.8"}[name]
        parts.append(f"""
    <body name="reagent_{name}" pos="{x:.4f} {y:.4f} {pw.BENCH_Z}">
        <geom type="cylinder" pos="0 0 0.0005" size="{TUBE_R + 0.0008} 0.0005" rgba="0.95 0.95 0.9 0.6"/>
        {walls_r}
        <geom name="liquid_{name}" type="cylinder" pos="0 0 0.0011" size="{TUBE_R - 0.0002} 0.0001" rgba="{color}" contype="0" conaffinity="0"/>
    </body>""")
    # stripper plate covering half of the waste cup opening (the -y half). The 5 mm slot opens toward the open
    # half, so the tip can be lowered into the cup first, then slid under the plate into the slot.
    sx, sy = lay["stripper"]
    z = lay["stripper_z"]
    parts.append(f"""
    <body name="stripper" pos="{sx} {sy} {z}">
        <geom name="stripper_a" type="box" pos="-0.0105 -0.013 0" size="0.008 0.013 0.0015" rgba="0.2 0.55 0.9 1"/>
        <geom name="stripper_b" type="box" pos="0.0105 -0.013 0" size="0.008 0.013 0.0015" rgba="0.2 0.55 0.9 1"/>
        <geom name="stripper_end" type="box" pos="0 -0.023 0" size="0.0025 0.003 0.0015" rgba="0.2 0.55 0.9 1"/>
    </body>""")
    # syringe pump on the bench (drawn; the plunger slides as volume is drawn)
    px, py = lay["pump"]
    parts.append(f"""
    <body name="pump" pos="{px} {py} {pw.BENCH_Z}">
        <geom type="box" pos="0 -0.03 0.02" size="0.015 0.012 0.02" rgba="0.3 0.3 0.33 1" contype="0" conaffinity="0"/>
        <geom type="cylinder" pos="0 0.025 0.02" quat="0.7071 0.7071 0 0" size="0.006 0.03" rgba="0.85 0.9 1.0 0.5" contype="0" conaffinity="0"/>
        <body name="plunger" pos="0 0 0.02">
            <joint name="plunger_slide" type="slide" axis="0 1 0" range="0 0.05" damping="5"/>
            <geom type="box" pos="0 0 0" size="0.008 0.002 0.008" rgba="0.3 0.3 0.33 1" contype="0" conaffinity="0" mass="0.01"/>
        </body>
    </body>""")
    welds = [f'<weld name="nozzle_lock" body1="{gripper_body}" body2="nozzle" active="false" solref="0.002 1" solimp="0.99 0.999 0.0001"/>']
    welds += [f'<weld name="tip_lock_{i}" body1="nozzle" body2="tip_{i}" active="false" solref="0.002 1" solimp="0.99 0.999 0.0001"/>'
              for i in range(N_TIPS)]
    actuator = '<position name="plunger_drive" joint="plunger_slide" kp="50" ctrlrange="0 0.05"/>'
    return "\n".join(parts), "\n    ".join(welds), actuator


# ---------------------------------------------------------------------------------------------
# Robot operations (work with So101CyanoSim or CyanoSim)
# ---------------------------------------------------------------------------------------------

def body_geoms(sim, body_name):
    b = sim.model.body(body_name).id
    return set(g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == b)


def available_force(sim, direction):
    """Largest force the arm can push with along `direction` (unit vector, world frame) at the current pose,
    from its motor torque limits minus the torque it already needs to hold itself against gravity."""
    m, d = sim.model, sim.data
    n = sim.n_arm
    jac_p = np.zeros((3, m.nv))
    jac_r = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jac_p, jac_r, sim.grip_site)
    a = jac_p[:, :n].T @ np.array(direction)
    hold = d.qfrc_bias[:n]
    limit = np.array([m.actuator_forcerange[i, 1] if m.actuator_forcelimited[i] else 87.0 for i in range(n)])
    best = 1e9
    for i in range(n):
        if abs(a[i]) < 1e-9:
            continue
        bound = (limit[i] - hold[i]) / a[i] if a[i] > 0 else (limit[i] + hold[i]) / (-a[i])
        best = min(best, bound)
    return max(0.0, best)


def axial_force(sim, geoms_a, geoms_b):
    """Net vertical force between two groups of geoms (N, magnitude). A cone wedged in a ring pushes mostly
    sideways in all directions; those parts cancel, and only the vertical part presses the tip on."""
    total = np.zeros(3)
    f6 = np.zeros(6)
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if (c.geom1 in geoms_a and c.geom2 in geoms_b) or (c.geom2 in geoms_a and c.geom1 in geoms_b):
            mujoco.mj_contactForce(sim.model, sim.data, i, f6)
            frame = c.frame.reshape(3, 3)
            force = frame.T @ f6[:3]
            total = total + (force if c.geom1 in geoms_a else -force)
    return abs(total[2])


def weld(sim, name, body1, body2, on):
    eq = sim.model.equality(name).id
    if on == True:
        a = sim.data.body(body1)
        b = sim.data.body(body2)
        R = a.xmat.reshape(3, 3)
        conj = np.zeros(4)
        mujoco.mju_negQuat(conj, a.xquat)
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, conj, b.xquat)
        sim.model.eq_data[eq, 0:3] = 0.0
        sim.model.eq_data[eq, 3:6] = R.T @ (b.xpos - a.xpos)
        sim.model.eq_data[eq, 6:10] = rel
        sim.model.eq_data[eq, 10] = 1.0
    sim.data.eq_active[eq] = 1 if on == True else 0


class Liquid:
    def __init__(self, sim, robot):
        self.sim = sim
        self.robot = robot
        self.lay = layout(robot)
        self.gripper_body = "gripper" if robot == "so101" else "hand"
        self.err = PUMP_ERROR[robot]
        self.rng = np.random.default_rng(7)
        self.volumes = dict(REAGENTS)
        self.pcr_volumes = [0.0] * pcr.N_TUBES
        self.mix_composition = {"premix": 0.0, "primers": 0.0, "water": 0.0}
        self.in_tip = 0.0
        self.tip = None
        self.nozzle_geoms = body_geoms(sim, "nozzle")
        self.stripper_geoms = set(sim.model.geom(n).id for n in ["stripper_a", "stripper_b", "stripper_end"])
        self.events = []
        self.log = sim.say
        for name in REAGENTS:
            self.draw_level(name)

    # ---- bookkeeping ----

    def pipette_error(self, target):
        return target * (1 + self.rng.normal(0, self.err["cv"])) + self.rng.normal(0, self.err["abs_sd"])

    def draw_level(self, name):
        g = self.sim.model.geom(f"liquid_{name}").id
        h = max(self.volumes[name] * UL_HEIGHT, 0.0002)
        self.sim.model.geom_size[g][1] = h / 2
        self.sim.model.geom_pos[g][2] = 0.001 + h / 2

    def draw_pcr(self, k):
        g = self.sim.model.geom(f"tube{k}_mix").id
        h = max(self.pcr_volumes[k] * PCR_UL_HEIGHT, 0.0002)
        self.sim.model.geom_size[g][1] = h / 2
        self.sim.model.geom_pos[g][2] = pcr.TUBE_FLOOR_Z + h / 2

    def plunger(self):
        self.sim.data.ctrl[self.sim.model.actuator("plunger_drive").id] = min(0.05, self.in_tip / 250.0 * 0.05)

    # ---- motion ----

    def set_tool(self, end_site, geoms):
        """Measure where the tool end is relative to the gripper (a fixed tip-finding camera), after settling."""
        sim = self.sim
        sim.wait(0.5)
        sim.tip_offset = sim.data.site_xpos[sim.model.site(end_site).id] - sim.get_grip_pos()
        sim.tool_geoms = geoms
        sim.tool_tip_site = sim.model.site(end_site).id

    def descend_to(self, xy, z_end, steps=25, check=None):
        """Lower the tool end straight down to z_end in small steps; stop early if check() returns True."""
        start = self.sim.tip_pos()[2]
        for z in np.linspace(start, z_end, steps):
            self.sim.move_tip_to([xy[0], xy[1], z], duration=0.08)
            if check is not None and check() == True:
                return True
        return False

    def pick_nozzle(self):
        sim = self.sim
        nx, ny = self.lay["nozzle"]
        top = pw.BENCH_Z + 0.0002 + NOZZLE_LENGTH
        if self.robot == "so101":
            import so101_sim as s1
            sim.set_yaw(s1.TOOL_YAW)
            grip_xy = np.array([nx - sim.toothpick_aim, ny])
            sim.travel(grip_xy, top - 0.008 + 0.03, s1.JAW_TOOTHPICK_OPEN)
            sim.move_grip_to([grip_xy[0], grip_xy[1], top - 0.008], s1.JAW_TOOTHPICK_OPEN, duration=0.8)
            angle = sim.data.ctrl[5]
            while angle > s1.JAW_CLOSED:
                angle = angle - 0.003
                sim.data.ctrl[5] = angle
                sim.wait(0.02)
                f1, _ = sim.contact(sim.fixed_finger, self.nozzle_geoms)
                f2, _ = sim.contact(sim.moving_finger, self.nozzle_geoms)
                if f1 > 0.05 and f2 > 0.05:
                    break
            sim.data.ctrl[5] = sim.data.qpos[5] - 0.001
        else:
            sim.move_grip_to([nx, ny, top + 0.05], 120, duration=1.5)
            sim.move_grip_to([nx, ny, top - 0.012], 120, duration=0.8)
            sim.set_gripper(int(NOZZLE_R / 0.04 * 255))
        weld(sim, "nozzle_lock", self.gripper_body, "nozzle", True)
        sim.move_grip_to([sim.get_grip_pos()[0], sim.get_grip_pos()[1], top + 0.03], sim.data.ctrl[sim.gripper_index], duration=1.0)
        if hasattr(sim, "holding_tool"):
            sim.holding_tool = True
        self.set_tool("nozzle_end", self.nozzle_geoms)
        self.log("picked up the printed nozzle")

    def seat_tip(self, i):
        """Lower the nozzle into tip i's collar, then check whether the arm can press with SEAT_FORCE here."""
        sim = self.sim
        x, y = tip_slots(self.lay)[i]
        tip_geoms = body_geoms(sim, f"tip_{i}")
        rim = sim.data.body(f"tip_{i}").xpos[2] + TIP_LENGTH
        self.go(x, y, rim + 0.012)
        self.set_tool("nozzle_end", self.nozzle_geoms)  # re-check the nozzle end before pressing into the collar
        sim.move_tip_to([x, y, rim + 0.012], duration=0.4)
        first_contact = [None]

        def touching():
            f, _ = sim.contact(self.nozzle_geoms, tip_geoms)
            if f > 0.05 and first_contact[0] is None:
                first_contact[0] = sim.tip_pos().copy()
            return False

        # the nozzle cone meets the collar ring about 3 mm below the rim; stop there and press
        self.descend_to((x, y), rim - 0.0035, steps=30, check=touching)
        capacity = available_force(sim, [0.0, 0.0, -1.0])
        p = first_contact[0]
        aligned = p is not None and abs(p[0] - x) < 0.0008 and abs(p[1] - y) < 0.0008
        sealed = aligned == True and capacity >= SEAT_FORCE
        result = {"tip": i + 1, "press_capacity_N": round(capacity, 2), "aligned": bool(aligned), "sealed": bool(sealed)}
        if sealed == True:
            weld(sim, f"tip_lock_{i}", "nozzle", f"tip_{i}", True)
            self.tip = i
            sim.move_tip_to([x, y, rim + 0.06], duration=1.0)
            self.set_tool(f"tip_end_{i}", self.nozzle_geoms | tip_geoms)
        else:
            sim.move_tip_to([x, y, rim + 0.03], duration=0.6)
        self.log(f"tip {i + 1}: nozzle {'in' if aligned else 'NOT in'} the collar, arm can press {capacity:.1f} N here "
                 f"(needs {SEAT_FORCE:.0f} N): {'sealed' if sealed else 'NOT sealed'}")
        self.events.append({"step": "seat", **result})
        return sealed

    def go(self, x, y, z):
        if self.robot == "so101":
            pcr.go_tip(self.sim, (x, y), z)
            return
        sim = self.sim  # Panda: rise so the tip clears the tool rack and plates, move across, then descend
        here = sim.tip_pos()
        high = max(here[2], pw.BENCH_Z + 0.10)
        sim.move_tip_to([here[0], here[1], high], duration=0.6)
        sim.move_tip_to([x, y, high], duration=1.2)
        sim.move_tip_to([x, y, z], duration=0.8)

    def aspirate(self, name, volume):
        sim = self.sim
        x, y = reagent_xy(self.lay, name)
        floor = pw.BENCH_Z + 0.001
        level = floor + self.volumes[name] * UL_HEIGHT
        target = max(floor + 0.0006, level - 0.0015)
        self.go(x, y, pw.BENCH_Z + TUBE_DEPTH + 0.006)
        self.descend_to((x, y), target, steps=25)
        sim.wait(0.2)
        immersed = sim.tip_pos()[2] < level - 0.0003
        got = self.pipette_error(volume) if immersed == True else 0.0
        got = min(got, self.volumes[name])
        self.volumes[name] = self.volumes[name] - got
        self.in_tip = self.in_tip + got
        if name == "mix":
            pass
        self.draw_level(name)
        self.plunger()
        sim.wait(0.4)
        self.descend_to((x, y), pw.BENCH_Z + TUBE_DEPTH + 0.006, steps=10)
        self.events.append({"step": "aspirate", "from": name, "target_uL": volume, "got_uL": round(got, 2),
                            "tip_immersed": bool(immersed)})
        return got

    def dispense_reagent(self, name, source):
        """Dispense the whole tip into the mix tube, touching the tube wall near the liquid surface."""
        sim = self.sim
        x, y = reagent_xy(self.lay, name)
        floor = pw.BENCH_Z + 0.001
        level = floor + self.volumes[name] * UL_HEIGHT
        self.go(x, y, pw.BENCH_Z + TUBE_DEPTH + 0.006)
        self.descend_to((x, y), max(floor + 0.0008, level + 0.0005), steps=20)
        out = self.in_tip
        self.volumes[name] = self.volumes[name] + out
        self.mix_composition[source] = self.mix_composition[source] + out
        self.in_tip = 0.0
        self.draw_level(name)
        self.plunger()
        sim.wait(0.4)
        self.descend_to((x, y), pw.BENCH_Z + TUBE_DEPTH + 0.006, steps=10)
        self.events.append({"step": "dispense", "into": name, "uL": round(out, 2)})

    def mix_by_pipetting(self, cycles=5):
        sim = self.sim
        x, y = reagent_xy(self.lay, "mix")
        floor = pw.BENCH_Z + 0.001
        self.go(x, y, pw.BENCH_Z + TUBE_DEPTH + 0.006)
        level = floor + self.volumes["mix"] * UL_HEIGHT
        self.descend_to((x, y), max(floor + 0.0006, level - 0.0015), steps=25)
        immersed = sim.tip_pos()[2] < level - 0.0003
        for c in range(cycles):
            self.in_tip = 0.6 * self.volumes["mix"]
            self.plunger()
            sim.wait(0.3)
            self.in_tip = 0.0
            self.plunger()
            sim.wait(0.3)
        self.descend_to((x, y), pw.BENCH_Z + TUBE_DEPTH + 0.006, steps=10)
        self.events.append({"step": "mix", "cycles": cycles, "tip_immersed": bool(immersed)})
        self.log(f"mixed the master mix by pipetting ({cycles} cycles)")

    def dispense_pcr(self, k, centers):
        sim = self.sim
        x, y = centers[k]
        top = pw.BENCH_Z + pcr.TUBE_FLOOR_Z + pcr.TUBE_DEPTH
        self.go(x, y, top + 0.008)
        self.descend_to((x, y), pw.BENCH_Z + pcr.TUBE_FLOOR_Z + 0.0008, steps=25)
        out = self.in_tip
        self.pcr_volumes[k] = self.pcr_volumes[k] + out
        self.in_tip = 0.0
        self.draw_pcr(k)
        self.plunger()
        sim.wait(0.3)
        self.descend_to((x, y), top + 0.008, steps=10)
        self.events.append({"step": "dispense_pcr", "tube": k + 1, "uL": round(out, 2)})

    def strip_tip(self):
        """Slide the nozzle into the stripper slot with the collar under the plate, then lift."""
        sim = self.sim
        if self.tip is None:
            return False
        i = self.tip
        sx, sy = self.lay["stripper"]
        plate_bottom = self.lay["stripper_z"] - 0.0015
        tip_geoms = body_geoms(sim, f"tip_{i}")
        # the collar top sits at the nozzle end + 9 mm insertion; put it 1.5 mm under the plate
        collar_top_above_end = TIP_LENGTH
        end_z = plate_bottom - 0.0015 - collar_top_above_end
        open_half = (sx, sy + 0.012)  # over the uncovered half of the cup
        self.go(open_half[0], open_half[1], end_z + 0.05)
        sim.move_tip_to([open_half[0], open_half[1], end_z], duration=1.0)  # lower the tip into the cup
        sim.move_tip_to([sx, sy - 0.010, end_z], duration=1.2)  # slide under the plate into the slot
        caught = [False]

        def touching():
            f, _ = sim.contact(tip_geoms, self.stripper_geoms)
            if f > 0.05:
                caught[0] = True
                return True
            return False

        self.descend_to((sx, sy - 0.010), end_z + 0.004, steps=20, check=touching)  # lift until the flange meets the plate
        capacity = available_force(sim, [0.0, 0.0, 1.0])
        stripped = caught[0] == True and capacity >= STRIP_FORCE
        peak = [capacity]
        if stripped == True:
            weld(sim, f"tip_lock_{i}", "nozzle", f"tip_{i}", False)
            sim.wait(0.6)
        if stripped == True:
            sim.move_tip_to([sx, sy - 0.010, end_z + 0.02], duration=0.5)  # nozzle alone rises through the slot
            sim.move_tip_to([sx, sy + 0.012, end_z + 0.06], duration=0.8)
        else:
            sim.move_tip_to([open_half[0], open_half[1], end_z], duration=0.8)
            sim.move_tip_to([open_half[0], open_half[1], end_z + 0.05], duration=0.8)
        fell_in = False
        if stripped == True:
            p = sim.data.body(f"tip_{i}").xpos
            fell_in = abs(p[0] - sx) < 0.026 and abs(p[1] - sy) < 0.026 and p[2] < self.lay["stripper_z"]
            self.tip = None
            self.set_tool("nozzle_end", self.nozzle_geoms)
        self.log(f"flange {'caught' if caught[0] else 'NOT caught'} under the plate, arm can lift {peak[0]:.1f} N (needs {STRIP_FORCE:.0f} N): tip {'stripped' if stripped else 'NOT stripped'}"
                 f"{', landed in the waste cup' if fell_in else ''}")
        self.events.append({"step": "strip", "tip": i + 1, "lift_capacity_N": round(peak[0], 2), "caught": bool(caught[0]), "stripped": bool(stripped),
                            "in_waste": bool(fell_in)})
        return stripped


def run_master_mix(sim, robot, centers):
    """Prepare master mix for 8 colony PCRs and aliquot it into the PCR strip."""
    lq = Liquid(sim, robot)
    if hasattr(sim, "calibrate"):
        sim.calibrate()
    lq.pick_nozzle()
    tip = 0
    for name, per_rxn in RECIPE:
        while lq.seat_tip(tip) == False:
            tip = tip + 1
            if tip >= N_TIPS:
                return lq, summarize(lq)
        tip = tip + 1
        target = per_rxn * N_RXN * EXTRA
        chunks = int(math.ceil(target / 180.0))
        for c in range(chunks):
            lq.aspirate(name, target / chunks)
            lq.dispense_reagent("mix", name)
        lq.log(f"added {target:.1f} uL {name} to the mix tube")
        if name != "primers":
            lq.strip_tip()
    lq.mix_by_pipetting()
    for k in range(pcr.N_TUBES):
        lq.aspirate("mix", REACTION_VOLUME)
        lq.dispense_pcr(k, centers)
    lq.log("aliquoted master mix into the 8 PCR tubes")
    lq.strip_tip()
    return lq, summarize(lq)


def summarize(lq):
    total = sum(lq.mix_composition.values())
    frac = {k: (v / total if total > 0 else 0.0) for k, v in lq.mix_composition.items()}
    ideal = {name: per / REACTION_VOLUME for name, per in RECIPE}
    in_tol = [abs(v - REACTION_VOLUME) / REACTION_VOLUME <= TOLERANCE for v in lq.pcr_volumes]
    seats = [e for e in lq.events if e["step"] == "seat"]
    strips = [e for e in lq.events if e["step"] == "strip"]
    return {
        "pcr_volumes_uL": [round(v, 2) for v in lq.pcr_volumes],
        "pcr_tubes_in_tolerance": sum(in_tol),
        "mix_fraction": {k: round(v, 4) for k, v in frac.items()},
        "mix_fraction_error_pct": {k: round(100 * (frac[k] - ideal[k]) / ideal[k], 1) for k in ideal},
        "tips_sealed": sum(1 for e in seats if e["sealed"]), "tip_attempts": len(seats),
        "press_capacity_N": [e["press_capacity_N"] for e in seats],
        "tips_stripped": sum(1 for e in strips if e["stripped"]), "strip_attempts": len(strips),
        "lift_capacity_N": [e["lift_capacity_N"] for e in strips],
        "tips_in_waste": sum(1 for e in strips if e.get("in_waste")),
        "aspirations_immersed": all(e["tip_immersed"] for e in lq.events if e["step"] in ("aspirate", "mix")),
        "time_s": round(lq.sim.data.time, 1),
    }


def monte_carlo(robot, runs=20000, seed=0, tolerance=TOLERANCE):
    """Volume error budget for the same pipetting sequence, without the robot (pump error model only)."""
    err = PUMP_ERROR[robot]
    rng = np.random.default_rng(seed)
    tubes_ok = 0
    mixes_ok = 0
    for _ in range(runs):
        comp = {}
        for name, per in RECIPE:
            target = per * N_RXN * EXTRA
            chunks = int(math.ceil(target / 180.0))
            comp[name] = sum(target / chunks * (1 + rng.normal(0, err["cv"])) + rng.normal(0, err["abs_sd"]) for _ in range(chunks))
        total = sum(comp.values())
        if all(abs(comp[n] / total - per / REACTION_VOLUME) / (per / REACTION_VOLUME) <= tolerance for n, per in RECIPE):
            mixes_ok += 1
        vols = REACTION_VOLUME * (1 + rng.normal(0, err["cv"], pcr.N_TUBES)) + rng.normal(0, err["abs_sd"], pcr.N_TUBES)
        tubes_ok += int(np.sum(np.abs(vols - REACTION_VOLUME) / REACTION_VOLUME <= tolerance))
    return {"runs": runs, "tolerance_pct": round(tolerance * 100), "mix_ratios_within": round(mixes_ok / runs, 4),
            "pcr_tubes_within": round(tubes_ok / (runs * pcr.N_TUBES), 4)}
