# MuJoCo Lab Sim

Simulated lab automation in MuJoCo, part of a sim-to-real / real-to-sim project inspired by Gemini 4 Hydrogen for robotics.

**Main project: cyanobacteria colony picking and quadrant restreaking**, a workflow a cyano researcher wants automated. Today's colony pickers (like the QPix) are built for E. coli, and no robot streaks from plate to plate. Two versions:

| | **Reference** (Panda) | **Build** (SO-101) |
|---|---|---|
| Robot | Franka Panda, a lab-grade 7-joint research arm | SO-101, a cheap 3D-printed 5-joint arm |
| Question | Can a robot do this workflow at all? | Can something you could actually build do it? |
| Lids | Suction pad | Our printed jaw with a V-shaped lid hook |
| Tools | Loop or toothpick from a 24-slot rack | 24 toothpicks jumbled in a cup: shake, then pick a separated one |
| Motors | Ideal (gravity compensated) | Realistic: weak servos, no gravity compensation, hidden calibration error |
| Sections | [1](#1-reference-version-panda) | [2](#2-build-version-so-101) |

Plus a warm-up vial demo ([3](#3-vial-rack-pick-and-place-warm-up)).

## 1. Reference version (Panda)

![Robot taking lids off, picking a colony with a plastic loop and quadrant streaking a fresh plate](outputs/cyano_loop.gif)

| Colony finder (red = chosen, most isolated) | Streak on the fresh plate (numbers = quadrant start, light to dark = stroke order, red would mean lifted off) |
|---|---|
| ![Detected colonies](outputs/cyano_loop_colonies.png) | ![Quadrant streak path](outputs/cyano_toothpick_streak.png) |

### What the robot does

1. Lifts the lids off both plates and parks them on the bench.
2. Photographs the source plate from above and finds every colony (green dot).
3. Chooses the colony with the most empty space around it, so it picks a single clean colony.
4. Takes the next tool from a **24-slot rack** (like a pipette tip box): a **plastic loop** or a **wooden toothpick** (you choose). It opens the fingers only 3 mm wider than the handle and lifts the tool straight up by its full length, so it never touches the neighbors.
5. Lowers the tool slowly until it feels the agar, touching down on the chosen colony. This also measures the agar height, which changes from plate to plate.
6. Quadrant streaks the fresh plate. In each quadrant it zigzags from the rim in toward the center, and each new quadrant starts by dragging out of the previous one. It presses about 0.3 mm into the agar the whole way so it never digs in.
7. Drops the tool in the waste bin and puts both lids back.

### Run it

```bash
~/.venvs/mujoco-lab/bin/python streak_plate.py
```

It asks which tool to use (1 = plastic loop, 2 = wooden toothpick). Or pass it directly:

```bash
~/.venvs/mujoco-lab/bin/python streak_plate.py --tool toothpick --seed 3
```

`--seed` changes the colony layout and agar thickness. To test many random plates with both tools:

```bash
~/.venvs/mujoco-lab/bin/python run_trials.py --trials 10
```

### Results

10 random plates per tool (different colony layouts, agar thickness and rack slot every time). Full table: [`outputs/trials.csv`](outputs/trials.csv).

| | Plastic loop | Wooden toothpick |
|---|---|---|
| Full workflow success | 10/10 | 10/10 |
| Took a tool from the rack without touching a neighbor | 10/10 | 10/10 |
| Touched a single clean colony | 10/10 | 10/10 |
| Worst miss from colony center | 0.12 mm | 0.13 mm |
| Streak in contact with agar | 100% | 100% |
| Deepest press into agar (limit 1 mm) | 0.49 mm | 0.36 mm |
| Peak streaking force | 0.30 to 0.58 N | 0.06 to 0.12 N |
| Agar height found by touch, error | within 0.26 mm | within 0.21 mm |
| Time per plate (lids off to lids back on) | about 175 s | about 174 s |

The colony finder caught 21 to 28 colonies per plate. The ones it missed were dots touching inside clumps, which merge into one blob. That's fine here, because the robot only picks isolated colonies.

**What this means:** in simulation, nothing in the motion, vision or contact control blocks a robot from doing plate-to-plate cyano restreaking with either tool. The sim is idealized, though: perfect lighting, a perfectly calibrated camera, colonies that are always clean dots, and a tool that never slips. Real plates will be harder, so these numbers are an upper bound.

Getting here surfaced real hardware requirements:
- **Gravity compensation is required.** Without it, the arm landed about 5 mm off, which is enough to miss every colony.
- **A 90 mm lid needs suction.** It's wider than a standard parallel gripper opens.
- **A toothpick needs an adapter.** A standard gripper can't hold a bare 2 mm stick securely.
- **Touch-sensing the agar is needed.** Agar thickness varies by more than 1.5 mm between plates.
- **The tool rack has design rules.** Tools need about 3 cm of space in the direction the fingers close, tall loops need a deep rack, and the arm has to lift each tool its full length before moving sideways. With those rules, all 48 test picks (every slot, both tools) came out without touching a neighbor.

### What the sim does and does not show

The sim tests whether **a robot can physically do the steps** reliably: see the colonies, touch a 1 mm target, follow a varying agar surface without gouging, and handle lids. It cannot test the biology: whether cyano cells actually stick to the tool, transfer, and grow. That needs real plates.

Modeling choices worth knowing:

| Choice | Why |
|---|---|
| Plate: 89 mm across, 15 mm tall. Agar 5.5 to 7.2 mm thick (up to about half the plate height), random per plate | Standard round microbial plate |
| Colonies: green dots 0.8 to 2 mm across, some scattered, some in clumps | "Visible dots," with real crowding so isolation matters |
| Lids are lifted with a small suction pad at the fingertips | A 90 mm lid is wider than the Panda gripper opens (80 mm). Real plate-handling robots use suction or a delidder |
| The tool is locked to the gripper once grasped (a tool adapter) | The simulated fingers do not grip thin sticks well, and real cells use adapters anyway |
| The toothpick sits in a 4 mm sleeve | The Panda fingertips cannot close tightly on a bare 2 mm toothpick. A real setup would need the same kind of adapter |
| Tools stand in a 3 x 8 rack: 3 cm apart where the fingers close, 2 cm apart the other way. The loop rack is 6 cm deep, the toothpick rack 3.5 cm | At 2 cm spacing the Panda's thick fingers hit neighbors. In a shallow rack the 12 cm loops lean into the fingers' path. A bump counts whenever the robot or its tool touches another tool |
| Plates are fixed in nests on the bench | Plates would slide while being streaked otherwise |
| Arm uses gravity compensation | Without it the arm sags about 5 mm, too much for 1 mm colonies. Real Franka arms compensate too |
| Grading uses hidden "truth" (real colony positions, real agar height) that the robot never sees | So the numbers measure what the robot actually achieved |

## 2. Build version (SO-101)

![SO-101 with printed jaw: lid hook, shaking the toothpick cup, picking one toothpick, streaking](outputs/so101_toothpick.gif)

| Colony finder | Streak on the fresh plate |
|---|---|
| ![SO-101 colony finder](outputs/so101_toothpick_colonies.png) | ![SO-101 streak](outputs/so101_toothpick_streak.png) |

```bash
~/.venvs/mujoco-lab/bin/python streak_plate.py --robot so101 --seed 2
~/.venvs/mujoco-lab/bin/python run_so101_trials.py --trials 8
```

### What is different

1. **Our printed two-zone jaw** ([`build_so101_model.py`](build_so101_model.py)). Slim fingertips (4 mm thick, 6 mm wide) pick a single toothpick. A V-shaped hook on the swinging jaw grips the lid by its rim when the jaw is opened 1.70 rad. No suction.
2. **Lids by touch.** The fixed fingertip comes down 4 mm outside the rim and slides in until it feels the rim (servo load), then the V-hook closes, which also centers the lid sideways. To put a lid back, the arm lowers it, and if it stops high (resting on the rim) it does a small spiral search until it drops in.
3. **Toothpicks from a cup.** 24 toothpicks lean together in a 3.2 cm holder on a small vibrating base. The robot shakes the cup, a depth camera finds toothpick tops (local high points), and the robot works out each stick's lean from how high its top is. It picks one where the fingertips fit between the neighbors, with the jaw opening toward the cup center so the hook hangs outside. Then it pulls straight up and checks that no other toothpick came with it. If nothing fits, it shakes again (up to 10 times).
4. **Look before touching.** A cheap arm misses by about 1 to 3 mm. The toothpick hovers 4.5 mm over the colony, the plate camera finds the tip's lowest point, the robot moves by the gap, then touches.
5. **Realistic motors.** Menagerie's SO-101 servo strength (2.94 N*m), no gravity compensation, and a hidden calibration error of up to 0.5 degrees per joint, different for every seed. The robot reads its encoders, which carry the same error, so it does not know it is off.

### Results

8 random plates. Each seed changes the colonies, agar, how the toothpicks fall, and the arm's hidden calibration error. Full table: [`outputs/so101_trials.csv`](outputs/so101_trials.csv).

| Step | Success | Notes |
|---|---|---|
| Lids off (printed hook) | 6/8 | The 2 misses are plates where no toothpick was ever picked, so the run stopped there |
| One toothpick out of the cup | 6/8 | 1 to 4 shakes when it works. 2 plates never had a toothpick with room for the fingers in 10 shakes |
| Clean colony touch | 5/6 picked | Misses 0.14 to 1.0 mm. With the camera correction: 0.14 and 0.35 mm. One plate missed by 3.5 mm (largest calibration error, tip hidden from the camera) |
| Streak (no gouging, limit 1 mm) | 4/6 | 100% contact, but two plates pressed 1.16 and 1.33 mm deep, from calibration error in height |
| Lids back on | 4/6 | Same two plates with the largest calibration error |
| **Full workflow** | **4/8** | About 3.5 minutes per plate |

**What this means:** a cheap printed arm *can* do every step of plate-to-plate cyano restreaking in simulation, including the lid, the toothpick from a jumbled cup, and the streak. But right now only about half of plates succeed end to end. The failures cluster in two places: **calibration error** (lid seating, streak depth, colony aim when the camera can't see the tip) and **toothpick separation** in the cup. Both have clear fixes to test next: a camera check for height and lid placement, and a better toothpick holder or a simple one-at-a-time dispenser. The Panda reference (20/20) shows the workflow itself is sound. The gap is the cheap hardware.

### What building it in the sim taught us

Every one of these came from the sim failing first:

- **The SO-101 needs an 8 cm riser.** Pointing straight down, it can only reach about 7 cm above its own base, too low to lift a toothpick out of a cup.
- **12 cm loops do not work** with this arm and a cup. A loop held upright is too tall to lift out. Toothpicks only.
- **A narrow, deep cup.** In a wide cup, 6.5 cm toothpicks fall flat. In a 3.2 cm by 4.5 cm holder they lean on the rim and their tops fan out about 6 mm apart.
- **Fingertips narrower than 6 mm.** Wider tips never fit between toothpick tops.
- **Stop squeezing at first contact.** Full servo force squirts a tilted toothpick out sideways. A real fingertip needs a small groove to catch the stick. The sim models that groove as a hold that switches on when both fingertips touch the same stick.
- **Where the lid hook sits matters.** The hook swings down as the jaw opens a little, so with the jaw gripping a toothpick it hangs 5.7 cm behind the fingertips at fingertip height. It must face away from the cup, or it lands on the toothpicks and blocks the jaw.
- **Remove the wrist camera mount.** It hits the SO-101's shoulder when the arm folds in to reach colonies near the base.
- **Use a V-shaped hook.** A flat hook holds the round lid wherever it lands, up to 7 mm sideways. A V centers it.
- **Calibration error is the cheap arm's real problem.** 0.5 degrees per joint becomes 1 to 3 mm at the tool tip. That's enough to miss a 1 mm colony or put a lid down on its rim. Camera checks and touch sensing fix most of it.

### Known limits

- Lid re-seating fails on plates where the calibration error is largest (the lid ends 3 to 4 mm off, on the rim). A camera check when placing, or a plate holder with a lead-in edge, would fix it.
- When the gripper blocks the camera's view of the toothpick tip, the robot skips the camera correction and touches with its own aim (about 1 mm off).
- Some shakes leave no toothpick with room for the fingers. More shakes, fewer toothpicks, or a different cup would raise the pick rate.

## 3. Vial rack pick-and-place (warm-up)

![Franka Panda placing three vials into a rack](outputs/pick_and_place.gif)

| Before | After |
|---|---|
| ![Vials on the bench](outputs/before_lab_cam.png) | ![Vials in the rack](outputs/after_lab_cam.png) |

```bash
~/.venvs/mujoco-lab/bin/python pick_and_place.py
```

Result: 3/3 vials placed, each within 0.4 mm of the slot center.

## Setup

The Python environment lives outside OneDrive so it does not sync thousands of files:

```bash
python3 -m venv ~/.venvs/mujoco-lab
~/.venvs/mujoco-lab/bin/pip install -r requirements.txt
```

Interactive 3D viewer for the vial scene (drag to rotate, Ctrl+drag to push things):

```bash
~/.venvs/mujoco-lab/bin/python -m mujoco.viewer --mjcf=models/lab_scene.xml
```

## Files

| File | What it is |
|---|---|
| `plate_world.py` | Builds the Panda cyano scene: plates, lids, agar, random colonies, 24-tool rack, waste bin, cameras |
| `cyano_sim.py` | The cyano workflow (Panda): lids, colony finder, touch-down, quadrant streak, grading |
| `so101_world.py` | Builds the SO-101 bench: compact layout, riser, toothpick cup on a shaker |
| `so101_sim.py` | The SO-101 workflow: lid hook by touch, shake and pick, look before touching, calibration error |
| `build_so101_model.py` | Builds `models/so101_lab.xml`: the Menagerie SO-101 with our printed jaw |
| `streak_plate.py` | Runs one plate (`--robot panda` or `--robot so101`) and saves pictures and a GIF |
| `run_trials.py`, `run_so101_trials.py` | Run many random plates and write `outputs/trials.csv` / `outputs/so101_trials.csv` |
| `lab_sim.py` | Shared helpers: inverse kinematics, smooth motion, images |
| `pick_and_place.py`, `models/lab_scene.xml` | The vial demo |
| `models/panda_lab.xml` | Menagerie's Panda with a `grip` site, gravity compensation, and its keyframe removed |
| `assets/franka_emika_panda/`, `assets/robotstudio_so101/` | Unchanged copies of the Panda and SO-101 from MuJoCo Menagerie (meshes, licenses) |

## Next steps

1. SO-101: camera check for lid placement and streak height, and a better toothpick holder or dispenser, to raise 4/8 toward the Panda's 20/20.
2. Colony PCR: pick colonies into a 96-well PCR plate.
3. Real-to-sim: run the colony finder on photos of real cyano plates.
4. Gemini in the loop: let `gemini-4-hydrogen` choose colonies from the camera image and compare with the hand-written finder.
