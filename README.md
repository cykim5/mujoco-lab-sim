# MuJoCo Lab Sim: vial rack pick-and-place

A simulated Franka Panda arm moves vials from a lab bench into a rack.
Step 1 of a sim-to-real / real-to-sim project inspired by Gemini 4 Hydrogen for robotics.

## Setup (already done on this Mac)

The Python environment lives outside OneDrive so it does not sync thousands of files:

```bash
python3 -m venv ~/.venvs/mujoco-lab
~/.venvs/mujoco-lab/bin/pip install -r requirements.txt
```

## Run

Scripted pick-and-place (saves images and a GIF to `outputs/`):

```bash
~/.venvs/mujoco-lab/bin/python pick_and_place.py
```

Open the scene in the interactive 3D viewer (drag to rotate, double-click to select, Ctrl+drag to push things):

```bash
~/.venvs/mujoco-lab/bin/python -m mujoco.viewer --mjcf=models/lab_scene.xml
```

## Files

| File | What it is |
|---|---|
| `models/lab_scene.xml` | The lab: bench, 2x3 vial rack, three capped vials, two cameras |
| `models/panda_lab.xml` | Menagerie's Panda with a `grip` site added between the fingertips and its keyframe removed |
| `assets/franka_emika_panda/` | Unchanged copy of the Panda from MuJoCo Menagerie (meshes, license) |
| `lab_sim.py` | Helpers: reset, inverse kinematics, smooth motion, success check, image/GIF saving |
| `pick_and_place.py` | Hand-written baseline plan: red to A1, blue to A2, green to A3 |
| `outputs/` | Rendered images and GIF |

## Baseline result

3/3 vials placed, each about 5 mm off the slot center (inside the 5 mm clearance).

## Next steps

1. Domain randomization: shuffle vial positions, friction, mass, lighting, camera pose. Measure success rate over many runs.
2. Gemini in the loop: send the `lab_cam` image to `gemini-4-hydrogen`, have it find the vials and pick slots, compare to the baseline.
3. Real-to-sim: rebuild this scene from a photo of a real bench.
