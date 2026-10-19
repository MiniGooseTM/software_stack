"""Quick check that env_v0.xml (goose.xml on the ground plane) loads in MuJoCo.

Usage:
    python model_test.py          # load and print a short summary
    python model_test.py --view   # also open the interactive viewer
"""
import sys
from pathlib import Path

import mujoco

XML_PATH = Path(__file__).parent / "env_v0.xml"

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"Loaded {XML_PATH.name}")
print(f"  bodies: {model.nbody}, joints: {model.njnt}, geoms: {model.ngeom}, actuators: {model.nu}")
print(f"  total mass: {model.body_subtreemass[1]:.4f} kg")

if "--view" in sys.argv:
    import mujoco.viewer
    mujoco.viewer.launch(model, data)