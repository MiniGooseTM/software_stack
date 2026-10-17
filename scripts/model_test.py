"""Load a MuJoCo model and print a short summary.

Usage:
    python scripts/model_test.py                 # check the training model
    python scripts/model_test.py --xml goose.xml  # check another model
    python scripts/model_test.py --view          # also open the interactive viewer
"""

import argparse
from pathlib import Path

import mujoco

ASSETS = Path(__file__).resolve().parents[1] / "src" / "self_balancing_robot" / "assets"


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a MuJoCo model and print a summary.")
    parser.add_argument("--xml", type=Path, default=ASSETS / "XML-test.xml")
    parser.add_argument("--view", action="store_true", help="Open the interactive viewer.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"Loaded {args.xml.name}")
    print(
        f"  bodies: {model.nbody}, joints: {model.njnt}, "
        f"geoms: {model.ngeom}, actuators: {model.nu}"
    )
    print(f"  robot mass: {model.body_subtreemass[model.body('base_link').id]:.4f} kg")

    if args.view:
        from mujoco import viewer

        viewer.launch(model, data)


if __name__ == "__main__":
    main()
