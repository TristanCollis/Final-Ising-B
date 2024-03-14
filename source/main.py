import argparse
import sys
import logging
from pathlib import Path

import numpy as np
from simulate import simulate
from graph import plot_3D

def main(args):
    lattice_size: int = args.lattice_size
    burn_in_steps: int = args.burn_in_steps
    total_steps: int = args.total_steps
    samples: int = args.samples

    label: str = args.label

    temperature_range: list[float] = args.temperature
    b_field_range: list[float] = args.magnetic_field

    do_simulation: bool = args.simulate
    do_graph: bool = args.graph

    force: bool = args.force

    if label == "":
        print("Please provide a label for the simulation")
        sys.exit()

    path = Path(".")

    if not (path / "data").exists():
        (path / "data").mkdir()
    path = path / "data"

    if force:
        if (path / label).exists():
            import shutil
            shutil.rmtree(path / label)

    if not (path / label).exists():
        (path / label).mkdir()
    else:
        print(f"Label {label} already exists, please provide a different label or run with -f")
        sys.exit()
    
    sim_path = path / label / "history"
    log_path = path / label / "log.log"

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    if do_simulation:
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num=samples)
        b_fields = np.linspace(b_field_range[0], b_field_range[1], num=samples)

        magnetization_history = simulate(
            lattice_size, total_steps, temperatures, b_fields
        )
        np.savez(sim_path, temperature=temperatures, bfield=b_fields, magnetization_history=magnetization_history)
    else:
        saved_arrays = np.load(sim_path)
        temperatures = saved_arrays['temperature']
        b_fields = saved_arrays['bfield']
        magnetization_history = saved_arrays['magnetization_history']


    if do_graph:
        magnetization_with_burn = np.mean(magnetization_history[:, :, burn_in_steps:], axis=2)

        T, B = np.meshgrid(temperatures, b_fields)

        plot_3D(T, B, magnetization_with_burn, 0, path / label)
        plot_3D(T, B, magnetization_with_burn, 45, path / label)
        plot_3D(T, B, magnetization_with_burn, 90, path / label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lattice_size", 
        type=int, 
        default="10"
    )

    parser.add_argument(
        "--burn_in_steps", 
        type=int, 
        default="1000"
    )

    parser.add_argument(
        "--total_steps",
        "-steps", 
        type=int, 
        default="5000"
    )

    parser.add_argument(
        "--samples", 
        type=int, 
        default="9"
    )

    parser.add_argument(
        "--label", 
        type=str, 
        default=""
    )

    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        nargs=2,
        default=[1, 5]
    )

    parser.add_argument(
        "--magnetic_field",
        "-b",
        type=float,
        nargs=2,
        default=[-5, 5]
    )

    parser.add_argument(
        "--simulate", 
        action='store_true'
    )

    parser.add_argument(
        "--graph", 
        action='store_true'
    )

    parser.add_argument(
        "-f",
        "--force",
        action='store_true'
    )

    args = parser.parse_args()

    main(args)
