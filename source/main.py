import argparse
import sys
import logging
from pathlib import Path

import numpy as np
from simulate import simulate
from graph import plot_3D
from custom_types import ndarray

def main(args):
    lattice_size: int = args.lattice_size
    burn_in_steps: int = args.burn_in_steps
    total_steps: int = args.total_steps
    samples: int = args.samples

    label: str = args.label

    temperature: list[float] = args.temperature
    b_field: list[float] = args.magnetic_field

    do_simulation: bool = args.simulate
    do_graph: bool = args.graph

    force: bool = args.force

    if label == "":
        print("Please provide a label for the simulation")
        sys.exit()

    path = Path("..") / "data"

    if force:
        if (path / label).exists():
            import shutil
            shutil.rmtree(path / label)

    if not (path / label).exists():
        (path / label).mkdir()
    else:
        print(f"Label {label} already exists, please provide a different label or run with -f")
        sys.exit()
    
    sim_path = str(path / label / "history")

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(path / label / "log.txt")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    if do_simulation:
        temperature, b_field, magnetization_history = simulate(
            lattice_size, total_steps, samples, temperature, b_field
        )
        np.savez(sim_path, temperature=temperature, bfield=b_field, magnetization_history=magnetization_history)
    else:
        saved_arrays = np.load(sim_path)
        temperature = saved_arrays['temperature']
        b_field = saved_arrays['bfield']
        magnetization_history = saved_arrays['magnetization_history']

    if do_graph:
        magnetization_with_burn = np.empty_like(magnetization_history)
        for i in range(magnetization_history.shape[0]):
            for j in range(magnetization_history.shape[1]):
                magnetization_with_burn[i, j] = np.mean(magnetization_history[i, j][burn_in_steps:])

        plot_3D(temperature, b_field, magnetization_with_burn, 0, path / label)
        plot_3D(temperature, b_field, magnetization_with_burn, 44, path / label)
        plot_3D(temperature, b_field, magnetization_with_burn, 90, path / label)


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
        default="2000"
    )

    parser.add_argument(
        "--samples", 
        type=int, 
        default="3"
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
        nargs=2
    )

    parser.add_argument(
        "--magnetic_field",
        "-b",
        type=float,
        default=2,
        nargs=2
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
