import argparse
import sys
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

    do_simulation: bool = args.simulate
    do_graph: bool = args.graph

    if label == "":
        print("Please provide a label for the simulation")
        sys.exit()

    path = Path("..") / "data"

    if not (path / label).exists():
        (path / label).mkdir()
    
    sim_path = str(path / label / "history.npy")

    if do_simulation:
        temperature, b_field, magnetization_history = simulate(
            lattice_size, total_steps, samples
        )
        np.savez(sim_path, temperature=temperature, bfield=b_field, magnetization_history=magnetization_history)
    else:
        saved_arrays = np.load(sim_path)
        temperature = saved_arrays['temperature']
        b_field = saved_arrays['bfield']
        magnetization_history = saved_arrays['magnetization_history']

    if do_graph:
        magnetization_no_burn = magnetization_history[burn_in_steps:]
        magnetization = np.mean(magnetization_no_burn, axis=1)
        #plot_3D(temperature, b_field, magnetization, 0, path / label)
        #plot_3D(temperature, b_field, magnetization, 45, path / label)
        #plot_3D(temperature, b_field, magnetization, 90, path / label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lattice_size", type=int, default="10")
    parser.add_argument("--burn_in_steps", type=int, default="1000")
    parser.add_argument("--total_steps", type=int, default="2000")
    parser.add_argument("--samples", type=int, default="3")

    parser.add_argument(
        "--simulate", 
        action='store_true'
    )
    parser.add_argument("--sim_path", type=str, default="")
    parser.add_argument("--label", type=str, default="")

    parser.add_argument(
        "--graph", 
        action='store_true'
    )
    parser.add_argument("--graph_path", type=str, default="")

    args = parser.parse_args()

    main(args)
