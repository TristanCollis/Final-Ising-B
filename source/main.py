import argparse

import numpy as np
from simulate import simulate


def main(args):
    lattice_size: int = args.lattice_size
    burn_in_steps: int = args.burn_in_steps
    total_steps: int = args.total_steps
    samples: int = args.samples

    do_simulation: bool = args.simulate.lower() == "true"
    sim_path: str = args.sim_path

    do_graph: bool = args.graph.lower() == "true"
    graph_path: str = args.graph_path

    if do_simulation:
        magnetization_history = simulate(
            lattice_size, burn_in_steps, total_steps, samples
        )

        if sim_path != "":
            magnetization_history.tofile(sim_path)

    else:
        magnetization_history = np.fromfile(sim_path, dtype=float)

    if do_graph:
        ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lattice_size", type=int, default="10")
    parser.add_argument("--burn_in_steps", type=int, default="1000")
    parser.add_argument("--total_steps", type=int, default="2000")
    parser.add_argument("--samples", type=int, default="3")

    parser.add_argument(
        "--simulate", type=str, choices=("true", "false"), default="true"
    )
    parser.add_argument("--sim_path", type=str, default="")

    parser.add_argument("--graph", type=str, choices=("true", "false"), default="true")
    parser.add_argument("--graph_path", type=str, default="")

    args = parser.parse_args()

    main(args)
