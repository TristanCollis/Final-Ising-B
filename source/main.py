import argparse

import numpy as np

from source.constants.constants import T_critical
from source.helpers.helpers import mcmc_full

def main(args):
    lattice_size = args.lattice_size
    burn_in_steps = args.burn_in_steps
    total_steps = args.total_steps
    samples = args.samples

    lattice = np.ones((lattice_size, lattice_size), dtype=int)
    magnetization_history = np.empty((samples, samples, total_steps), dtype=float)

    for t, T in enumerate(np.linspace(T_critical / 2, T_critical * 1.5, num=samples)):
        for b, B in enumerate(np.linspace(-1, 1, num=samples)):
            magnetization_history[t, b] = mcmc_full(
                lattice=lattice, temperature=T, b_field=B, total_steps=total_steps
            )

    truncated_magnetization_history = magnetization_history[:, :, burn_in_steps:]


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
        type=int, 
        default="2000"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default="3"
    )

    args = parser.parse_args()
    main(args)
