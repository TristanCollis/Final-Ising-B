from configparser import ConfigParser

import numpy as np

from source.constants.constants import T_critical
from source.helpers.helpers import mcmc_full


def main():
    config_path = "./config.ini"
    config = ConfigParser()
    config.read(config_path)

    lattice_size = int(config["DEFAULT"]["lattice_size"])
    burn_in_steps = int(config["DEFAULT"]["burn_in_steps"])
    total_steps = int(config["DEFAULT"]["total_steps"])
    samples = int(config["DEFAULT"]["samples"])

    lattice = np.ones((lattice_size, lattice_size), dtype=int)
    magnetization_history = np.empty((samples, samples, total_steps), dtype=float)

    for t, T in enumerate(np.linspace(T_critical / 2, T_critical * 1.5, num=samples)):
        for b, B in enumerate(np.linspace(-1, 1, num=samples)):
            magnetization_history[t, b] = mcmc_full(
                lattice=lattice, temperature=T, b_field=B, total_steps=total_steps
            )

    truncated_magnetization_history = magnetization_history[:, :, burn_in_steps:]


if __name__ == "__main__":
    main()
