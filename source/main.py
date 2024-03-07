from configparser import ConfigParser

import numpy as np

from source.constants.constants import T_critical
from source.helpers.helpers import mcmc_full, get_config_section


def main():
    config_path = "./config.ini"
    config = get_config_section(config_path)

    lattice_size = int(config["lattice_size"])
    burn_in_steps = int(config["burn_in_steps"])
    total_steps = int(config["total_steps"])
    samples = int(config["samples"])

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
