import numpy as np
from constants import T_critical
from custom_types import ndarray
from helpers import delta_energy, magnetization


def mcmc_step(
    lattice: ndarray[int], temperature: float, b_field: float
) -> ndarray[int]:

    new_lattice = np.copy(lattice)

    while True:
        cell_index = np.random.randint(0, lattice.shape)

        if delta_energy(lattice, cell_index, b_field) < 0:
            break

        if np.random.random() < np.exp(-temperature):
            break

    new_lattice[*cell_index] *= -1

    return new_lattice


def mcmc_full(
    lattice: ndarray[int],
    temperature: float,
    b_field: float,
    total_steps: int,
) -> ndarray[float]:

    history = np.empty(total_steps)

    temp_lattice = np.copy(lattice)

    for t in range(total_steps):
        temp_lattice = mcmc_step(temp_lattice, temperature, b_field)

        history[t] = magnetization(temp_lattice)

    return history


def simulate(
    lattice_size: int, burn_in_steps: int, total_steps: int, samples: int
) -> ndarray[int]:
    lattice = np.ones((lattice_size, lattice_size), dtype=int)
    magnetization_history = np.empty((samples, samples, total_steps), dtype=float)

    for t, T in enumerate(np.linspace(T_critical / 2, T_critical * 1.5, num=samples)):
        for b, B in enumerate(np.linspace(-1, 1, num=samples)):
            magnetization_history[t, b] = mcmc_full(
                lattice=lattice, temperature=T, b_field=B, total_steps=total_steps
            )

    truncated_magnetization_history = magnetization_history[:, :, burn_in_steps:]

    return truncated_magnetization_history
