import numpy as np
from numba import njit

from custom_types import ndarray
from helpers import delta_energy, magnetization


@njit
def metropolis_hasting(
    lattice: ndarray[int],
    temperature: float,
    b_field: float,
    cell_index: ndarray[int],
) -> bool:

    if (delta_energy(lattice, cell_index, b_field) < 0) or (
        np.random.random() < np.exp(-temperature)
    ):
        return True
    return False


@njit
def mcmc_full(
    lattice: ndarray[int], temperature: float, b_field: float, total_steps: int
) -> ndarray[float]:

    history = np.empty(total_steps)
    temp_lattice = np.copy(lattice)
    cell_indices = np.random.randint(0, lattice.shape[0], size=(total_steps, 2))

    for t, cell_index in enumerate(cell_indices):
        if metropolis_hasting(temp_lattice, temperature, b_field, cell_index):
            temp_lattice[cell_index] *= -1
        history[t] = magnetization(temp_lattice)

    return history


def simulate(
    lattice_size: int,
    total_steps: int,
    samples: int,
    temperature_range: list[float],
    b_field_range: list[float],
) -> tuple[ndarray[float], ndarray[float], ndarray[int]]:

    lattice = np.ones((lattice_size, lattice_size))

    temperatures = np.linspace(temperature_range[0], temperature_range[1], num=samples)
    b_fields = np.linspace(b_field_range[0], b_field_range[1], num=samples)

    T, B = np.meshgrid(temperature_range, b_field_range)
    magnetization_history = np.zeros((len(B), len(T)), dtype=object)

    for t, T in enumerate(temperature_range):
        for b, B in enumerate(b_field_range):
            magnetization_history[t, b] = mcmc_full(
                lattice=lattice, temperature=T, b_field=B, total_steps=total_steps
            )

    return temperatures, b_fields, magnetization_history
