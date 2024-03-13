import numpy as np
from numba import njit

from custom_types import ndarray


@njit
def magnetization(lattice: ndarray[int]) -> float:
    return np.sum(lattice) / len(lattice.flat)


@njit
def energy(lattice: ndarray[int], b_field: float) -> float:
    lattice_offset_x = np.roll(lattice, 1, 0)
    lattice_offset_y = np.roll(lattice, 1, 1)

    neighbor_spin_energy = -1 * np.sum(lattice * (lattice_offset_x + lattice_offset_y))

    field_energy = -b_field * np.sum(lattice)

    return neighbor_spin_energy + field_energy


@njit
def fast_neighbor_sum(lattice: ndarray[int], neighbor_indices: ndarray[int]) -> float:

    cum_sum: float = 0.0
    for index in neighbor_indices:
        cum_sum += float(lattice[index[0], index[1]])

    return cum_sum


@njit
def delta_energy(
    lattice: ndarray[int], cell_index: ndarray[int], b_field: float
) -> float:

    i, j = cell_index
    spin = lattice[i, j]

    neighbor_sum = (
        lattice[(i + 1) % lattice.shape[0], j]
        + lattice[i, (j + 1) % lattice.shape[1]]
        + lattice[(i - 1) % lattice.shape[0], j]
        + lattice[i, (j - 1) % lattice.shape[1]]
    )

    return float(2 * spin * neighbor_sum + 2 * b_field * spin)
