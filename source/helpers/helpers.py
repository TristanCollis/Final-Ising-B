import numpy as np
from custom_types import ndarray
from numba import njit


@njit
def magnetization(lattice: ndarray[int]) -> float:
    return np.sum(lattice) / len(lattice.flat)


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


def delta_energy(
    lattice: ndarray[int], cell_index: ndarray[int], b_field: float
) -> float:
    neighbor_offsets = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int64)
    neighbor_indices = np.mod(
        neighbor_offsets + cell_index, lattice.shape[0], dtype=np.int64
    )

    neighbor_delta_E = (
        2
        * lattice[*cell_index]
        * fast_neighbor_sum(lattice, neighbor_indices)
        # * np.sum([lattice[*index] for index in neighbor_indices])
    )

    field_delta_E = 2 * b_field * lattice[*cell_index]

    return float(neighbor_delta_E + field_delta_E)
