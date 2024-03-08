import numpy as np
from custom_types import ndarray


def magnetization(lattice: ndarray[int]) -> float:
    return np.sum(lattice) / np.size(lattice)


def energy(lattice: ndarray[int], b_field: float) -> float:
    lattice_offset_x = np.roll(lattice, 1, 0)
    lattice_offset_y = np.roll(lattice, 1, 1)

    neighbor_spin_energy = -1 * np.sum(lattice * (lattice_offset_x + lattice_offset_y))

    field_energy = -b_field * np.sum(lattice)

    return neighbor_spin_energy + field_energy


def delta_energy(
    lattice: ndarray[int], cell_index: tuple[int] | ndarray[int], b_field: float
) -> float:
    neighbor_offsets = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    neighbor_indices = np.mod(neighbor_offsets + cell_index, lattice.shape)

    neighbor_delta_E = (
        2
        * lattice[*cell_index]
        * np.sum([lattice[*index] for index in neighbor_indices])
    )

    field_delta_E = 2 * b_field * lattice[*cell_index]

    return neighbor_delta_E + field_delta_E
