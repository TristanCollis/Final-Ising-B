import copy
import numpy as np
from numba import njit

from custom_types import ndarray
from helpers import compareH

@njit
def mcmc_full(
    lattice: ndarray[int], 
    temperature: float, 
    b_field: float,
    total_steps: int
) -> ndarray[float]:

    history = np.empty(total_steps)
    cell_indices = np.random.randint(0, lattice.shape[0], size=(total_steps, 2))

    for t, cell_index in enumerate(cell_indices):
        i, j = cell_index
        spin = lattice[i, j]

        neighbor_sum = (
            lattice[(i + 1) % lattice.shape[0], j]
            + lattice[i, (j + 1) % lattice.shape[1]]
            + lattice[(i - 1) % lattice.shape[0], j]
            + lattice[i, (j - 1) % lattice.shape[1]]
        )
        energy_diff = 2 * spin * neighbor_sum + 2 * b_field * spin
        if energy_diff <= 0 or np.random.rand() < np.exp(-energy_diff / temperature):
            lattice[i, j] *= -1
        history[t] = np.sum(lattice) / lattice.size

    return history


def simulate(
    lattice_size: int,
    total_steps: int,
    temperatures: ndarray[float],
    b_fields: ndarray[float],
) -> ndarray[int]:

    lattice = np.random.choice((-1, 1), (lattice_size, lattice_size))
    magnetization_history = np.zeros((len(temperatures), len(b_fields), total_steps), dtype=float)

    for j, B in enumerate(b_fields):
        for i, T in enumerate(temperatures):
            temp_lattice = copy.deepcopy(lattice)
            magnetization_history[j, i, :] = mcmc_full(
                lattice=temp_lattice, temperature=T, b_field=B, total_steps=total_steps
            )

    return magnetization_history

@njit
def burnout(
    total_steps: int,
    percent: float,
    error: float,
    magnetization: ndarray[float]
) -> float:
    Msize = len(magnetization) - 1
    h1 = magnetization[Msize]           #100%
    h2 = magnetization[int(Msize* 0.5)] #50%

    flip = -1 

    if abs(h1 - h2) < error:
        for i in range(total_steps):
            n = 2**(i+2)
            percent += 1/(flip * n)
            flip = compareH(magnetization[int(Msize * percent)], h1, error)

    if abs(h1 - h2) > error:
        return np.nan
    return percent * (Msize+1)