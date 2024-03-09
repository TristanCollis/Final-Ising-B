import numpy as np
from constants import T_critical
from custom_types import ndarray
from helpers import delta_energy, magnetization


def mcmc_step(
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


def mcmc_full(
    lattice: ndarray[int],
    temperature: float,
    b_field: float,
    total_steps: int,
    burn_in_steps: int,
) -> ndarray[float]:

    history = np.empty(total_steps - burn_in_steps)

    temp_lattice = np.copy(lattice)

    cell_indices = np.random.randint(0, lattice.shape[0], size=(total_steps, 2), dtype=np.int64)

    for t, cell_index in enumerate(cell_indices[:burn_in_steps]):
        if mcmc_step(
            temp_lattice, temperature, b_field, cell_index
        ):
            temp_lattice[*cell_index] *= -1

    for t, cell_index in enumerate(cell_indices[burn_in_steps:]):
        if mcmc_step(
            temp_lattice, temperature, b_field, cell_index
        ):
            temp_lattice[*cell_index] *= -1

        history[t] = magnetization(temp_lattice)

    return history


def simulate(
    lattice_size: int, burn_in_steps: int, total_steps: int, samples: int
) -> ndarray[int]:
    lattice = np.ones((lattice_size, lattice_size))
    magnetization_history = np.empty((samples, samples, total_steps-burn_in_steps), dtype=float)

    for t, T in enumerate(np.linspace(T_critical / 2, T_critical * 1.5, num=samples)):
        for b, B in enumerate(np.linspace(-1, 1, num=samples)):
            magnetization_history[t, b] = mcmc_full(
                lattice=lattice, temperature=T, b_field=B, total_steps=total_steps, burn_in_steps=burn_in_steps
            )

    

    return magnetization_history
