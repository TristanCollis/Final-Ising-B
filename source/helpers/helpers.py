import numpy as np
from custom_types import ndarray


def magnetization(lattice: ndarray[int]) -> float: ...


def energy(lattice: ndarray[int], b_field: float) -> float: ...


def delta_energy(
    lattice: ndarray[int], cell_index: tuple[int], b_field: float
) -> float: ...


def mcmc_step(
    lattice: ndarray[int], temperature: float, b_field: float
) -> ndarray[int]: ...


def mcmc_full(
    lattice: ndarray[int],
    temperature: float,
    b_field: float,
    total_steps: int,
) -> ndarray[float]: ...
