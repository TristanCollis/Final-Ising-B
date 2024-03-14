import numpy as np

from custom_types import ndarray
from helpers.helpers import compareH

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