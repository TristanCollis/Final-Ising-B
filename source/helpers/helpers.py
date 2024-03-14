import numpy as np
from numba import njit

@njit
def compareH(h_val, h1, error):
    if abs(h1 - h_val) < error:
        return -1
    else: 
        return 1
