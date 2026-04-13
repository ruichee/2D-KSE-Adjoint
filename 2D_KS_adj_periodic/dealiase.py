import numpy as np
from input_vars import KX, KS

def dealiase(ff) -> np.ndarray:

    kx_abs = np.absolute(KX)
    ks_abs = np.absolute(KS)

    kx_max = 2/3 * np.max(kx_abs)                       # maximum frequency that we will keep
    ks_max = 2/3 * np.max(ks_abs)                       # maximum frequency that we will keep

    ff_filterx = np.where(np.abs(KX) < kx_max, ff, 0)           # all higher frequencies in x are set to 0
    ff_filterxs = np.where(np.abs(KS) < ks_max, ff_filterx, 0)  # all higher frequencies in y are set to 0
    
    return ff_filterxs
