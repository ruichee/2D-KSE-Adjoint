import numpy as np

def get_vars(Lx: float, Ls: float, nx: int, ns: int) -> tuple[np.ndarray[any, float],...]:

    dx = Lx/nx                                  # define x spatial step
    ds = Ls/ns                                  # define s temporal step
    
    x = np.linspace(0, Lx-dx, nx)               # nx = EVEN no. of collocation points, define grid
    s = np.linspace(0, Ls-ds, ns)
    
    kx = 2*np.pi * np.fft.fftfreq(nx, d=dx)     # fourier wave numbers (kx) for DFT in x-dir
    ks = 2*np.pi * np.fft.fftfreq(ns, d=ds)
    
    KX, KS = np.meshgrid(kx, ks)          # meshgrid of all combinations of kx and ky waves
    X, S = np.meshgrid(x, s)              # meshgrid of all combinations of x and y values
    
    return X, KX, S, KS                   # NOTE: L-dx ensure no cutting into next period