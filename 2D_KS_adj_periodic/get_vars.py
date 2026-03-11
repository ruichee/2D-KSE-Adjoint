import numpy as np

def get_vars(Lx: float, Ly: float, Ls: float, nx: int, ny: int, ns: int) -> tuple[np.ndarray[any, float],...]:

    dx = Lx/nx                                  # define x spatial step
    dy = Ly/ny                                  # define y spatial step
    ds = Ls/ns                                  # define s temporal step
    
    x = np.linspace(0, Lx-dx, nx)               # nx = EVEN no. of collocation points, define grid
    y = np.linspace(0, Ly-dy, ny)               # ny = EVEN no. of collocation points, define grid
    s = np.linspace(0, Ls-ds, ns)
    
    kx = 2*np.pi * np.fft.fftfreq(nx, d=dx)     # fourier wave numbers (kx) for DFT in x-dir
    ky = 2*np.pi * np.fft.fftfreq(ny, d=dy)     # fourier wave numbers (ky) for DFT in y-dir
    ks = 2*np.pi * np.fft.fftfreq(ns, d=ds)
    
    KX, KS, KY = np.meshgrid(kx, ks, ky)        # meshgrid of all combinations of kx and ky waves
    X, S, Y = np.meshgrid(x, s, y)              # meshgrid of all combinations of x and y values
    
    return X, KX, Y, KY, S, KS                  # NOTE: L-dx ensure no cutting into next period