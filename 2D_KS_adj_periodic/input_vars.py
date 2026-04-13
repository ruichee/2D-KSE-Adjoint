from get_vars import get_vars
import numpy as np

# define variables 
Lx = 32*np.pi                 # spatial domain size
nx = 64                 # number of spatial collocation points
Ls = 1                          # time (normalized) domain size
ns = 64                         # number of temporal collocation points
dt = 100                        # controls what interval in optimization time (tau) we receive the output 
                                # u_lst and t_lst to be (actual time step is controlled in solve_ivp)
k0 = 4
omega = 2*np.pi/Lx
epsilon = 0.057
f = epsilon * np.sin(k0 * omega * X)
stage = 0

# obtain domain field (x,s), and fourier wave numbers kx
# note s = t/T, where s is in [0,1) => normalized time
X, KX, S, KS = get_vars(Lx, Ls, nx, ns)