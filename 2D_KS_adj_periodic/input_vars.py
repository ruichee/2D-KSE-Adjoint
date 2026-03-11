from get_vars import get_vars

# define variables 
Lx, Ly = 10, 10                 # spatial domain size
nx, ny = 64, 64                 # number of spatial collocation points
Ls = 1                          # time (normalized) domain size
ns = 64                         # number of temporal collocation points
dt = 100                        # controls what interval in optimization time (tau) we receive the output 
                                # u_lst and t_lst to be (actual time step is controlled in solve_ivp)
f = 0
stage = 0

# obtain domain field (x,s), and fourier wave numbers kx
# note s = t/T, where s is in [0,1) => normalized time
X, KX, Y, KY, S, KS = get_vars(2*Lx, 2*Ly, Ls, nx, ny, ns)