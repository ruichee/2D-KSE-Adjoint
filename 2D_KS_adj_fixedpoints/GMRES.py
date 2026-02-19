import numpy as np
from get_R import get_R
from input_vars import nx, ny
from scipy.optimize import newton_krylov

def get_pinned_residual(u_1d, nx=64, ny=64):
    """
    Wraps the KSE physics operator and applies Symmetry Reduction
    to destroy the zero-eigenvalues caused by periodic sliding.
    """
    u_2d = u_1d.reshape((nx, ny))
    
    # 1. Get the pure, strictly real physical residual
    R_2d = np.real(get_R(0, u_2d))
    
    # 2. Apply the Phase Pins (Central difference across periodic boundaries)
    # Pin the X-sliding symmetry (force dx = 0 at origin)
    pin_x = u_2d[1, 0] - u_2d[-1, 0] 
    
    # Pin the Y-sliding symmetry (force dy = 0 at origin)
    pin_y = u_2d[0, 1] - u_2d[0, -1] 
    
    # 3. Overwrite the first two arbitrary equations with our pins
    R_2d[0, 0] = pin_x
    R_2d[0, 1] = pin_y
    
    return R_2d.flatten()


def gmres_step(u0):
    print("\n=== Executing Pinned Newton-Krylov Step ===")
    
    global nx, ny
    u_flat_start = u0.flatten()
    
    try:
        # We pass our pinned wrapper to the solver
        u_perfect_flat = newton_krylov(
            lambda u: get_pinned_residual(u, nx, ny), 
            u_flat_start, 
            method='lgmres',
            f_tol=1e-10,       # The absolute bottom of the cliff
            line_search=None,  # NO BRAKES: Take the full quadratic step
            inner_maxiter=50,  # Let GMRES fully map the steep bowl
            verbose=False,     # Watch the cliff-drop happen in real-time
            iter=1000
        )
        
        u_final = u_perfect_flat.reshape((nx, ny))
        print("\nSUCCESS: Solver hit the 1e-10 noise floor!")
        
    except Exception as e:
        print(f"\nSolver stalled or crashed: {e}")
        # If it fails, we just keep the best Adjoint state
        u_final = u0 

    if np.linalg.norm(get_R(0, u_final)) > np.linalg.norm(get_R(0, u0)):
        return u0
    
    return u_final