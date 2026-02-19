import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from get_R import get_R
from input_vars import nx, ny

def perform_one_NGh_step(u_current, epsilon=1e-7, damping_mu=0.1):
    """
    Performs exactly ONE outer Newton-GMRES step with a Hookstep-style damping factor.
    """
    print("\n--- Starting 1-Step NGh ---")
    
    # 1. Calculate the current physical residual
    u_2d = u_current.reshape((nx, ny))
    R_current_2d = np.real(get_R(0, u_2d)) # Ensure pure real!
    R_current = R_current_2d.flatten()
    
    residual_norm = np.linalg.norm(R_current)
    print(f"Initial Residual ||R||: {residual_norm:.4e}")
    
    # 2. Define the Matrix-Free Jacobian Operator
    # This is the "Jedi trick" that GMRES will use to probe the physics
    def jacobian_vector_product(v):
        # Perturb the state in the direction of v
        u_perturbed_2d = (u_current + epsilon * v).reshape((nx, ny))
        R_perturbed_2d = np.real(get_R(0, u_perturbed_2d))
        R_perturbed = R_perturbed_2d.flatten()
        
        # Calculate directional derivative: J*v = (R(u + eps*v) - R(u)) / eps
        Jv = (R_perturbed - R_current) / epsilon
        
        # THE HOOKSTEP FIX: Add damping to shift the zero-eigenvalues
        # This prevents GMRES from blowing up in the flat symmetry ravine
        return Jv + (damping_mu * v) 

    # Create the SciPy LinearOperator wrapper
    J_op = LinearOperator((nx*nx, ny*ny), matvec=jacobian_vector_product)
    
    # 3. Fire the Inner GMRES Engine to find the step (Delta u)
    # We allow GMRES to take up to 50 inner iterations to build the Krylov subspace
    print("Inner GMRES calculating trust-region step...")
    delta_u, exit_code = gmres(J_op, -R_current, rtol=1e-10, maxiter=100)
    
    if exit_code != 0:
        print("Warning: GMRES inner loop did not perfectly converge.")
        
    # 4. Take the 1 Single Outer Step
    u_new = u_current + delta_u
    
    # 5. Evaluate the result
    u_new_2d = u_new.reshape((nx, ny))
    R_new_2d = np.real(get_R(0, u_new_2d))
    R_new = R_new_2d.flatten()
    new_norm = np.linalg.norm(R_new)
    
    print(f"Final Residual ||R||: {new_norm:.4e}")
    print("--- 1-Step NGh Complete ---\n")
    
    return u_new, new_norm