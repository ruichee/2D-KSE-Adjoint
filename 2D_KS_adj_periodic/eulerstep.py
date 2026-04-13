import numpy as np
from get_G import get_G
from input_vars import KX, KS, nx, ns

def euler_descent(L_initial: np.ndarray, t_end: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    
    num_steps = max(1, int(t_end / dt))
    
    L_current = np.copy(L_initial)
    L_lst = []
    t_lst = []
    
    tau = 0.0
    
    for i in range(num_steps):
        print_flag = (i % 1000 == 0)
        store_flag = (i % 1000 == 0)
        
        # 1. Get the total explicit gradient from your working get_G
        G_current = get_G(tau, L_current, print_res=print_flag) 
        
        G1_flat = G_current[:-1]
        G2 = G_current[-1]
        T_current = L_current[-1]
        
        # --- IMEX STEP FOR THE FIELD ---
        # 1. Bring the field and gradient into Fourier space
        u_f = np.fft.fft2(L_current[:-1].reshape(ns, nx))
        G1_f = np.fft.fft2(G1_flat.reshape(ns, nx))
        
        # 2. Build the Linear Operator (Equation 4.9 from the paper)
        L_hat = -(KS/T_current)**2 - KX**8 + 2*KX**6 - KX**4
        
        # 3. Apply the implicit dampening to the explicit step
        u_new_f = u_f + (dt * G1_f) / (1.0 - dt * L_hat)
        
        # 4. Return to physical space
        u_new_flat = np.real(np.fft.ifft2(u_new_f)).flatten()
        
        # --- EXPLICIT STEP FOR T ---
        # The paper uses purely explicit Euler for T, explicit preconditioning by *10
        T_new = T_current + (dt*0) * G2
        
        # Pack and step
        L_current = np.append(u_new_flat, T_new)
        tau += dt
        
        if store_flag:
            L_lst.append(L_current)
            t_lst.append(tau)

            '''
            u_k = (np.fft.fft(u_new_flat.reshape(64,64), axis=1))

            P1, P2 = u_k[:, 1].imag, u_k[:, 2].imag
            from plotting import Plotting
            Plotting.plot_init_orbit(P1, P2)'''
        
    return np.array(L_lst), np.array(t_lst)