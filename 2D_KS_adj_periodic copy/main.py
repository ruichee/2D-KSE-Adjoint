import numpy as np
import input_vars
from input_vars import X, S, Lx, Ls, dt, KX, KS
from adj_descent import adj_descent
from plotting import Plotting
from get_R import get_R


def main(u0: np.ndarray[tuple[int, int], float], T0,
         stages: tuple[tuple[int, float]], dt) -> None:

    # plot initial fields
    #Plotting.plot_initial(u0[0])

    # plot loop via fourier coefficients
    import dealiase
    u_k = (np.fft.fft(u0, axis=1))

    P1, P2 = u_k[:, 1].imag, u_k[:, 2].imag
    print(P1)

    Plotting.plot_init_orbit(P1, P2)

    L_prev = np.append(u0.flatten(), T0)
    L_lst = np.array([L_prev])
    t_lst = np.array([0])

    from eulerstep import euler_descent

    for t_end in stages:
        input_vars.stage += 1
        if t_end == 0:
             continue
        
        # --- CHANGED LINE HERE ---
        # We drop the 'tol' arguments because explicit Euler doesn't use adaptive tolerances
        L_lst1, t_lst1 = euler_descent(L_prev, t_end=t_end, dt=0.15)
        # -------------------------
        
        L_prev = L_lst1[-1]

        t_lst1_shifted = t_lst1 + t_lst[-1] 
        L_lst = np.concatenate((L_lst, L_lst1[1:]), axis=0)
        t_lst = np.concatenate((t_lst, t_lst1_shifted[1:]), axis=0)


    '''for t_end, tol in stages:
        input_vars.stage += 1
        if t_end == 0:
             continue
        L_lst1, t_lst1 = adj_descent(L_prev, tol, tol, t_end=t_end, dt=dt)
        L_prev = L_lst1[-1]

        t_lst1_shifted = t_lst1 + t_lst[-1] 
        L_lst = np.concatenate((L_lst, L_lst1[1:]), axis=0)
        t_lst = np.concatenate((t_lst, t_lst1_shifted[1:]), axis=0)
        '''

    '''
        u_new = gmres_step(u_lst[-1])
        # Append the final teleportation step for plotting
        ngh_time_jump = t_lst[-1] + 1000 
        u_lst = np.concatenate((u_lst, [u_new]), axis=0)
        t_lst = np.concatenate((t_lst, [ngh_time_jump]), axis=0)
        '''
    
    # extract final u field
    from input_vars import nx, ns
    L_final = L_lst[-1]
    u_final, T_final = L_final[:-1].reshape(nx, ns), L_final[-1]

    # remove DC offset
    '''u_k = np.fft.fft2(u_final)
    mask = (KX==0) 
    u_k = np.where(mask, 0, u_k) 
    u_final = np.real(np.fft.ifft2(u_k))'''

    u_k_final = np.fft.fft(u_final, axis=1)

    P1, P2 = u_k_final[:, 1].imag, u_k_final[:, 2].imag
    Plotting.plot_init_orbit(P1, P2)

    import matplotlib.pyplot as plt
    T_lst = L_lst.T[-1]
    plt.plot(T_lst)
    plt.show()

    residual_lst = [np.linalg.norm(get_R(0, L)) for L in L_lst]
    
    plt.plot(residual_lst)
    plt.semilogy()
    plt.show()

    '''print(np.linalg.norm(get_R(0, u_final)))
    u_lst[-1] = u_final

    # check fourier values
    func = lambda x,y: np.round(np.abs(u_k[x,y]), 2)
    print("\nFourier Coefficients")
    print("\t", func(1, 0), func(1, 1), func(0, 1), "\n")
    print(f"\t e(2,0) e(2,1) e(3,0) e(3,1) e(0,2) e(1,2) e(2,2)")
    print(f"\t {func(0, 2)} {func(1, 2)} {func(0, 3)} {func(1, 3)} {func(2, 0)} {func(2, 1)} {func(2, 2)} \n")
    print(f"\t e(0,3): {func(3, 0)}, e(1,3): {func(3, 1)}")
    print()

    print("\nFourier Coefficients (mirrored direction)")
    print("\t", func(-1, 0), func(-1, 1), func(0, 1), "\n")
    print(f"\t e(2,0) e(2,1) e(3,0) e(3,1) e(0,2) e(1,2) e(2,2)")
    print(f"\t {func(0, 2)} {func(-1, 2)} {func(0, 3)} {func(-1, 3)} {func(-2, 0)} {func(-2, 1)} {func(-2, 2)} \n")
    print(f"\t e(0,3): {func(-3, 0)}, e(1,3): {func(-3, 1)}")
    print()

    # plot final results 
    #Plotting.plot_final(u_lst, t_lst)'''

    # save entire u_final array data to output_u.csv file
    np.savetxt(r'2D_KS_adj_periodic copy\store_orbit.dat', L_final, delimiter=' ', fmt='%.18e')


if __name__ == "__main__":
    #from get_R import get_R
    #print(np.linalg.norm(get_R(0, np.loadtxt(r"2D_KS_adj\fixed_points\output_u.dat", delimiter=" "))))

    # define initial conditions of field variable u
    kx = 2*np.pi * (X / Lx)
    ks = 2*np.pi * (S / Ls)
    u0 = 1.5 * np.cos(ks) * np.sin(kx) + \
            0.8 * np.sin(2 * ks + 2 * kx + 0.5)

    #L0 = np.loadtxt(r"2D_KS_adj_periodic copy\store_orbit.dat", delimiter=' ')
    #u0 = L0[:-1].reshape(ns, nx)
    #T0 = L0[-1]

    # define iteration time variables
    '''T1, tol1 = 100, 1e-6
    T2, tol2 = 100, 1e-8
    T3, tol3 = 1000, 1e-12
    T4, tol4 = 2000, 1e-12
    T5, tol5 = 10000, 1e-14
    T6, tol6 = 100000, 1e-16
    stages = ((T1, tol1), (T2, tol2), (T3, tol3), (T4, tol4), (T5, tol5), (T6, tol6), )'''

    stages = (1e4,)*2

    main(u0, 25, stages, dt)