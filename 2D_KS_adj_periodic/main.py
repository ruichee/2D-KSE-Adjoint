import numpy as np
import input_vars
from input_vars import X, Y, S, Lx, Ly, Ls, dt, KX, KY, KS
from adj_descent import adj_descent
from plotting import Plotting
from get_R import get_R


def main(u0: np.ndarray[tuple[int, int], float], 
         stages: tuple[tuple[int, float]], dt) -> None:

    # plot initial fields
    #Plotting.plot_initial(u0[0])


    u_prev = u0
    u_lst = np.array([u0])
    t_lst = np.array([0])

    for T, tol in stages:
        input_vars.stage += 1
        if T == 0:
             continue
        u_lst1, t_lst1 = adj_descent(u_prev, tol, tol, T=T, dt=dt)
        u_prev = u_lst1[-1]

        t_lst1_shifted = t_lst1 + t_lst[-1] 
        u_lst = np.concatenate((u_lst, u_lst1[1:]), axis=0)
        t_lst = np.concatenate((t_lst, t_lst1_shifted[1:]), axis=0)

        '''
        u_new = gmres_step(u_lst[-1])
        # Append the final teleportation step for plotting
        ngh_time_jump = t_lst[-1] + 1000 
        u_lst = np.concatenate((u_lst, [u_new]), axis=0)
        t_lst = np.concatenate((t_lst, [ngh_time_jump]), axis=0)
        '''
    
    # extract final u field
    u_final = u_lst[-1]

    # remove DC offset
    u_k = np.fft.fft2(u_final)
    mask = (KX==0) * (KY==0)
    u_k = np.where(mask, 0, u_k) 
    u_final = np.real(np.fft.ifft2(u_k))

    print(np.linalg.norm(get_R(0, u_final)))
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
    #Plotting.plot_final(u_lst, t_lst)

    # save entire u_final array data to output_u.csv file
    np.savetxt(r'2D_KS_adj_fixedpoints\fixed_points\output_u.dat', u_final, delimiter=' ', fmt='%.18e')


if __name__ == "__main__":
    #from get_R import get_R
    #print(np.linalg.norm(get_R(0, np.loadtxt(r"2D_KS_adj\fixed_points\output_u.dat", delimiter=" "))))

    # define initial conditions of field variable u
    kx = np.pi * (X / Lx)
    ky = np.pi * (Y / Ly)
    ks = 2*np.pi * (S / Ls)
    u0 = np.sin(kx + ky) * np.cos(ks)

    #u0 = np.loadtxt(r"2D_KS_adj_fixedpoints\fixed_points\output_u.dat", delimiter=' ')

    # define iteration time variables
    T1, tol1 = 10, 1e-8
    T2, tol2 = 10, 1e-10
    T3, tol3 = 100, 1e-12
    T4, tol4 = 2e4, 1e-14
    T5, tol5 = 3e5, 1e-16
    T6, tol6 = 1e6, 1e-17
    stages = ((T1, tol1), (T2, tol2), (T3, tol3), (T4, tol4), (T5, tol5), )

    main(u0, stages, dt)