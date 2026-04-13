import numpy as np
import matplotlib.pyplot as plt
from get_R import get_R
from get_G import get_G
from input_vars import X, S
from residual import compute_residuals

class Plotting:

    def plot_init_orbit(P1, P2, P3, P4):
            
        # setup axis and figure
        fig, (p1p2, p1p3, p1p4) = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Force the loop to close by appending the first point to the end
        P1_closed = np.append(P1, P1[0])
        P2_closed = np.append(P2, P2[0])
        P3_closed = np.append(P3, P3[0])
        P4_closed = np.append(P4, P4[0])

        # 2. Setup a publication-quality figure
        #plt.figure(figsize=(6, 6)) # A square aspect ratio is standard for phase portraits
        plt.rc('text', usetex=False) # Use True if you have LaTeX installed on your machine
        plt.rc('font', family='serif', size=14)

        # 3. Plot without markers (just a clean, smooth solid line)
        # Use a slight transparency (alpha) if comparing multiple orbits
        p1p2.plot(P1_closed, P2_closed, color='midnightblue', linewidth=1.5, linestyle='-')
        p1p3.plot(P1_closed, P3_closed, color='midnightblue', linewidth=1.5, linestyle='-')
        p1p4.plot(P1_closed, P4_closed, color='midnightblue', linewidth=1.5, linestyle='-')

        # 4. Clean up the axes
        p1p2.set_xlabel('$P_1$')
        p1p2.set_ylabel('$P_2$')
        p1p3.set_xlabel('$P_1$')
        p1p3.set_ylabel('$P_3$')
        p1p4.set_xlabel('$P_1$')
        p1p4.set_ylabel('$P_4$')

        # Keep the grid subtle so it doesn't distract from the orbit's topology
        p1p2.grid(True, linestyle='--', alpha=0.5)
        p1p3.grid(True, linestyle='--', alpha=0.5)
        p1p4.grid(True, linestyle='--', alpha=0.5)

        # Force the physical scale of x and y to be equal so the shape isn't warped
        p1p2.set_aspect('equal', adjustable='datalim') 
        p1p3.set_aspect('equal', adjustable='datalim') 
        p1p4.set_aspect('equal', adjustable='datalim') 

        plt.tight_layout()
        plt.show()

    def plot_from_data(path):

        u = np.loadtxt(path)
        plt.contourf(X, Y, u, figsize=(7, 7))
        plt.show()

    ###############################################################################################

    def plot_initial(u0: np.ndarray[tuple[int, int], float]) -> None:

        # setup axis and figure
        fig, (u0_ax, R0_ax, G0_ax) = plt.subplots(1, 3, figsize=(15, 4))

        # obtain R and G fields
        R = get_R(0, u0)
        G = get_G(0, u0)

        # plot contours 
        u0_contlines = u0_ax.contour(X, Y, u0, colors="black", linewidths=1, linestyles="solid")
        u0_cont = u0_ax.contourf(X, Y, u0)
        R0_contlines = R0_ax.contour(X, Y, R, colors="black", linewidths=1, linestyles="solid")
        R0_cont = R0_ax.contourf(X, Y, R)
        G0_contlines = G0_ax.contour(X, Y, G, colors="black", linewidths=1, linestyles="solid")
        G0_cont = G0_ax.contourf(X, Y, G)

        # set titles and add colorbars
        u0_ax.set_title("Initial u")
        R0_ax.set_title("Initial R")
        G0_ax.set_title("Initial G")
        fig.colorbar(u0_cont)
        fig.colorbar(R0_cont)
        fig.colorbar(G0_cont)

        # display plots for u, G, R at initialization
        plt.show()

    ###############################################################################################

    def plot_final(u_lst: np.ndarray[tuple[int, int], float], t_lst) -> None:

        fig, (u_val, G, R) = plt.subplots(1, 3, figsize=(16, 4))
        
        # extract final u field
        u_final = u_lst[-1]
        np.nan_to_num(u_final, nan=0)

        # plot u field
        u_cont = u_val.contourf(X, Y, u_final)
        u_val.contour(X, Y, u_final, colors="black", linewidths=1, linestyles="solid")
        u_val.set_xlabel('x')
        u_val.set_ylabel('y')
        fig.colorbar(u_cont)

        # plot residuals
        t_lst_trunc, R_lst_trunc, G_lst_trunc = compute_residuals(t_lst, u_lst)
        G.plot(t_lst_trunc, G_lst_trunc)
        G.semilogy()
        G.set_xlabel('τ')
        G.set_title('RMS of G(u)')
        G.set_xlim(0, t_lst_trunc[-1])
        G.grid()

        R.plot(t_lst_trunc, R_lst_trunc)
        R.semilogy()
        R.set_xlabel('τ')
        R.set_title('L2-norm ||R(u)||')
        R.set_xlim(0, t_lst_trunc[-1])
        R.grid()

        plt.show()