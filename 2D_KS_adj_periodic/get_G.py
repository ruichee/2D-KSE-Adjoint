import numpy as np
import input_vars
from get_R import get_R
from dealiase import dealiase
from input_vars import stage, nx, ns, KX, KS, f, Lx, Ls

def get_G(t, L, print_res=False):

    global stage, nx, ns, KX, KS, f

    # unpack loop variable
    u, T = L[:-1].reshape(nx, ns), L[-1]

    # first obtain R and its fourier transform
    r = get_R(t, L)[:-1].reshape(nx, ns)
    r_f = np.fft.fft2(r)

    # unsteady term
    r_s_f = 1j * KS * r_f
    unsteady = np.real(-1/T * np.fft.ifft2(r_s_f))

    # non-linear term in fourier space
    r_x_f = 1j * KX * r_f
    r_x = np.fft.ifft2(r_x_f)
    nonlin_term = np.real(-u * r_x)

    # linear terms in fourier space
    r_xx = np.fft.ifft2(-KX**2 * r_f)
    r_xxxx = np.fft.ifft2(KX**4 * r_f)
    lin_term = np.real(r_xx + r_xxxx)

    # add all terms together in fourier space
    G1_f = np.fft.fft2(unsteady + nonlin_term + lin_term)
    G1_f = dealiase(G1_f)

    # --- MINIMAL CHANGE: GALERKIN TRUNCATION ---
    # Cut off high spatial frequencies to prevent k^8 explicit Euler explosion
    # k_cutoff = 15 is standard for nx=64 (d=32 in the paper)
    k_cutoff = 15 
    mask_k = np.abs(KX) > k_cutoff
    G1_f[mask_k] = 0.0
    # -------------------------------------------

    # set mean flow = 0, no DC component/offset
    mask = (KX==0) 
    G1_f = np.where(mask, 0, G1_f)

    # convert back to physical space
    G1 = np.real(np.fft.ifft2(G1_f))

    # T component of G
    u_f = np.fft.fft2(u)
    u_s_f = 1j * KS * u_f
    u_s = np.fft.ifft2(u_s_f)
    integrand = -r / (T**2) * u_s 
    integrand_hat = np.fft.fft2(integrand)
    dx = Lx / nx
    ds = Ls / ns
    G2 = np.real(integrand_hat[0, 0]) * dx * ds


    # print to track iteration progress, use to check for sticking points
    # might need normalization based on the grid size - divide by np.sqrt(nx*ny)
    if print_res:
        print(f"stage: {input_vars.stage}, t: {np.round(t, 5)}, \
              \t G1: {np.round(np.linalg.norm(G1), 12)}, \
              \t G2: {np.round(G2, 12)}, \
              \t |R|: {np.round(np.linalg.norm(r), 12)}, \t T: {T}")

    return np.append(G1, G2)