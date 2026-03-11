import numpy as np
from dealiase import dealiase
from input_vars import KX, KS, f, nx, ns

def get_R(t, L, print_res=False): 

    # unpack loop variable
    u, T = L[:-1].reshape(nx, ns), L[-1]
    
    # obtain u in fourier space
    u_f = np.fft.fft2(u)                        # bring u into fourier
    u_f = dealiase(u_f)                         # dealise u

    # non-linear and unsteady term 
    u_x_f = 1j * KX * u_f
    u_s_f = 1j * KS * u_f
    u_x = np.real(np.fft.ifft2(u_x_f))
    u_s = np.real(np.fft.ifft2(u_s_f))
    nonlin_terms = - 1/T * u_s - u * u_x

    # linear terms  
    lin_terms_f =  (KX**2 - KX**4) * u_f    # n-derivative = multiply u by (ik)^n
    
    # add terms together 
    R_f = np.fft.fft2(nonlin_terms) + lin_terms_f
    R_f = dealiase(R_f)                         # dealise R

    # set mean flow = 0, no DC component/offset
    #mask = (KX==0)
    #R_f = np.where(mask, 0, R_f)               # ensures the sine wave has no constant component (kx=0 and ky=0)

    # convert back to physical space
    R = np.real(np.fft.ifft2(R_f)) + f         # obtain R(u)

    # print to track iteration progress, use to check for sticking points
    if print_res:
        print(f"time: {t}, \t ||R||: {np.linalg.norm(R)}")
    
    return np.append(R, 0)