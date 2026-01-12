import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def get_vars(domain_size, num_colloc_pts):

    L, n = domain_size, num_colloc_pts
    dx = L/n                                    # define spatial step
    x = np.linspace(0, L-dx, n)                 # n = EVEN no. of collocation points, define grid
    kx = 2*np.pi * np.fft.fftfreq(n, d=L/n)     # fourier wave numbers (k) for DFT
    return (x, kx)                              # NOTE: L-dx ensure no cutting into next period


def dealiase(ff, kx):

    k = np.absolute(kx)
    k_max = 1/3 * np.max(k)                     # maximum frequency that we will keep
    ff_filtered = np.where(k < k_max, ff, 0)    # all higher frequencies are set to 0
    return ff_filtered


def get_R(u, f, kx): # TRY IMPLEMENTING VIA FINITE DIFFERENCE, VALIDATE IF FEASIBLE

    # non-linear term -u∂ₓu in fourier space
    #u_sq = u**2                                 # obtain u^2, since -u∂ₓu = -0.5*∂ₓ(u^2)
    #u_sqf = np.fft.fft(u_sq)                    # bring u^2 into fourier space
    #u_sqf_x = 1j * kx * u_sqf                   # multiply by ik to each u_k (differentiate in fourier)
    #u_sq_x = np.fft.ifft(u_sqf_x)               # convert back to physical space, we get ∂ₓ(u^2)
    #udu = -0.5 * u_sq_x                         # multiply by minus half to obtain -u∂ₓu

    # alternatively, find -u∂ₓu more directly
    '''u_f = dealiase(np.fft.fft(u), kx)
    du = np.fft.ifft(1j * kx * u_f)
    udu = -u * du'''

    # obtain u in fourier space
    u_f = np.fft.fft(u)                         # bring u into fourier
    u_f = dealiase(u_f, kx)                     # dealise u

    # non-linear term 
    u_x_f = 1j * kx * u_f                       # ∂ₓu in fourier, differentiate via multiply ik
    u_x = np.fft.ifft(u_x_f)                    # bring back to physical space
    u_x_sq = -0.5 * u_x * u_x                   # get -1/2(∂ₓu)^2

    # add linear terms -∂ₓₓu-∂ₓₓₓₓu in fourier space 
    u_x_sq_f = np.fft.fft(u_x_sq)               # bring u∂ₓu back to fourier
    R_f = u_x_sq_f + (kx**2 - kx**4)*u_f        # add linear terms, n-derivative = multiply u by (ik)^n
    R_f = dealiase(R_f, kx)                     # dealise R
    
    # set mean flow = 0, no DC component/offset
    R_f = np.where(kx == 0, 0, R_f)             # ensures the sine wave has no constant component (k=0)

    # convert back to physical space
    R = np.real(np.fft.ifft(R_f)) + f           # obtain R(u) = -u∂ₓu - ∂ₓₓu - ∂ₓₓₓₓu + f
    
    return R


def get_G(u, f, kx):

    # first obtain R
    R = get_R(u, f, kx)

    # non-linear term -∂ₓ(R∂ₓu) in fourier space
    u_f = np.fft.fft(u)
    u_f = dealiase(u_f)


    # add linear terms -∂ₓₓR-∂ₓₓₓₓR in fourier space

    lin_terms = (kx**2 - kx**4)*R


    pass


def adj_descent(u, f, dt, n_iter, tol):

    u_lst = [u]

    for _ in tqdm(range(n_iter)):

        un = u.copy()
        G = get_G(u, f)

        u = un + dt*G # can implement rk45 later on
        u_lst.append(u)
        
        err: float # implement error calc
        if err < tol: 
            break

    return u, u_lst
        

def ngh_descent(u, f, dt, n_iter, tol):

    u_lst = [u]
    
    for _ in tqdm(range(n_iter)):

        un = u.copy()
        R = get_R(u, f)
        
        u = un + dt*R # can implement rk45 later on
        u_lst.append(u)

        err: float # implement error calc
        if err < tol:
            break

    return u, u_lst


def plot_data(u_lst):
    
    def update():
        pass
    
    ax, fig = plt.figure()

    # animate convergence
    ani = FuncAnimation(fig=fig, frames=update)
    plt.show()



def main(u0, L, n, f, dt, n_iter_adj, n_iter_ngh, tol_adj, tol_ngh):

    u_lst = [u0]

    u, u_lst1 = adj_descent(u0, f, dt, n_iter_adj, tol_adj)
    u_lst += u_lst1
    
    # check if ngh descent is required here

    u, u_lst2 = ngh_descent(u, f, dt, n_iter_ngh, tol_ngh)
    u_lst += u_lst2

    plot_data(u_lst)


# define variables 
L = 22      # domain size
n = 128     # number of collocation points

# obtain domain field (x), and fourier wave numbers kx
x, kx = get_vars(domain_size=L, num_colloc_pts=n)

# define initial conditions of field variable u
m = 2
u0 = 2*np.sin(m*2*np.pi*x/L)

u = get_R(u0, 0, kx)

U = -(np.fft.ifft(np.fft.fft(u) * 1j * kx))


plt.plot(u)
plt.plot(U)
plt.show()