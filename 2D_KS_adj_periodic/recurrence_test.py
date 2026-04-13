import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import RectBivariateSpline


# --- 1. Parameters ---
Lx = 39.0
nx = 64
dx = Lx / nx
x = np.linspace(0, Lx - dx, nx)
k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)

# ---> CHANGED: Lower dt for the tighter dx grid
dt = 0.05           
t_spinup = 100.0    
t_prod = 150.0      
save_every = 5      # Increased since dt is smaller, to keep array sizes manageable

# --- 2. ETDRK4 Setup (Kassam & Trefethen Contour Integral) ---
L = k**2 - k**4
E = np.exp(L * dt)
E2 = np.exp(L * dt / 2.0)

M = 32 
r = 15.0 
theta = np.linspace(0, 2*np.pi, M, endpoint=False)
z = L * dt + r * np.exp(1j * theta)[:, np.newaxis] 

# ---> CHANGED: Stripped imaginary numerical noise
Q  = np.real(dt * np.mean((np.exp(z/2) - 1) / z, axis=0))
f1 = np.real(dt * np.mean((-4 - z + np.exp(z)*(4 - 3*z + z**2)) / z**3, axis=0))
f2 = np.real(dt * np.mean((2 + z + np.exp(z)*(-2 + z)) / z**3, axis=0))
f3 = np.real(dt * np.mean((-4 - 3*z - z**2 + np.exp(z)*(4 - z)) / z**3, axis=0))

# ... [Section 3 remains the same] ...
# --- 3. Nonlinear Operator ---
k_max = (2/3) * np.max(np.abs(k)) # 2/3 dealiasing rule

def N_op(v_hat):
    """Computes the nonlinear term -u*u_x in Fourier space"""
    u = np.real(np.fft.ifft(v_hat))
    nonlin_hat = -0.5 * 1j * k * np.fft.fft(u**2)
    nonlin_hat[np.abs(k) > k_max] = 0.0 # Dealiase
    return nonlin_hat

# --- 4. Time Integration Loop ---
# ---> CHANGED: Made the initial wave perfectly periodic over Lx
u0 = (0.5 * np.cos(2 * np.pi * x / Lx) + 
      1.0 * np.sin(4 * np.pi * x / Lx) + 
      0.5 * np.sin(6 * np.pi * x / Lx)) + 0.5 * np.random.randn(nx)
v = np.fft.fft(u0)
v[0] = 0.0

print("Spinning up (ETDRK4)...")
for _ in range(int(t_spinup / dt)):
    Nv = N_op(v)
    a = E2 * v + Q * Nv
    Na = N_op(a)
    b = E2 * v + Q * Na
    Nb = N_op(b)
    c = E2 * a + Q * (2*Nb - Nv)
    Nc = N_op(c)
    
    v = E * v + Nv * f1 + 2*(Na + Nb)*f2 + Nc * f3
    v[0] = 0.0 # Constrain mean flow

print("Running production simulation...")
u_history = []
t_history = []

for step in range(int(t_prod / dt)):
    Nv = N_op(v)
    a = E2 * v + Q * Nv
    Na = N_op(a)
    b = E2 * v + Q * Na
    Nb = N_op(b)
    c = E2 * a + Q * (2*Nb - Nv)
    Nc = N_op(c)
    
    v = E * v + Nv * f1 + 2*(Na + Nb)*f2 + Nc * f3
    v[0] = 0.0 # Constrain mean flow
    
    if step % save_every == 0:
        u_history.append(np.real(np.fft.ifft(v)))
        t_history.append(step * dt)

u_history = np.array(u_history)
t_history = np.array(t_history)

# --- 5. Recurrence Search ---
print("Computing recurrence matrix...")
distances = cdist(u_history, u_history, metric='euclidean')

# Avoid trivial self-matches (require period to be at least T=5.0)
time_gap = int(5.0 / (dt * save_every))
mask = np.triu(np.ones_like(distances), k=time_gap)
distances = np.where(mask > 0, distances, np.inf)

min_idx = np.unravel_index(np.argmin(distances), distances.shape)
t_start, t_end = t_history[min_idx[0]], t_history[min_idx[1]]
T_guess = t_end - t_start

print(f"\nBest recurrence found!")
print(f"Start Time: {t_start:.2f}, End Time: {t_end:.2f}")
print(f"Guessed Period T0: {T_guess:.3f}")
print(f"L2 Distance: {distances[min_idx]:.5f}")

# --- 6. Extract & Interpolate for Adjoint Main Loop ---
orbit_window = u_history[min_idx[0]:min_idx[1], :]
s_original = np.linspace(0, 1, orbit_window.shape[0])
s_target = np.linspace(0, 1, 64) # ns = 64 from your input_vars

# RectBivariateSpline is highly stable for 2D regular grids
spline = RectBivariateSpline(s_original, x, orbit_window)
u_initial_guess = spline(s_target, x) # Shape is now exactly (64, 64)

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(distances, origin='lower', extent=[0, t_prod, 0, t_prod], cmap='viridis_r')
plt.colorbar(label='L2 Distance')
plt.plot(t_history[min_idx[1]], t_history[min_idx[0]], 'ro')
plt.title(f"Recurrence Plot (Best T={T_guess:.2f})")
plt.xlabel("Time t")
plt.ylabel("Reference Time t_ref")

plt.subplot(1, 2, 2)
plt.imshow(u_initial_guess, aspect='auto', origin='lower', extent=[0, Lx, 0, 1], cmap='RdBu')
plt.title(f"Interpolated Guessed Orbit ($n_x=64, n_s=64$)")
plt.xlabel("Space x")
plt.ylabel("Normalized Time s=t/T")
plt.colorbar(label='u(x,s)')
plt.tight_layout()
plt.show()

# save entire u_final array data to output_u.csv file
np.savetxt(r'2D_KS_adj_periodic copy\store_orbit.dat', u_initial_guess, delimiter=' ', fmt='%.18e')
# Now you have u_initial_guess (shape: 64x64) and T_guess!