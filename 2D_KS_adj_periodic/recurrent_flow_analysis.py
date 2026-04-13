import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# STRICT 1D GRID DEFINITION
# ==========================================
nx = 64
L = 24

# Physical Grid (1D array from 0 to L)
# endpoint=False ensures we don't double-count the periodic boundary
X = np.linspace(0, L, nx, endpoint=False)

# Spectral Grid (1D array of wavenumbers)
# np.fft.fftfreq gives standard frequencies, multiply by 2*pi to get angular frequencies
dx = L / nx
KX = np.fft.fftfreq(nx, d=dx) * 2 * np.pi

def compute_N(u_hat):
    """
    Computes the nonlinear term strictly in 1D, bypassing the external dealiase file.
    Relies on the global 'dealias_mask' and 'KX' defined in the main script.
    """
    # Apply local 1D mask instead of calling the external function
    u_f = u_hat * dealias_mask 
    u = np.real(np.fft.ifft(u_f))
    
    u_x_f = 1j * KX * u_f
    u_x = np.real(np.fft.ifft(u_x_f))
    
    # Nonlinear term u * u_x
    u_sq_terms = u * u_x
    
    # Back to Fourier and dealias again
    N_hat = np.fft.fft(u_sq_terms)
    N_hat = N_hat * dealias_mask
    
    # Zero mean-flow (mask out the DC component)
    N_hat[KX == 0] = 0.0 + 0.0j
    
    return N_hat

def setup_etdrk4(L, dt):
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    
    M = 32 
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[..., np.newaxis] + r
    
    Q  = dt * np.real(np.mean((np.exp(LR/2) - 1) / LR, axis=-1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=-1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1))
    f3 = dt * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1))
    
    return E, E2, Q, f1, f2, f3

def etdrk4_step(v, E, E2, Q, f1, f2, f3):
    Nv = compute_N(v)
    a = E2 * v + Q * Nv
    
    Na = compute_N(a)
    b = E2 * v + Q * Na
    
    Nb = compute_N(b)
    c = E2 * a + Q * (2 * Nb - Nv)
    
    Nc = compute_N(c)
    v_new = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    
    # 1D FIX: Enforce zero mean-flow
    v_new[0] = 0.0 + 0.0j
    return v_new

# ==========================================
# SETUP AND INITIALIZATION
# ==========================================
dt = 0.005      
T_end = 250.0  # Extended slightly so you can see a full cycle 
num_steps = int(T_end / dt)
kx = 2 * np.pi * X / L

# Initial condition
u0 = np.sin(kx) + np.cos(2*kx)

q2 = KX**2 
L_op = q2 - q2**2

kmax_x = np.max(np.absolute(kx))
dealias_mask = (np.abs(KX) < (2.0/3.0)*kmax_x) 

u_hat = np.fft.fft(u0) * dealias_mask
u_hat[0] = 0.0 + 0.0j # 1D FIX

print("Initializing ETDRK4 Coefficients...")
E, E2, Q, f1, f2, f3 = setup_etdrk4(L_op, dt)

time_record = []
norm_record = []
u_lst = []

# Capture the reference state for the difference calculation
u_ref = np.real(np.fft.ifft(u_hat))

print(f"Running DNS for {num_steps} steps...")
for step in range(num_steps):
    u_hat = etdrk4_step(u_hat, E, E2, Q, f1, f2, f3)
    
    if step % 10 == 0:
        t = step * dt
        # 1D FIX: Use ifft, and explicitly grab the real part
        u_physical = np.real(np.fft.ifft(u_hat))
        
        # Calculate the mathematical difference from the reference state
        dist = np.linalg.norm(u_physical - u_ref)
        
        time_record.append(t)
        norm_record.append(dist)
        u_lst.append(u_physical)

# Convert physical history to a 2D array for the contour plot
u_matrix = np.array(u_lst)

# ==========================================
# PLOTTING
# ==========================================
# Crush any dummy dimensions from the imported X variable
u_matrix = np.squeeze(u_lst)  # Now shape is (num_steps, nx)
X_flat = np.squeeze(X)        # Now shape is (nx,)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1.5]})

# Plot 1: Recurrence Distance
ax1.plot(time_record, norm_record, '-k', linewidth=2)
ax1.set_xlim(0, T_end)
ax1.set_ylabel(r'$||u(t) - u(0)||_2$', fontsize=12)
ax1.set_title('Recurrent Flow Distance', fontsize=14)
ax1.grid(True, alpha=0.3)

# Plot 2: Spatiotemporal Contour
# Create a meshgrid using the flattened X
T_grid, X_grid = np.meshgrid(time_record, X_flat)

# Transpose u_matrix so time is on the X-axis and space is on the Y-axis
contour = ax2.contourf(T_grid, X_grid, u_matrix.T, levels=50, cmap='RdBu_r')
ax2.set_xlabel('Time (t)', fontsize=12)
ax2.set_ylabel('Space (x)', fontsize=12)
ax2.set_title('KSE Spatiotemporal Evolution', fontsize=14)
plt.colorbar(contour, ax=ax2, orientation='horizontal', fraction=0.05, pad=0.15)

plt.tight_layout()
plt.show()