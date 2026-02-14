import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from get_R import get_R

# 1. Load your newly found fixed point
u_fixed = np.loadtxt(r"2D_KS_adj\fixed_points\test.dat", delimiter=" ") 

# 2. Run the forward simulation (BDF/Radau)
# Use a long enough time T to see it go turbulent
sol = solve_ivp(lambda t,u: get_R(t,u.reshape(64, 64), print_res=True).flatten(), [0, 200], u_fixed.flatten(), method='BDF', rtol=1e-14)

# 3. Calculate the Perturbation Norm for every time step
# sol.y shape is (N_gridpoints, N_timesteps)
time_steps = sol.t
perturbation_norm = []

for i in range(len(time_steps)):
    u_current = sol.y[:, i]
    # Calculate the distance from the initial fixed point
    distance = np.linalg.norm(u_current - u_fixed.flatten())
    perturbation_norm.append(distance)




# 1. Define your predicted eigenvalue (e.g., from Jiang's paper or your Jacobian)
predicted_eigenvalue = 0.3776 

# 2. Choose an "Anchor Point" in the middle of your linear growth phase.
# Look at your DNS graph and pick a time (t) where the line is perfectly straight.
t_anchor = 50  # Adjust this based on your specific graph

# Find the index of this time, and get the corresponding DNS Y-value
idx_anchor = np.searchsorted(time_steps, t_anchor)
y_anchor = perturbation_norm[idx_anchor]

# 3. Generate the theoretical red line
# Equation: y(t) = y_anchor * exp(lambda * (t - t_anchor))
theoretical_line = y_anchor * np.exp(predicted_eigenvalue * (time_steps - t_anchor))

# 4. Plotting
plt.figure(figsize=(8, 6))

# Plot the DNS Data (Blue Line)
plt.semilogy(time_steps, perturbation_norm, '-b', linewidth=2, label='2D KSE DNS')

# Plot the Theoretical Slope (Red Line)
plt.semilogy(time_steps, theoretical_line, '-r', linewidth=2, 
             label=f'Linear stability analysis: slope = {predicted_eigenvalue}')

# Formatting to match the reference style
plt.ylim(1e-14, 1e4) # Adjust to your noise floor and saturation ceiling
plt.xlim(-10, 260)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('||U(t) - U(t_0)||_2', fontsize=12)
plt.title('L2 Norm of Difference Over Time - FixedPoint', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='lower right', fontsize=11)

plt.show()