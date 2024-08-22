import numpy as np
import matplotlib.pyplot as plt
from trajectory_planning_helpers.calc_head_curv_num import calc_head_curv_num

# # Example path: A circular path
# theta = np.linspace(0, 2 * np.pi, 100)
# radius = 5
# x = radius * np.cos(theta)
# y = radius * np.sin(theta)
# path = np.vstack((x, y)).T

# # Element lengths: Approximate using differences in the path points
# el_lengths = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))

# # Append an additional length (e.g., last element) to match the path length
# el_lengths = np.append(el_lengths, el_lengths[-1])

# Parameters for the path
straight_length = 20
curve_radius = 5
num_straight_points = 50
num_curve_points = 50

# Generate the straight part of the path
x_straight = np.linspace(0, straight_length, num_straight_points)
y_straight = np.zeros(num_straight_points)

# Generate the curved part of the path (a quarter circle)
theta_curve = np.linspace(0, np.pi/2, num_curve_points)  # Quarter circle
x_curve = straight_length + curve_radius * np.sin(theta_curve)
y_curve = -curve_radius * (1 - np.cos(theta_curve))

# Combine the straight and curved parts
x = np.concatenate([x_straight, x_curve])
y = np.concatenate([y_straight, y_curve])
path = np.vstack((x, y)).T

# Calculate element lengths (distances between consecutive points)
el_lengths = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))

# Append an additional length to match the path length
#el_lengths = np.append(el_lengths, el_lengths[-1])

# Call the calc_head_curv_num function
psi, kappa = calc_head_curv_num(
    path=path,
    el_lengths=el_lengths,
    is_closed=False,
    stepsize_psi_preview=1.0,
    stepsize_psi_review=1.0,
    stepsize_curv_preview=2.0,
    stepsize_curv_review=2.0,
    calc_curv=True
)

# Visualization
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot the path
axs[0].plot(path[:, 0], path[:, 1], label='Path')
axs[0].set_title('Path')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].axis('equal')
axs[0].legend()

# Plot the heading (psi)
axs[1].plot(np.arange(len(psi)), psi, label='Heading (psi)')
axs[1].set_title('Heading (psi) along the Path')
axs[1].set_xlabel('Point Index')
axs[1].set_ylabel('Heading (radians)')
axs[1].legend()

# Plot the curvature (kappa)
axs[2].plot(np.arange(len(kappa)), kappa, label='Curvature (kappa)', color='r')
axs[2].set_title('Curvature (kappa) along the Path')
axs[2].set_xlabel('Point Index')
axs[2].set_ylabel('Curvature (1/m)')
axs[2].legend()

plt.tight_layout()
plt.show()
