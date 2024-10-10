import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats

# Step 1: Generate a random 3x3 covariance matrix and means
mean = [0, 0, 0]  # Adding theta as the third state
cov_matrix = np.array([[3.0, 1.5, 0.0], 
                       [1.5, 2.0, 0.0],
                       [0.0, 0.0, 0.5]])  # Variance for theta added

# Step 2: Define function to plot the covariance ellipse
def plot_covariance_ellipse(ax, mean, cov_matrix, n_std=1.0, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov_matrix[:2, :2])  # Only for x and y
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean[:2], width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return angle, eigvecs

# Step 3: Project distributions onto new axes
def project_distribution(mean, cov_matrix, angle, x_values):
    # Define the rotation matrix for the tangent direction
    rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                 [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]])

    # Project the marginal distributions
    proj_mean = np.dot(rotation_matrix, mean[:2])
    proj_cov = rotation_matrix @ cov_matrix[:2, :2] @ rotation_matrix.T

    # Compute the projected marginal distribution
    projected_marginal = stats.norm.pdf(x_values, proj_mean[0], np.sqrt(proj_cov[0, 0]))
    
    return proj_mean, projected_marginal

# Step 4: Plot covariance ellipse and marginal distributions
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

# Covariance ellipse plot
angle, eigvecs = plot_covariance_ellipse(ax[0], mean, cov_matrix, edgecolor='blue', fc='none', lw=2)
ax[0].set_xlim(-6, 6)
ax[0].set_ylim(-6, 6)
ax[0].scatter(mean[0], mean[1], c='red', label='Mean')
ax[0].set_title("Covariance Ellipse (X-Y Plane)")
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].legend()

# Marginal distribution of X
x_values = np.linspace(-6, 6, 100)
x_marginal = stats.norm.pdf(x_values, mean[0], np.sqrt(cov_matrix[0, 0]))
ax[1].plot(x_values, x_marginal, color='blue')
ax[1].set_title("Marginal Distribution of X")
ax[1].set_xlabel('X')
ax[1].set_ylabel('Density')

# Marginal distribution of Y
y_values = np.linspace(-6, 6, 100)
y_marginal = stats.norm.pdf(y_values, mean[1], np.sqrt(cov_matrix[1, 1]))
ax[2].plot(y_values, y_marginal, color='green')
ax[2].set_title("Marginal Distribution of Y")
ax[2].set_xlabel('Y')
ax[2].set_ylabel('Density')

# Project distributions onto Frenet axes
frenet_x_values = np.linspace(-6, 6, 100)
proj_mean, proj_marginal = project_distribution(mean, cov_matrix, angle, frenet_x_values)

# Plotting the projected marginal distribution
ax[3].plot(frenet_x_values, proj_marginal, color='orange', label='Projected Marginal (Tangent)')
ax[3].set_title("Projected Marginal Distribution onto Frenet Axis")
ax[3].set_xlabel('Frenet Tangent Axis')
ax[3].set_ylabel('Density')
ax[3].legend()

plt.tight_layout()
plt.show()
