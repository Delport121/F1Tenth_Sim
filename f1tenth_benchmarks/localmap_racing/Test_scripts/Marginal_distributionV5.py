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
    return ellipse

# Step 3: Function to project marginal distributions onto new axes
def project_distribution_on_axes(mean, cov_matrix, angle, translation):
    # Create rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle),  np.cos(angle)]])
    
    # Calculate new mean
    new_mean = rotation_matrix @ np.array(mean[:2]) + translation
    
    # Calculate new covariance
    new_cov = rotation_matrix @ cov_matrix[:2, :2] @ rotation_matrix.T
    
    return new_mean, new_cov

# Step 4: Plot covariance ellipse and marginal distributions
fig, ax = plt.subplots(1, 5, figsize=(25, 5))

# Covariance ellipse plot
ax[0].set_xlim(-6, 6)
ax[0].set_ylim(-6, 6)
ax[0].scatter(mean[0], mean[1], c='red', label='Mean')
plot_covariance_ellipse(ax[0], mean, cov_matrix, edgecolor='blue', fc='none', lw=2, label='Original Covariance Ellipse')
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

# Generate random angle for new axes (in radians)
random_angle = np.random.uniform(0, 2 * np.pi)

# Define arbitrary translation away from the mean
translation = np.array([2, 1])  # Translation vector (x, y)

# Project distributions onto new axes
new_mean, new_cov = project_distribution_on_axes(mean, cov_matrix, random_angle, translation)

# Marginal distribution along the new axes
new_x_values = np.linspace(-6, 6, 100)
new_x_marginal = stats.norm.pdf(new_x_values, new_mean[0], np.sqrt(new_cov[0, 0]))

# Marginal distribution along the orthogonal axis
new_y_values = np.linspace(-6, 6, 100)
new_y_marginal = stats.norm.pdf(new_y_values, new_mean[1], np.sqrt(new_cov[1, 1]))

# Plotting the new axes
ax[0].plot([new_mean[0], new_mean[0] + 6 * np.cos(random_angle)], 
           [new_mean[1], new_mean[1] + 6 * np.sin(random_angle)], color='orange', lw=2, label='New Axis 1')
ax[0].plot([new_mean[0], new_mean[0] + 6 * np.cos(random_angle + np.pi/2)], 
           [new_mean[1], new_mean[1] + 6 * np.sin(random_angle + np.pi/2)], color='purple', lw=2, label='New Axis 2')

# Plotting the new covariance ellipse
plot_covariance_ellipse(ax[0], new_mean, new_cov, edgecolor='orange', fc='none', lw=2, label='New Covariance Ellipse')

ax[0].legend()

# Plotting the projected marginal distributions
ax[3].plot(new_x_values, new_x_marginal, color='orange')
ax[3].set_title("Projected Marginal Distribution on New Axis 1")
ax[3].set_xlabel('New X Axis')
ax[3].set_ylabel('Density')

ax[4].plot(new_y_values, new_y_marginal, color='purple')
ax[4].set_title("Projected Marginal Distribution on New Axis 2")
ax[4].set_xlabel('New Y Axis')
ax[4].set_ylabel('Density')

plt.tight_layout()
plt.show()
