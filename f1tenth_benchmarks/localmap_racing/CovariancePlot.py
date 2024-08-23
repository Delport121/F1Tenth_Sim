import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Mean vector and covariance matrix
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]

# Create a grid of points for both 2D and 3D plots
x_2d, y_2d = np.mgrid[-3:3:.01, -3:3:.01]
x_3d, y_3d = np.mgrid[-3:3:.1, -3:3:.1]
pos_2d = np.dstack((x_2d, y_2d))
pos_3d = np.dstack((x_3d, y_3d))

# Multivariate normal distribution
rv = multivariate_normal(mean, cov)
z_2d = rv.pdf(pos_2d)
z_3d = rv.pdf(pos_3d)

# Create the figure and subplots
fig = plt.figure(figsize=(16, 8))

# 2D Contour Plot
ax1 = fig.add_subplot(121)
contour = ax1.contourf(x_2d, y_2d, z_2d, levels=30, cmap='viridis')
fig.colorbar(contour, ax=ax1)
ax1.set_title('2D Contour Plot of Multivariate Gaussian Distribution')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')

# 3D Surface Plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_3d, y_3d, z_3d, cmap='viridis', edgecolor='none')
ax2.set_title('3D Surface Plot of Multivariate Gaussian Distribution')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_zlabel('Probability Density')

# Show the combined plots
plt.tight_layout()
plt.show()
