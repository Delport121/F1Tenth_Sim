import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Generate noisy data points along a parametric curve (e.g., a circle)
theta = np.linspace(0, 2 * np.pi, 100)
x = np.cos(theta) + 0.1 * np.random.randn(100)
y = np.sin(theta) + 0.1 * np.random.randn(100)

# Fit a B-spline to the data
tck, u = splprep([x, y], s=0)

# Evaluate the B-spline at more points to get a smooth curve
u_fine = np.linspace(0, 1, 400)
x_smooth, y_smooth = splev(u_fine, tck)

# Plot the original data points and the B-spline approximation
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'bo', label='Noisy data points')
plt.plot(x_smooth, y_smooth, 'r-', label='B-spline fit')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('B-spline fit using splprep and splev')
plt.axis('equal')
plt.show()
