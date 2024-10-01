import numpy as np
import matplotlib.pyplot as plt

def plot_curvature(x_list, y_list, heading_list, curvature,
                   k=0.01, c="-c", label="Curvature"):
    """
    Plot curvature on 2D path. This plot is a line from the original path,
    the lateral distance from the original path shows curvature magnitude.
    Left turning shows right side plot, right turning shows left side plot.
    For straight path, the curvature plot will be on the path, because
    curvature is 0 on the straight path.

    Parameters
    ----------
    x_list : array_like
        x position list of the path
    y_list : array_like
        y position list of the path
    heading_list : array_like
        heading list of the path
    curvature : array_like
        curvature list of the path
    k : float
        curvature scale factor to calculate distance from the original path
    c : string
        color of the plot
    label : string
        label of the plot
    """
    cx = [x + d * k * np.cos(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]
    cy = [y + d * k * np.sin(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]

    plt.plot(cx, cy, c, label=label)
    for ix, iy, icx, icy in zip(x_list, y_list, cx, cy):
        plt.plot([ix, icx], [iy, icy], c)


# Sample path data
x_list = np.linspace(0, 10, 100)  # X positions from 0 to 10
y_list = np.sin(x_list)           # Y positions are a sine wave for illustration

# Simulate heading (in radians)
heading_list = np.arctan2(np.gradient(y_list), np.gradient(x_list))

# Simulate curvature
curvature = np.gradient(heading_list) / np.gradient(np.sqrt(np.gradient(x_list)**2 + np.gradient(y_list)**2))

# Plot the original path
plt.plot(x_list, y_list, label="Original Path")

# Plot the curvature using the function
plot_curvature(x_list, y_list, heading_list, curvature, k=0.1, c="r-", label="Curvature Plot")

# Show the plot
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Curvature Plot Example")
plt.legend()
plt.grid(True)
plt.show()


