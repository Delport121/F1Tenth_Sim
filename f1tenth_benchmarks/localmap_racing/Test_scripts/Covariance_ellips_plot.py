import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_covariance_ellipse(mean, cov, n_std=2.0, ax=None, **kwargs):
    """
    Plots a covariance ellipse representing the covariance matrix.

    Parameters:
        mean (array-like): The mean of the Gaussian distribution, [x, y].
        cov (2x2 array-like): The covariance matrix of the Gaussian distribution.
        n_std (float): Number of standard deviations for the ellipse radius (default is 2).
        ax (matplotlib.axes.Axes, optional): The axes on which to plot the ellipse. 
                                             If not provided, a new plot will be created.
        **kwargs: Additional keyword arguments to pass to the Ellipse patch (e.g., color, alpha).

    Returns:
        matplotlib.patches.Ellipse: The ellipse patch representing the covariance.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Compute the angle for the ellipse based on the first eigenvector
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Compute the width and height of the ellipse based on eigenvalues
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    # Create the ellipse and add it to the plot
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    
    # Plot the mean
    ax.plot(*mean, 'ro', markersize=5)  # Mark the mean point

    # Set axis limits to ensure ellipse is fully visible
    margin = max(width, height) / 2
    ax.set_xlim(mean[0] - margin, mean[0] + margin)
    ax.set_ylim(mean[1] - margin, mean[1] + margin)
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.show()
    return ellipse

mean = [5, 5]
cov = [[4, 1.5], 
       [1.5, 3]]
plot_covariance_ellipse(mean, cov, n_std=2, edgecolor='blue', facecolor='none', linestyle='--')
