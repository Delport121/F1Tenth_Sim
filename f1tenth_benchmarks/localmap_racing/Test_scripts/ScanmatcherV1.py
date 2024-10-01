import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import minimize

# Define weighting factors for distance, rotation, and translation
w_dist = 1.0
w_rot = 0.5
w_trans = 0.5

def generate_global_map(num_points=100):
    """ Generate a set of points along a circular path (global map) """
    theta = np.linspace(0, 2 * np.pi, num_points)
    radius = 50
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.vstack((x, y)).T

def get_local_map(global_map, num_local_points=20):
    """ Extract a random segment from the global map and apply a random transformation """
    start_idx = np.random.randint(0, len(global_map) - num_local_points)
    local_map = global_map[start_idx:start_idx + num_local_points]

    # Apply a random rotation and translation
    theta = np.random.uniform(0, 2 * np.pi)  # Random rotation
    tx = np.random.uniform(-20, 20)          # Random translation (x)
    ty = np.random.uniform(-20, 20)          # Random translation (y)

    transformation = np.array([theta, tx, ty])
    transformed_local_map = transform_points(local_map, transformation)

    return transformed_local_map, transformation

def transform_points(points, transformation):
    """ Apply a transformation to the points (2D rotation + translation) """
    theta, tx, ty = transformation
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return points @ rotation_matrix.T + np.array([tx, ty])

def icp_scan_matching(global_map, local_scan, max_iterations=50, tolerance=1e-4, num_initial_guesses=5):
    def compute_error(transformation, global_map, local_scan):
        """ Compute a weighted cost based on distance, translation, and rotation """
        transformed_scan = transform_points(local_scan, transformation)
        distances, _ = kdtree.query(transformed_scan)
        
        # Distance cost: sum of distances between closest points
        distance_cost = np.sum(distances)
        
        # Rotation cost: absolute value of the rotation angle
        rotation_cost = np.abs(transformation[0])  # theta
        
        # Translation cost: Euclidean distance of the translation vector
        translation_cost = np.sqrt(transformation[1]**2 + transformation[2]**2)  # tx, ty
        
        # Total cost as weighted sum of the components
        total_cost = w_dist * distance_cost + w_rot * rotation_cost + w_trans * translation_cost
        return total_cost

    # Create KDTree for fast nearest neighbor search
    kdtree = KDTree(global_map)

    best_transformation = None
    best_transformed_scan = None
    best_cost = np.inf

    for _ in range(num_initial_guesses):
        # Generate random initial guess for [theta, tx, ty]
        initial_guess = np.array([np.random.uniform(0, 2 * np.pi),   # Random rotation
                                  np.random.uniform(-20, 20),         # Random translation (x)
                                  np.random.uniform(-20, 20)])        # Random translation (y)
        
        # Run optimization with current random initial guess
        result = minimize(
            compute_error, 
            initial_guess, 
            args=(global_map, local_scan), 
            method='L-BFGS-B',
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )
        
        # Compare result cost to find the best one
        if result.fun < best_cost:
            best_cost = result.fun
            best_transformation = result.x
            best_transformed_scan = transform_points(local_scan, result.x)

    # Return the best transformation, transformed scan, and cost
    return best_transformation, best_transformed_scan, best_cost

def visualize_multiple_scan_matching(global_map, local_scans, results):
    """ Visualize multiple local scans with their respective costs and the best transformation """
    plt.figure(figsize=(10, 10))
    plt.scatter(global_map[:, 0], global_map[:, 1], c='blue', label='Global Map', alpha=0.5)
    
    best_scan_idx = np.argmin([result[2] for result in results])
    
    # Define distinct colors for the local scans
    colors = plt.cm.viridis(np.linspace(0, 1, len(local_scans)))
    
    # Plot each local scan with its cost
    for idx, (local_scan, (_, transformed_scan, cost)) in enumerate(zip(local_scans, results)):
        color = colors[idx] if idx != best_scan_idx else 'green'
        plt.scatter(local_scan[:, 0], local_scan[:, 1], label=f'Local Scan {idx+1} (Cost: {cost:.2f})', alpha=0.7, c=color)
    
    # Highlight the best matching scan
    best_transformation, best_transformed_scan, best_cost = results[best_scan_idx]
    plt.scatter(best_transformed_scan[:, 0], best_transformed_scan[:, 1], c='green', label=f'Best Match (Cost: {best_cost:.2f})', alpha=0.7)
    
    # Show transformation matrix for the best match
    theta, tx, ty = best_transformation
    transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), tx],
                                      [np.sin(theta), np.cos(theta), ty],
                                      [0, 0, 1]])
    plt.text(-50, 70, f'Transformation Matrix (Best Match):\n{transformation_matrix}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.legend()
    plt.title('Multiple Scan Matching with Costs')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Example usage (same as before)
global_map = generate_global_map(100)  # Global map points (e.g., circular path)

# Generate multiple random local scans
num_local_scans = 5
local_scans = [get_local_map(global_map) for _ in range(num_local_scans)]

# Perform scan matching for each local scan and store results
results = [icp_scan_matching(global_map, local_scan, num_initial_guesses=10) for local_scan, _ in local_scans]

# Visualize the results
visualize_multiple_scan_matching(global_map, [scan for scan, _ in local_scans], results)
