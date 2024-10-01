import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import minimize
from f1tenth_benchmarks.data_tools.plotting_utils import *
from f1tenth_benchmarks.utils.MapData import MapData
from f1tenth_benchmarks.data_tools.plotting_utils import *
from f1tenth_benchmarks.utils.track_utils import CentreLine


# Define weighting factors for distance, rotation, and translation
w_dist = 1.0
w_rot = 0.5
w_trans = 0.5

def generate_global_map(map_name = "gbr"):
    """ Generate a set of points to represent the track centerline and boundaries"""
   
    map_name = "gbr"
    map_data = MapData(map_name)
   
    # Map_origin = map_data.map_origin
    # map_data.plot_map_img_light()
    # plt.scatter(Map_origin[0], Map_origin[1], c='red', label='Map Origin')
    # plt.show()
    
    track = CentreLine(map_name)

    # Plotting the track centerline
    track.path[:, 0] = (track.path[:, 0] - map_data.map_origin[0]) / map_data.map_resolution
    track.path[:, 1] = (track.path[:, 1] - map_data.map_origin[1]) / map_data.map_resolution
    # plt.plot(track.path[:, 0], track.path[:, 1], '--', linewidth=2, color='black')

    # Plotting the inner and outer boundaries of the track
    track.widths = track.widths / map_data.map_resolution
    l1 = track.path[:, :2] + track.nvecs * track.widths[:, 0][:, None]
    l2 = track.path[:, :2] - track.nvecs * track.widths[:, 1][:, None]
    # plt.plot(l1[:, 0], l1[:, 1], color='green')
    # plt.plot(l2[:, 0], l2[:, 1], color='green')
    # plt.scatter(l1[:, 0], l1[:, 1], color='green')
    # plt.scatter(l2[:, 0], l2[:, 1], color='green')

    # # Display the plot
    # plt.show()
    
    return np.vstack((l1, l2))

def get_local_maps(planner_name, test_id, map_name="aut", i = 0):
    """ Extracts one  of the local maps, and also get poistion of car in the map"""
    root = f"Logs/{planner_name}/"
    localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
    try:
        Logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
        scans = np.load(root + f"RawData_{test_id}/ScanLog_{map_name}_0.npy")
    except:
        Logs, scans = None, None
        
    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)
    
    scan_xs, scan_ys = scans[i+1] * np.array([coses, sines])
    position = Logs[i+1, :2]
    orientation = Logs[i+1, 4]
    
    local_track = np.load(localmap_data_path + f"local_map_{i}.npy")
    
    scan_pts = np.vstack([scan_xs, scan_ys]).T
    scan_pts = reoreintate_pts(scan_pts, [0,0], 0)
    Corrected_scan_pts = reoreintate_pts(scan_pts, position, orientation)
    map_data = MapData(map_name)
    scan_xs, scan_ys = map_data.pts2rc(scan_pts)
    correct_scanx, correct_scany = map_data.pts2rc(Corrected_scan_pts)
    
    plt.plot(scan_xs, scan_ys, '.', color=free_speech, alpha=0.5, label='Raw Scan Points')
    plt.plot(correct_scanx, correct_scany, '.', color=sunset_orange, alpha=0.5, label='Corrected Scan Points')
    map_data.plot_map_img_light()
    plt.legend()
    plt.show()
    
    scan_pts = np.vstack([scan_xs, scan_ys])
    
    return scan_pts, position, orientation
    
def reoreintate_pts(pts, position, theta):
    rotation_mtx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = np.matmul(pts, rotation_mtx.T) + position

    return pts

def transform_points(points, transformation):
    """ Apply a transformation to the points (2D rotation + translation) """
    theta, tx, ty = transformation
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return points @ rotation_matrix.T + np.array([tx, ty])

def icp_scan_matching(global_map, local_scan, max_iterations=100, tolerance=1e-7, num_initial_guesses=5):
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
    all_costs = []
    
    # List to store cost at each iteration
    cost_convergence = []

    plt.scatter(global_map[:, 0], global_map[:, 1], c='black', label='Global Map', alpha=0.5, s=0.5)
    for p in range(num_initial_guesses):
        # Generate random initial guess for [theta, tx, ty]
        initial_guess = np.array([np.random.uniform(0, 2 * np.pi),   # Random rotation
                                  np.random.uniform(-20, 1200),         # Random translation (x)
                                  np.random.uniform(-20, 600)])        # Random translation (y)
        
        plt.scatter(local_scan[:, 0], local_scan[:, 1], c='red', label='Local Scan', alpha=0.5, s=0.5)
        initial_guess_transformed = transform_points(local_scan, initial_guess)
        plt.scatter(initial_guess_transformed[:, 0], initial_guess_transformed[:, 1], c='green', label='Initial Guess', alpha=0.5, s=0.5)
        
        # Callback function to store cost at each iteration
        def callback(transformation):
            current_cost = compute_error(transformation, global_map, local_scan)
            cost_convergence.append(current_cost)

        # Run optimization with current random initial guess
        result = minimize(
            compute_error, 
            initial_guess, 
            args=(global_map, local_scan), 
            method='L-BFGS-B',
            options={'maxiter': max_iterations, 'ftol': tolerance},
            callback=callback  # Track cost at each iteration
        )
        
        cost = result.fun
        all_costs.append(cost)

        # Compare result cost to find the best one
        if result.fun < best_cost:
            best_cost = result.fun
            best_transformation = result.x
            best_transformed_scan = transform_points(local_scan, result.x)
    
    
    # Plot cost convergence
    plt.figure()
    plt.plot(cost_convergence, label="Cost during optimization")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (ftol)")
    plt.title("Convergence of the Optimization Process")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"All Costs: {all_costs}")
    # Return the best transformation, transformed scan, and cost
    return best_transformation, best_transformed_scan, best_cost

# Example usage (same as before)
local_map_number = 40
global_map = generate_global_map("gbr")  # Global map points
local_scan, position, orientation = get_local_maps("LocalMPCC", "mu60", "gbr", local_map_number)

# # Perform scan matching for local scan and store results
results = icp_scan_matching(global_map, local_scan.T, max_iterations=100, tolerance=1e-7, num_initial_guesses=10)
# print(results)

# Highlight the best matching scan
best_transformation, best_transformed_scan, best_cost = results
print(best_transformation)

plt.scatter(global_map[:, 0], global_map[:, 1], c='black', label='Global Map', alpha=0.5, s=0.5)
plt.scatter(local_scan[0], local_scan[1], c='red', label='Local Scan', alpha=0.5, s=0.5)
plt.scatter(best_transformed_scan[:, 0], best_transformed_scan[:, 1], c='green', label=f'Best Match (Cost: {best_cost:.2f})',alpha=0.5, s=0.5)
plt.legend()
plt.show()

# Visualize the results
# visualize_multiple_scan_matching(global_map, [scan for scan, _ in local_scans], results)
# visualize_multiple_scan_matching(global_map, local_scan, results)