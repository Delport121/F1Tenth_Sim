import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import minimize
import os
from f1tenth_benchmarks.data_tools.plotting_utils import *
from f1tenth_benchmarks.utils.MapData import MapData
from f1tenth_benchmarks.data_tools.plotting_utils import *
from f1tenth_benchmarks.utils.track_utils import CentreLine
from f1tenth_benchmarks.utils.track_utils import RaceTrack, CentreLine
from trajectory_planning_helpers.calc_head_curv_num import calc_head_curv_num
from scipy import interpolate



# Define weighting factors for distance, rotation, and translation
w_dist = 1.0
w_rot = 0.5
w_trans = 0.5

def generate_global_map(map_name = "gbr"):
    """ Generate a set of points to represent the track centerline and boundaries"""
   
    map_name = "gbr"
    centre_line = CentreLine(map_name.lower(), directory="maps/")  #Make sure your in the coorect directory
    track = centre_line
    kappa = centre_line.kappa
    resampled_points, smooth_line = resample_track_points(centre_line.path, seperation_distance=0.2, smoothing=0.5)
    
    el_lengthsSmooth = np.sqrt(np.sum(np.diff(smooth_line, axis=0)**2, axis=1))
    psiSmooth, kappaSmooth = calc_head_curv_num(
                path=smooth_line,
                el_lengths=el_lengthsSmooth,
                is_closed=False,
                stepsize_psi_preview=0.1,
                stepsize_psi_review=0.1,
                stepsize_curv_preview=0.2,
                stepsize_curv_review=0.2,
                calc_curv=True
            )

    el_lengthsSmooth2 = np.sqrt(np.sum(np.diff(resampled_points, axis=0)**2, axis=1))
    print(f'Average lenght between points on resampled map: {np.mean(el_lengthsSmooth2)}')
    psiSmooth2, kappaSmooth2 = calc_head_curv_num(
                path=resampled_points,
                el_lengths=el_lengthsSmooth2,
                is_closed=False,
                stepsize_psi_preview=0.1,
                stepsize_psi_review=0.1,
                stepsize_curv_preview=0.2,
                stepsize_curv_review=0.2,
                calc_curv=True
            )    
    
    plt.plot(kappa, label='Kappa')
    plt.plot(np.flip(-kappaSmooth2), label='Kappa Smooth2')
    plt.legend()
    plt.show()
    
    return kappa, kappaSmooth2

def interpolate_track_new(points, n_points=None, s=0):
        if len(points) <= 1:
            return points
        order_k = min(3, len(points) - 1)
        tck = interpolate.splprep([points[:, 0], points[:, 1]], k=order_k, s=s)[0]
        if n_points is None: n_points = len(points)
        track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T
        return track

def resample_track_points(points, seperation_distance=0.2, smoothing=0.2):
        if points[0, 0] > points[-1, 0]:
            points = np.flip(points, axis=0)

        line_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        n_pts = max(int(line_length / seperation_distance), 2)
        smooth_line = interpolate_track_new(points, None, smoothing)
        resampled_points = interpolate_track_new(smooth_line, n_pts, 0)

        return resampled_points, smooth_line

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
    
    local_x, local_y = local_track[:, 0], local_track[:, 1]
    el_lengthsLocal = np.sqrt(np.sum(np.diff(local_track, axis=0)**2, axis=1))
    
    psiLocal, kappaLocal = calc_head_curv_num(
            path=local_track,
            el_lengths=el_lengthsLocal,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )
    
    # plt.plot(scan_xs, scan_ys, '.', color=free_speech, alpha=0.5, label='Raw Scan Points')
    # plt.plot(correct_scanx, correct_scany, '.', color=sunset_orange, alpha=0.5, label='Corrected Scan Points')
    # map_data.plot_map_img_light()
    # plt.legend()
    # plt.show()
    
    scan_pts = np.vstack([scan_xs, scan_ys])
    
    return scan_pts, position, orientation, kappaLocal
    
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

def icp_scan_matching_1d_bruteforce(global_map, local_scan):
    def compute_error(translation, global_map, local_scan):
        """ Compute the cost based on the distance between the shifted local_scan and the global_map """
        # Translate the local scan along the x-axis
        transformed_scan_x = np.arange(len(local_scan)) + translation

        # Clip the transformed scan x positions to stay within the bounds of the global map
        transformed_scan_x = np.clip(transformed_scan_x, 0, len(global_map) - 1)

        # Compute absolute distance between the corresponding points in global_map and local_scan
        distances = np.abs(global_map[transformed_scan_x.astype(int)] - local_scan)

        # Sum of all distances as the cost
        distance_cost = np.sum(distances)
        return distance_cost

    best_translation = None
    best_cost = np.inf
    all_costs = []

    # Plot the global map as a 1D signal
    plt.plot(np.arange(len(global_map)), global_map, c='orange', label='Global Map', alpha=0.5)

    # Slide the local scan across all possible positions in the global map
    for translation in range(len(global_map) - len(local_scan) + 1):
        # Compute the cost for the current translation
        cost = compute_error(translation, global_map, local_scan)
        all_costs.append(cost)

        # Plot the local scan for this translation (for visualization)
        plt.plot(np.arange(len(local_scan)) + translation, local_scan, c='red', alpha=0.2)

        # Keep track of the best translation (lowest cost)
        if cost < best_cost:
            best_cost = cost
            best_translation = translation

    # Compute the best transformed scan using the best translation
    best_transformed_scan_x = np.arange(len(local_scan)) + best_translation
    best_transformed_scan = global_map[best_transformed_scan_x.astype(int)]

    # Plot the final best transformed local scan
    plt.plot(best_transformed_scan_x, best_transformed_scan, c='green', label='Best Fit', alpha=0.8)

    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.plot(all_costs)
    plt.show()

    print(f"All Costs: {all_costs}")
    # Return the best translation, transformed scan, and cost
    return best_translation, best_transformed_scan, best_cost





# Example usage (same as before)
local_map_number = 400
local_map_number = 800
local_map_number = 0 
kappa, resampledkappa = generate_global_map("gbr")  # Global map points
local_scan, position, orientation , kappalocal = get_local_maps("LocalMPCC", "mu60", "gbr", local_map_number)

x_coords = np.arange(len(resampledkappa))
resampledkappa = np.flip(-resampledkappa)
# coordinates = np.column_stack((x_coords, resampledkappa))

# local_x_coords = np.arange(len(kappalocal))
# local_coordinates = np.column_stack((local_x_coords, kappalocal))
# # local_coordinates = np.flip(-local_coordinates)

# # # Perform scan matching for local scan and store results
results = icp_scan_matching_1d_bruteforce(resampledkappa, kappalocal)
# # print(results)

# # Highlight the best matching scan
best_transformation, best_transformed_scan, best_cost = results
print(best_transformation)
print(best_cost)


# plt.plot(coordinates[:, 0], coordinates[:, 1], c='black', label='Global Map', alpha=0.5)
# plt.plot(best_transformed_scan[:, 1], label='Best Transformed Scan')
# plt.legend()
# plt.show()

# plt.scatter(global_map[:, 0], global_map[:, 1], c='black', label='Global Map', alpha=0.5, s=0.5)
# plt.scatter(local_scan[0], local_scan[1], c='red', label='Local Scan', alpha=0.5, s=0.5)
# plt.scatter(best_transformed_scan[:, 0], best_transformed_scan[:, 1], c='green', label=f'Best Match (Cost: {best_cost:.2f})',alpha=0.5, s=0.5)
# plt.legend()
# plt.show()

# Visualize the results
# visualize_multiple_scan_matching(global_map, [scan for scan, _ in local_scans], results)
# visualize_multiple_scan_matching(global_map, local_scan, results)