import numpy as np
import yaml
from numba import njit 
from PIL import Image
import os 
from scipy.ndimage import distance_transform_edt as edt
import matplotlib.pyplot as plt

from f1tenth_benchmarks.utils.BasePlanner import load_parameter_file_with_extras
from f1tenth_benchmarks.utils.track_utils import CentreLine
from scipy import interpolate
from trajectory_planning_helpers.calc_head_curv_num import calc_head_curv_num
from f1tenth_benchmarks.localmap_racing.LocalMapGenerator import LocalMapGenerator
from f1tenth_benchmarks.utils.MapData import MapData
from scipy.spatial import KDTree
from scipy.optimize import minimize


class CurvatureFilter:
    def __init__(self, planner_name, test_id, extra_params={}) -> None:
        self.params = load_parameter_file_with_extras("particle_filter_params", extra_params)
        self.planner_name = planner_name
        self.test_id = test_id
        self.data_path = f"Logs/{planner_name}/RawData_{test_id}/"
        self.estimates = None
        self.scan_simulator = None
        self.Q = np.diag(self.params.motion_q_stds) **2 
        self.NP = self.params.number_of_particles
        self.dt = self.params.dt
        self.num_beams = self.params.number_of_beams
        self.lap_number = 0
        self.map_name = None

        self.particles = None
        self.proposal_distribution = None
        self.weights = np.ones(self.NP) / self.NP
        self.particle_indices = np.arange(self.NP)
        
        self.Full_map_curvature = None

        self

    def init_pose(self, init_pose):
        '''Randomly generate a bunch of particles'''
        self.estimates = [init_pose]
        self.proposal_distribution = init_pose + np.random.multivariate_normal(np.zeros(3), self.Q*self.params.init_distribution, self.NP)
        self.particles = self.proposal_distribution
        # plt.plot(self.particles[:,0], self.particles[:,1], 'ro') #(initiallise the filter assuming the car is at the initial pose)
        # plt.show()

        return init_pose

    def set_map(self, map_name):
        self.map_name = map_name
        self.scan_simulator = SensorModel(f"maps/{map_name}", self.test_id, self.num_beams, self.params.fov)
        self.Full_map_curvature = self.scan_simulator.kappaSmooth

        image = self.scan_simulator.map_img
        image_np = np.array(image, dtype=int)
        image_np[image_np <= 128.] = 0
        image_np[image_np > 128.] = 1
        occupancy_grid = image_np
        # print(image_np)
        # # Visualize the occupancy grid
        # plt.imshow(occupancy_grid, cmap='gray', interpolation='nearest')
        # plt.title('Occupancy Grid')
        # plt.show()

    def localise(self, action, observation):
        '''MCL algorithm'''
        vehicle_speed = observation["vehicle_speed"] 
        self.particle_control_update(action, vehicle_speed)

        self.measurement_update(observation["scan"]) 

        estimate = np.dot(self.particles.T, self.weights)
        self.estimates.append(estimate)

        # image = self.scan_simulator.map_img
        # image_np = np.array(image, dtype=int)
        # image_np[image_np <= 128.] = 0
        # image_np[image_np > 128.] = 1
        # occupancy_grid = image_np
        # plt.imshow(occupancy_grid, cmap='gray', interpolation='nearest')

        
        # plt.xlim([-2, 10])
        # plt.ylim([-2, 10])
        # plt.plot(self.particles[:,0], self.particles[:,1], 'ro', markersize=0.2)
        # plt.plot(estimate[0], estimate[1], 'yo', markersize=5)
        # plt.title('Particle distribution')
        # plt.show()

        return estimate

    def particle_control_update(self, control, vehicle_speed):
        '''Add noise to motion model and update particles'''
        next_states = particle_dynamics_update(self.proposal_distribution, control, vehicle_speed, self.dt, self.params.wheelbase) #next_states(x, y, theta)
        random_samples = np.random.multivariate_normal(np.zeros(3), self.Q, self.NP)
        self.particles = next_states + random_samples

    def measurement_update(self, measurement):
        '''Update the weights of the particles based on the measurement
            The measurement in this case is the actual lidar scan from the car
        '''
        global_track = self.scan_simulator.centreline_resampled
        # Get actual scan data from the car
        localTrack = self.scan_simulator.ExtractLocalTrack(measurement)
        localTrack = localTrack[:, :2] # Extract only the x and y coordinates
        
        # plt.plot(global_track[:, 0], global_track[:, 1], 'ro')
        # plt.plot(localTrack[:, 0], localTrack[:, 1], 'bo')
        # plt.show()
        
        best_transformation, best_transformed_scan, costs = self.scan_simulator.icp_scan_matching(global_track, localTrack, max_iterations=10, tolerance=1e-7, initial_guesses=self.particles)
        # plt.plot(costs)
        # plt.show()
        # plt.plot(localTrack[:, 0], localTrack[:, 1], 'ro')
        # plt.show()
       
        # # Simulate scans for each particle
        # particle_measurements = np.zeros((self.NP, self.num_beams))
        # for i, state in enumerate(self.particles): 
        #     particle_measurements[i] = self.scan_simulator.scan(state)

        # # # Importance sampling 
        # # z = particle_measurements - measurement
        # # sigma = np.clip(np.sqrt(np.average(z**2, axis=0)), 0.01, 10)
        # # weights =  np.exp(-z ** 2 / (2 * sigma ** 2))
        # # self.weights = np.prod(weights, axis=1)
        # # # Normalize weights
        # # self.weights = self.weights / np.sum(self.weights)
        
        # correlation_scores = np.correlate(np.flip(-self.Full_map_curvature), kappaLocal, mode='valid')
        # # Shift the scores to ensure all are positive (if necessary)
        # min_score = np.min(correlation_scores)
        # shifted_scores = correlation_scores - min_score  # Now all scores are >= 0
        # # Normalize the scores to a range between 0 and 1
        # normalized_scores = shifted_scores / np.max(shifted_scores)
        # # Convert to probabilities by dividing by the sum of the normalized scores
        # probabilities = normalized_scores / np.sum(normalized_scores)
        # self.weights = probabilities
        
        # print(f"Particle indices: {len(self.particle_indices)}")
        # print(f"NP: {self.NP}")
        # print(f"Self weights: {len(self.weights)}")

        # # Resampling
        # proposal_indices = np.random.choice(self.particle_indices, self.NP, p=self.weights)
        # self.proposal_distribution = self.particles[proposal_indices,:]

    def lap_complete(self):
        estimates = np.array(self.estimates)
        np.save(self.data_path + f"cf_estimates_{self.map_name}_{self.lap_number}.npy", estimates)
        self.lap_number += 1



def particle_dynamics_update(states, actions, speed, dt, L):
    '''Motion model: Update the particles using the bicycle model'''
    states[:, 0] += speed * np.cos(states[:, 2]) * dt
    states[:, 1] += speed * np.sin(states[:, 2]) * dt
    states[:, 2] += speed * np.tan(actions[0]) / L * dt
    return states



class SensorModel:
    def __init__(self, map_name, test_id,  num_beams, fov, eps=0.01, theta_dis=2000, max_range=30.0):
        self.test_id = test_id
        self.num_beams = num_beams
        self.fov = fov
        self.eps = eps
        self.theta_dis = theta_dis
        self.max_range = max_range
        self.angle_increment = self.fov / (self.num_beams - 1)
        self.theta_index_increment = theta_dis * self.angle_increment / (2. * np.pi)
        self.orig_x = None
        self.orig_y = None
        self.map_img = None
        self.map_height = None
        self.map_width = None
        self.map_resolution = None
        self.dt = None
        
        self.map_data = None
        self.number_of_track_points = None
        self.centreline_raw = None
        self.centreline_resampled = None
        self.L1 = None
        self.L2 = None
        
        self.psi_raw = None
        self.kappa_raw = None
        self.psiSmooth = None
        self.kappaSmooth = None
        self.number_of_track_points = None
        
        theta_arr = np.linspace(0.0, 2*np.pi, num=theta_dis)
        self.sines = np.sin(theta_arr)
        self.cosines = np.cos(theta_arr)
        
        # Define weighting factors for distance, rotation, and translation
        self.w_dist = 1.0
        self.w_rot = 0.5
        self.w_trans = 0.5
        
        self.load_map(map_name)
        self.FeatureExtractor = LocalMapGenerator(map_name, 0, False) #False to avoid saving data
        
    
    def load_map(self, map_path):
        # load map image
       
        map_name = os.path.splitext(os.path.basename(map_path))[0]
        map_img_path = os.path.splitext(map_path)[0] + ".png"
        map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        map_img = map_img.astype(np.float64)
        # print(map_path)
        # print(map_name)

         # grayscale -> binary
        map_img[map_img <= 128.] = 0.
        map_img[map_img > 128.] = 255.
        self.map_img = map_img

        self.map_height = map_img.shape[0]
        self.map_width = map_img.shape[1]

        with open(map_path + ".yaml", 'r') as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            self.map_resolution = map_metadata['resolution']
            self.origin = map_metadata['origin']

        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]

        self.dt = self.map_resolution * edt(map_img)
        
        
        self.map_data = MapData(map_name)
        
        # Map_origin = self.map_data.map_origin
        # self.map_data.plot_map_img_light()
        # plt.scatter(Map_origin[0], Map_origin[1], c='red', label='Map Origin')
        # plt.show()
        
        #Initialise centreline class for preprocessing 
        track = CentreLine(map_name)
        
        # Plotting the track centerline
        # track.path[:, 0] = (track.path[:, 0] - self.map_data.map_origin[0]) / self.map_data.map_resolution
        # track.path[:, 1] = (track.path[:, 1] - self.map_data.map_origin[1]) / self.map_data.map_resolution
        # plt.plot(track.path[:, 0], track.path[:, 1], '--', linewidth=2, color='black')

        # Plotting the inner and outer boundaries of the track
        track.widths = track.widths / self.map_data.map_resolution
        self.L1 = track.path[:, :2] + track.nvecs * track.widths[:, 0][:, None]
        self.L2 = track.path[:, :2] - track.nvecs * track.widths[:, 1][:, None]
        # plt.plot(self.L1[:, 0], self.L1[:, 1], color='green')
        # plt.plot(self.L2[:, 0], self.L2[:, 1], color='green')
        # plt.scatter(self.L1[:, 0], self.L1[:, 1], color='green')
        # plt.scatter(self.L2[:, 0], self.L2[:, 1], color='green')

        # # Display the plot
        # plt.show()
        
        self.centreline_raw = track.path
        self.psi_raw = track.psi
        self.kappa_raw = track.kappa
        self.number_of_track_points = len(self.kappa_raw)
        # print(f"Number of track points: {self.number_of_track_points}")
        # print(self.centreline_raw)
        
        
        self.centreline_resampled, smooth_line = resample_track_points(self.centreline_raw, seperation_distance=0.2, smoothing=0.5)
        el_lengthsSmooth = np.sqrt(np.sum(np.diff(self.centreline_resampled, axis=0)**2, axis=1))
        self.psiSmooth, self.kappaSmooth = calc_head_curv_num(
                path=self.centreline_resampled,
                el_lengths=el_lengthsSmooth,
                is_closed=False,
                stepsize_psi_preview=0.1,
                stepsize_psi_review=0.1,
                stepsize_curv_preview=0.2,
                stepsize_curv_review=0.2,
                calc_curv=True
            )

    def scan(self, pose):
        scan = get_scan(pose, self.theta_dis, self.fov, self.num_beams, self.theta_index_increment, self.sines, self.cosines, self.eps, self.orig_x, self.orig_y, self.map_height, self.map_width, self.map_resolution, self.dt, self.max_range)
        # self.local_track = self.local_map_generator.generate_line_local_map(np.copy(obs['scan']))
        
        return scan
    
    def ExtractLocalTrack(self, scan):
        feature = self.FeatureExtractor.generate_line_local_map(scan)
        scan_xs, scan_ys = self.map_data.pts2rc(feature)
        # feature = np.vstack([scan_xs, scan_ys])
        
        return feature
    
    def transform_points(self, points, transformation):
        """ Apply a transformation to the points (2D rotation + translation) """
        theta, tx, ty = transformation
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return points @ rotation_matrix.T + np.array([tx, ty])
    
    def icp_scan_matching(self, global_map, local_scan, max_iterations=100, tolerance=1e-7, initial_guesses=None):
        def compute_error(transformation, global_map, local_scan):
            """ Compute a weighted cost based on distance, translation, and rotation """
            transformed_scan = self.transform_points(local_scan, transformation)
            distances, _ = kdtree.query(transformed_scan)
            
            # Distance cost: sum of distances between closest points
            distance_cost = np.sum(distances)
            
            # Rotation cost: absolute value of the rotation angle
            rotation_cost = np.abs(transformation[0])  # theta
            
            # Translation cost: Euclidean distance of the translation vector
            translation_cost = np.sqrt(transformation[1]**2 + transformation[2]**2)  # tx, ty
            
            # Total cost as weighted sum of the components
            total_cost = self.w_dist * distance_cost + self.w_rot * rotation_cost + self.w_trans * translation_cost
            return total_cost

        # Create KDTree for fast nearest neighbor search
        kdtree = KDTree(global_map)

        best_transformation = None
        best_transformed_scan = None
        best_cost = np.inf
        all_costs = []
        
        # List to store cost for each particle
        particle_costs = []

        plt.scatter(global_map[:, 0], global_map[:, 1], c='black', label='Global Map', alpha=0.5, s=0.5)

        for particle in initial_guesses:
            # Use particle [theta, tx, ty] as the initial guess for optimization
            initial_guess = np.array([particle[2], particle[0], particle[1]])  # [theta, x, y]
            
            plt.scatter(local_scan[:, 0], local_scan[:, 1], c='red', label='Local Scan', alpha=0.5, s=0.5)
            initial_guess_transformed = self.transform_points(local_scan, initial_guess)
            plt.scatter(initial_guess_transformed[:, 0], initial_guess_transformed[:, 1], c='green', label='Initial Guess', alpha=0.5, s=0.5)
            
            # Callback function to store cost at each iteration
            cost_convergence = []
            def callback(transformation):
                current_cost = compute_error(transformation, global_map, local_scan)
                cost_convergence.append(current_cost)

            # Run optimization with the current particle as the initial guess
            result = minimize(
                compute_error, 
                initial_guess, 
                args=(global_map, local_scan), 
                method='L-BFGS-B',
                options={'maxiter': max_iterations, 'ftol': tolerance},
                callback=callback  # Track cost at each iteration
            )
            
            cost = result.fun
            particle_costs.append(cost)  # Store cost for this particle

            # Compare result cost to find the best one
            if cost < best_cost:
                best_cost = cost
                best_transformation = result.x
                best_transformed_scan = self.transform_points(local_scan, result.x)
        plt.show()
        # Return the best transformation, transformed scan, and costs for all particles
        return best_transformation, best_transformed_scan, np.array(particle_costs)


    def get_increment(self):
        return self.angle_increment

    def xy_2_rc(self, points):
        r, c = xy_2_rc_vec(points[:, 0], points[:, 1], self.orig_x, self.orig_y, self.map_resolution)
        return np.stack((c, r), axis=1)
    
 
# @njit(cache=True)
def interpolate_track_new(points, n_points=None, s=0):
    if len(points) <= 1:
        return points
    order_k = min(3, len(points) - 1)
    tck = interpolate.splprep([points[:, 0], points[:, 1]], k=order_k, s=s)[0]
    if n_points is None: n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T
    
    return track

# @njit(cache=True)
def resample_track_points(points, seperation_distance=0.2, smoothing=0.2):
    if points[0, 0] > points[-1, 0]:
        points = np.flip(points, axis=0)

    line_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    n_pts = max(int(line_length / seperation_distance), 2)
    smooth_line = interpolate_track_new(points, None, smoothing)
    resampled_points = interpolate_track_new(smooth_line, n_pts, 0)

    return resampled_points, smooth_line

@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, height, width, resolution):
    x_trans = x - orig_x
    y_trans = y - orig_y

    if x_trans < 0 or x_trans >= width * resolution or y_trans < 0 or y_trans >= height * resolution:
        c = -1
        r = -1
    else:
        c = int(x_trans/resolution)
        r = int(y_trans/resolution)


    return r, c

@njit(cache=True)
def xy_2_rc_vec(x, y, orig_x, orig_y, resolution):
    x_trans = x - orig_x
    y_trans = y - orig_y

    c = x_trans/resolution
    r = y_trans/resolution

    return r, c

@njit(cache=True)
def distance_transform(x, y, orig_x, orig_y, height, width, resolution, dt):
    r, c = xy_2_rc(x, y, orig_x, orig_y, height, width, resolution)
    distance = dt[r, c]
    return distance

@njit(cache=True)
def trace_ray(x, y, theta_index, sines, cosines, eps, orig_x, orig_y, height, width, resolution, dt, max_range):
    theta_index_ = int(theta_index)
    s = sines[theta_index_]
    c = cosines[theta_index_]

    dist_to_nearest = distance_transform(x, y, orig_x, orig_y, height, width, resolution, dt)
    total_dist = dist_to_nearest

    while dist_to_nearest > eps and total_dist <= max_range:
        x += dist_to_nearest * c
        y += dist_to_nearest * s

        dist_to_nearest = distance_transform(x, y, orig_x, orig_y, height, width, resolution, dt)
        total_dist += dist_to_nearest

    if total_dist > max_range:
        total_dist = max_range
    
    return total_dist

@njit(cache=True)
def get_scan(pose, theta_dis, fov, num_beams, theta_index_increment, sines, cosines, eps, orig_x, orig_y, height, width, resolution, dt, max_range):
    scan = np.empty((num_beams,))

    theta_index = theta_dis * (pose[2] - fov/2.)/(2. * np.pi)

    theta_index = np.fmod(theta_index, theta_dis)
    while (theta_index < 0):
        theta_index += theta_dis

    for i in range(0, num_beams):
        scan[i] = trace_ray(pose[0], pose[1], theta_index, sines, cosines, eps, orig_x, orig_y, height, width, resolution, dt, max_range)

        theta_index += theta_index_increment

        while theta_index >= theta_dis:
            theta_index -= theta_dis

    return scan