import numpy as np
import matplotlib.pyplot as plt

def closest_point_on_track(vehicle_x, vehicle_y, centerline_x, centerline_y):
    distances = np.hypot(centerline_x - vehicle_x, centerline_y - vehicle_y)
    closest_idx = np.argmin(distances)
    return closest_idx, centerline_x[closest_idx], centerline_y[closest_idx]

def frenet_coordinates(vehicle_x, vehicle_y, vehicle_theta, centerline_x, centerline_y):
    idx, closest_x, closest_y = closest_point_on_track(vehicle_x, vehicle_y, centerline_x, centerline_y)
    
    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    ds = np.hypot(dx, dy)
    s = np.sum(ds[:idx])
    
    centerline_dx = np.gradient(centerline_x)
    centerline_dy = np.gradient(centerline_y)
    norm = np.hypot(centerline_dx[idx], centerline_dy[idx])
    track_tangent = np.array([centerline_dx[idx], centerline_dy[idx]]) / norm
    vehicle_vector = np.array([vehicle_x - closest_x, vehicle_y - closest_y])
    d = np.cross(track_tangent, vehicle_vector)
    
    track_heading = np.arctan2(centerline_dy[idx], centerline_dx[idx])
    d_theta = vehicle_theta - track_heading
    d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))
    
    return s, d, d_theta, (closest_x, closest_y)

def visualize_frenet(vehicle_x, vehicle_y, vehicle_theta, centerline_x, centerline_y, s, d, d_theta, closest_point):
    plt.figure(figsize=(10, 6))
    plt.plot(centerline_x, centerline_y, label="Centerline", color="blue", linewidth=2)
    plt.plot(vehicle_x, vehicle_y, 'ro', label="Vehicle Position")
    
    closest_x, closest_y = closest_point
    plt.plot(closest_x, closest_y, 'go', label="Closest Point on Track")
    
    # Draw the vehicle's orientation
    arrow_length = 0.5
    plt.arrow(vehicle_x, vehicle_y, arrow_length * np.cos(vehicle_theta), arrow_length * np.sin(vehicle_theta), 
              head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    # Draw perpendicular distance d
    plt.plot([vehicle_x, closest_x], [vehicle_y, closest_y], 'k--', label=f"Perpendicular Distance d = {d:.2f}")
    
    plt.title("Frenet Coordinate Frame Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example centerline and vehicle pose
centerline_x = np.array([0, 1, 2, 3, 4, 5])
centerline_y = np.array([0, 0, 0, 1, 1, 1])
vehicle_x, vehicle_y, vehicle_theta = 2.5, 0.5, np.pi / 4

# Compute Frenet coordinates
s, d, d_theta, closest_point = frenet_coordinates(vehicle_x, vehicle_y, vehicle_theta, centerline_x, centerline_y)

# Visualize the results
visualize_frenet(vehicle_x, vehicle_y, vehicle_theta, centerline_x, centerline_y, s, d, d_theta, closest_point)
