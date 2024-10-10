import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

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

def cartesian_coordinates(s, d, centerline_x, centerline_y):
    idx = np.searchsorted(np.cumsum(np.hypot(np.diff(centerline_x), np.diff(centerline_y))), s)
    
    centerline_dx = np.gradient(centerline_x)
    centerline_dy = np.gradient(centerline_y)
    norm = np.hypot(centerline_dx[idx], centerline_dy[idx])
    track_tangent = np.array([centerline_dx[idx], centerline_dy[idx]]) / norm
    
    closest_x, closest_y = centerline_x[idx], centerline_y[idx]
    point_x = closest_x + d * -track_tangent[1]
    point_y = closest_y + d * track_tangent[0]
    
    return point_x, point_y

def visualize_frenet(vehicle_x, vehicle_y, vehicle_theta, centerline_x, centerline_y, s, d, d_theta, closest_point):
    plt.figure(figsize=(10, 6))
    
    plt.plot(centerline_x, centerline_y, label="Centerline", color="blue", linewidth=2)
    plt.plot(vehicle_x, vehicle_y, 'ro', label="Vehicle Position")
    
    closest_x, closest_y = closest_point
    plt.plot(closest_x, closest_y, 'go', label="Closest Point on Track")
    
    arrow_length = 0.5
    plt.arrow(vehicle_x, vehicle_y, arrow_length * np.cos(vehicle_theta), arrow_length * np.sin(vehicle_theta), 
              head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    plt.plot([vehicle_x, closest_x], [vehicle_y, closest_y], 'k--', label=f"Perpendicular Distance d = {d:.2f}")
    
    plt.text(vehicle_x, vehicle_y, f'({vehicle_x:.2f}, {vehicle_y:.2f})\n(s, d) = ({s:.2f}, {d:.2f})', fontsize=9)
    
    point_x, point_y = cartesian_coordinates(s, d, centerline_x, centerline_y)
    plt.text(point_x, point_y, f'Converted Back:\n({point_x:.2f}, {point_y:.2f})', fontsize=9, color='purple')
    
    plt.title("Frenet Coordinate Frame Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()

# Define the centerline as a curve
t = np.linspace(0, 2 * np.pi, 100)
centerline_x = 5 * np.cos(t)
centerline_y = 5 * np.sin(t)

# Place the vehicle off the centerline
vehicle_x, vehicle_y, vehicle_theta = 3.5, 1.5, np.pi / 3

# Calculate Frenet coordinates
s, d, d_theta, closest_point = frenet_coordinates(vehicle_x, vehicle_y, vehicle_theta, centerline_x, centerline_y)

# Visualize the results
visualize_frenet(vehicle_x, vehicle_y, vehicle_theta, centerline_x, centerline_y, s, d, d_theta, closest_point)
