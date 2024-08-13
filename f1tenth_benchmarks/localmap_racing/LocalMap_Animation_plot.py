"""
The following code is used to plot the local map data that is used for mapless racing.
The algorithm predicts and extends the left and right boundaries of the track from local track data.
This data is stored in the the logs folder when the simulator is running and can be used to plot the boundaries afterwords.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from matplotlib.animation import FuncAnimation
import matplotlib
import math
matplotlib.use('TkAgg')  # or another suitable backend like 'Qt5Agg', 'Agg', etc.

#Plotting functions
# ------------------------------------------------------------------------------------------------------------------
# Custom sort function to sort filenames numerically
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[-1]) if numbers else float('inf')

def close_event(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)

def PrintDataArray(n):

    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

    # Print the arrays
    print("Local Track Data:")
    print(local_track)

    print("\nLeft Line Data:")
    print(left_line)

    print("\nRight Line Data:")
    print(right_line)

    print("\nBoundaries Data:")
    print(boundaries)

    print("\nBoundary Extension Data:")
    print(bound_extension)

def plot_once(n):
    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

    # Extract x and y coordinates
    left_x, left_y = left_line[:, 0], left_line[:, 1]
    right_x, right_y = right_line[:, 0], right_line[:, 1]
    
    fig, ax = plt.subplots()

    # Plot the coordinates
    ax.plot(left_x, left_y, label=f'Left Line ()', marker='o')
    ax.plot(right_x, right_y, label=f'Right Line ()', marker='o')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Left and Right Line Coordinates Over Time for n = ' + str(n))
    ax.legend()
    ax.grid(True)

    # Set fixed axis limits based on initial data range (adjust according to your data range)
    ax.set_xlim([-1, 17])  # Example limits, adjust according to your data range
    ax.set_ylim([-10, 10])  # Example limits, adjust according to your data range

    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()

def PlotAnimation():

    # Load the boundary files
    boundaries = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_*.npy"), key=numerical_sort)
    boundaries_ext = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_*.npy"), key=numerical_sort)

    # Preload the data
    Bound_data = [np.load(file) for file in boundaries]
    Bound_ext_data = [np.load(file) for file in boundaries_ext]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot initial data to set up the plot objects
    left_line, = ax.plot([], [], 'o-', label='Right bound')
    right_line, = ax.plot([], [], 'o-', label='Left bound')
    left_ext, = ax.plot([], [], 'o-', label='Right ext')
    right_ext, = ax.plot([], [], 'o-', label='Left ext')

    # Set labels, title, legend, and grid
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Left and Right Line Coordinates Over Time')
    ax.legend()
    ax.grid(True)

    # Set fixed axis limits based on initial data range (adjust according to your data range)
    ax.set_xlim(-10, 17)  # Example limits, adjust according to your data range
    ax.set_ylim(-10, 10)  # Example limits, adjust according to your data range

    # Define the update function
    def update(frame):
        # Update data in plot objects
        left_x, left_y = Bound_data[frame][:, 0], Bound_data[frame][:, 1]
        right_x, right_y = Bound_data[frame][:, 2], Bound_data[frame][:, 3]

        # Check if Bound_ext_data[frame] is not empty
        if Bound_ext_data[frame].size > 0:
            left_ext_x, left_ext_y = Bound_ext_data[frame][:, 0], Bound_ext_data[frame][:, 1]
            right_ext_x, right_ext_y = Bound_ext_data[frame][:, 2], Bound_ext_data[frame][:, 3]
        else:
            left_ext_x, left_ext_y = [], []
            right_ext_x, right_ext_y = [], []

        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)
        left_ext.set_data(left_ext_x, left_ext_y)
        right_ext.set_data(right_ext_x, right_ext_y)

        # Update the title to indicate the current frame
        ax.set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(Bound_data), repeat=False, interval=200)
    plt.show()



def plot_in_sequence():
    # Custom sort function to sort filenames numerically
    def numerical_sort(value):
        numbers = re.findall(r'\d+', value)
        return int(numbers[-1]) if numbers else float('inf')

    # Load the boundary files
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)

    # # Preload the data
    left_data = [np.load(file) for file in left_files]
    right_data = [np.load(file) for file in right_files]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # # Plot initial data to set up the plot objects
    left_line, = ax.plot([], [], 'o-', label='Left Line')
    right_line, = ax.plot([], [], 'o-', label='Right Line')

    # Set labels, title, legend, and grid
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Left and Right Line Coordinates Over Time')
    ax.legend()
    ax.grid(True)

    # Set fixed axis limits based on initial data range (adjust according to your data range)
    ax.set_xlim(-1, 17)  # Example limits, adjust according to your data range
    ax.set_ylim(-10, 10)  # Example limits, adjust according to your data range

    # Define the update function
    def update(frame):
        # Update data in plot objects
        left_x, left_y = left_data[frame][:, 0], left_data[frame][:, 1]
        right_x, right_y = right_data[frame][:, 0], right_data[frame][:, 1]
        
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)

        # Update the title to indicate the current frame
        ax.set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=400)

    plt.show()

#------------------------------------------------------------------------------------------------------------------

class FeatureExtraction:
    def __init__(self, path):
        #self.radii = np.array([])
        #self.centers = np.array([])
        self.radii = []
        self.centers = []
        
    def getCurvature(self):
        pass

    def calculate_circle_radius_and_center(points):
        if len(points) < 3:
            raise ValueError("At least three points are required to form a circle.")
        
        radii = []
        centers = []

        for i in range(len(points) - 2):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            x3, y3 = points[i + 2]

            # Calculate the determinant (related to twice the area of the triangle)
            A = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            
            if A == 0:
                raise ValueError("The points are collinear, so a unique circle cannot be formed.")

            # Calculate the squared lengths of the sides of the triangle
            a_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
            b_sq = (x3 - x2) ** 2 + (y3 - y2) ** 2
            c_sq = (x1 - x3) ** 2 + (y1 - y3) ** 2

            # Calculate the circumradius using the correct formula
            a = math.sqrt(a_sq)
            b = math.sqrt(b_sq)
            c = math.sqrt(c_sq)
            
            # Area of the triangle (using determinant method)
            area = abs(A) / 2

            # Circumradius
            radius = (a * b * c) / (4 * area)

            # Calculate the circumcenter coordinates
            x_center = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / (2 * A)
            y_center = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / (2 * A)

            radii.append(radius)
            centers.append((x_center, y_center))

        return np.array(radii), np.array(centers)

def plot_points_and_circles(points, radii, centers):
    fig, ax = plt.subplots()
    
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], color='blue', label='Points')
    
    # Plot the circles
    for radius, (x_center, y_center) in zip(radii, centers):
        circle = plt.Circle((x_center, y_center), radius, color='red', fill=False, label='Circle')
        ax.add_patch(circle)
    
    # Set equal scaling
    ax.set_aspect('equal', 'box')
    
    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    # Show plot
    plt.show()

def plot_curvature(curvature, points):
    fig, ax = plt.subplots()
    
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], color='blue', label='Points')
    
    # Plot the curvature
    ax.plot(curvature, color='red', label='Curvature')
    
    # Add labels and legend
    ax.set_xlabel('Index')
    ax.set_ylabel('Curvature')
    ax.legend()

    fig.canvas.mpl_connect('key_press_event', close_event)
    
    # Show plot
    plt.show()
    




def main():
    # PrintDataArray(53)
    # plot_once(53)
    PlotAnimation()

    points = np.array([
    [-1.13979656, -1.1504081],
    [-0.88752856, -1.08038959],
    [-0.64219188, -1.05320218],
    [-0.38959808, -1.0280293],
    [-0.13795937, -0.96584606],
    [0.11108701, -0.96568564]
    ])

    # n =200
    n = 400
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")

    radii, centers = FeatureExtraction.calculate_circle_radius_and_center(right_line)
    curvature = 1 / radii
    #print("Radii:", radii)
    #print("Centers:", centers)
    print("Curvature:", len(curvature))
    print("Right Line:", len(right_line))

    plot_in_sequence()
    #plot_once(n)
    #plot_points_and_circles(right_line, radii, centers)
    # plot_curvature(curvature, right_line)

if __name__ == '__main__':
     main()