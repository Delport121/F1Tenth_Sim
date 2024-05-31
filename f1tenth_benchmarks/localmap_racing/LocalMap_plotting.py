import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # or another suitable backend like 'Qt5Agg', 'Agg', etc.

#Plot one set
# ------------------------------------------------------------------------------------------------------------------
def plot_once(n):
    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
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

def plot_all():
    
     # Custom sort function to sort filenames numerically
    def numerical_sort(value):
        numbers = re.findall(r'\d+', value)
        return int(numbers[-1]) if numbers else float('inf')

    # Get a list of all left and right line files
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)
    track_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_*.npy"), key=numerical_sort)

    # Plot the lines continuously
    plt.figure(figsize=(12, 8))

    for left_file, right_file in zip(left_files, right_files):
        # Load the numpy arrays
        left_line = np.load(left_file)
        right_line = np.load(right_file)
        #local_track = np.load(track_files)

        # Extract x and y coordinates
        left_x, left_y = left_line[:, 0], left_line[:, 1]
        right_x, right_y = right_line[:, 0], right_line[:, 1]
        #local_track_x, local_track_y = local_track[:, 0], local_track[:, 1]

        # Plot the coordinates
        plt.plot(left_x, left_y, label=f'Left Line ({os.path.basename(left_file)})', marker='o')
        plt.plot(right_x, right_y, label=f'Right Line ({os.path.basename(right_file)})', marker='o')
        #plt.plot(local_track_x, local_track_y, label=f'Local Track ({os.path.basename(track_files)})', marker='o')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Left and Right Line Coordinates Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()



#------------------------------------------------------------------------------------------------------------------
#Boundary extension start at 53
n = 53

local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

def close_event(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)

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

# plot_once(13)
# plot_once(14)
# plot_once(15)
# #plot_all()
# plot_in_sequence()

# Custom sort function to sort filenames numerically
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[-1]) if numbers else float('inf')

# Load the boundary files
boundaries = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_*.npy"), key=numerical_sort)
boundaries_ext = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_*.npy"), key=numerical_sort)

# Preload the data
Bound_data = [np.load(file) for file in boundaries]
Bound_ext_data = [np.load(file) for file in boundaries_ext]

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Plot initial data to set up the plot objects
left_line, = ax.plot([], [], 'o-', label='Left bound')
right_line, = ax.plot([], [], 'o-', label='Right bound')
left_ext, = ax.plot([], [], 'o-', label='Left ext')
right_ext, = ax.plot([], [], 'o-', label='Right ext')

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
ani = FuncAnimation(fig, update, frames=len(Bound_data), repeat=False, interval=100)

plt.show()