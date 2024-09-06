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
from scipy.optimize import curve_fit
from trajectory_planning_helpers.calc_head_curv_num import calc_head_curv_num
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

def plot_lines_once(n):
    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

    try:
        scan_dir = f"/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_0.npy"
        scans = np.load(scan_dir)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)

    scan_xs, scan_ys = scans[n+1] * np.array([coses, sines])


    # Extract x and y coordinates
    left_x, left_y = left_line[:, 0], left_line[:, 1]
    right_x, right_y = right_line[:, 0], right_line[:, 1]
    
    fig, ax = plt.subplots()

    # Plot the coordinates
    ax.plot(local_track[:, 0], local_track[:, 1], label=f'Local track ()', marker='o')
    ax.plot(left_x, left_y, label=f'Left Line ()', marker='o')
    ax.plot(right_x, right_y, label=f'Right Line ()', marker='o')
    ax.plot(scan_xs, scan_ys, label=f'Scan Data', marker='o')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Left and Right Line Coordinates Over Time for n = ' + str(n))
    ax.legend()
    ax.grid(True)

    # Set fixed axis limits based on initial data range (adjust according to your data range)
    ax.set_xlim([-1, 20])  # Example limits, adjust according to your data range
    ax.set_ylim([-20, 10])  # Example limits, adjust according to your data range

    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()

def func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def plot_Polyfit(n):
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    
    # Extract x and y coordinates
    left_x, left_y = left_line[:, 0], left_line[:, 1]
    right_x, right_y = right_line[:, 0], right_line[:, 1]
    
    poptL, pcovL = curve_fit(func, left_x, left_y)
    poptR, pcovR = curve_fit(func, right_x, right_y)
    
    fig, ax = plt.subplots()

    # Plot the coordinates
    ax.plot(left_x, left_y, label=f'Left Line ()', marker='o')
    ax.plot(right_x, right_y, label=f'Right Line ()', marker='o')
    ax.plot(left_x, func(left_x, *poptL), 'r-', 
            label='Lfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptL))
    ax.plot(right_x, func(right_x, *poptR), 'r-', 
            label='Rfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptR))

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Left and Right Line Coordinates Over Time for n = ' + str(n))
    ax.legend()
    ax.grid(True)

    # Set fixed axis limits based on initial data range (adjust according to your data range)
    ax.set_xlim([-1, 20])  # Example limits, adjust according to your data range
    ax.set_ylim([-20, 10])  # Example limits, adjust according to your data range

    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()
    
def plot_lines_and_curvature(n):
    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

    try:
        scan_dir = f"/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_0.npy"
        scans = np.load(scan_dir)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)
    
    # Extract x and y coordinates
    scan_xs, scan_ys = scans[n+1] * np.array([coses, sines])
    local_x, local_y = local_track[:, 0], local_track[:, 1]
    left_x, left_y = left_line[:, 0], left_line[:, 1]
    right_x, right_y = right_line[:, 0], right_line[:, 1]
    
    LocalLine = local_track[:, 0:2]
    
    # Ca;culate the curvature using cirlce through three points method
    radiiLocal, centersLocal, curvatureLocal = FeatureExtraction.calculate_circle_radius_and_center(LocalLine)
    radiiR, centersR, curvatureR = FeatureExtraction.calculate_circle_radius_and_center(right_line)
    radiiL, centersL, curvatureL = FeatureExtraction.calculate_circle_radius_and_center(left_line)

    #Calculate the curvature using the calc_head_curv_num function
    # Calculate element lengths (distances between consecutive points)
    pathL = np.vstack((left_x, left_y)).T
    pathR = np.vstack((right_x, right_y)).T
    el_lengthsLocal = np.sqrt(np.sum(np.diff(local_track, axis=0)**2, axis=1))
    el_lengthsL = np.sqrt(np.sum(np.diff(pathL, axis=0)**2, axis=1))
    el_lengthsR = np.sqrt(np.sum(np.diff(pathR, axis=0)**2, axis=1))
    # print(el_lengthsLocal)

    # Call the calc_head_curv_num function
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
    psiL, kappaL = calc_head_curv_num(
        path=pathL,
        el_lengths=el_lengthsL,
        is_closed=False,
        stepsize_psi_preview=0.1,
        stepsize_psi_review=0.1,
        stepsize_curv_preview=0.2,
        stepsize_curv_review=0.2,
        calc_curv=True
    )
    # Call the calc_head_curv_num function
    psiR, kappaR = calc_head_curv_num(
        path=pathR,
        el_lengths=el_lengthsR,
        is_closed=False,
        stepsize_psi_preview=0.1,
        stepsize_psi_review=0.1,
        stepsize_curv_preview=0.2,
        stepsize_curv_review=0.2,
        calc_curv=True
    )
    
    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot the coordinates
    axs[0].plot(left_x, left_y, label=f'Left Line ()', marker='o')
    axs[0].plot(right_x, right_y, label=f'Right Line ()', marker='o')
    axs[0].plot(local_x, local_y, label=f'Local Line ()', marker='o')
    # axs[0].plot(scan_xs, scan_ys, label=f'Scan Data', marker='o')
    axs[0].set_title('Left and Right Line Coordinates Over Time for n = ' + str(n))
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')

    # # Plot the heading (psi)
    # axs[1].plot(np.arange(len(psi)), psi, label='Heading (psi)')
    # axs[1].set_title('Heading (psi) along the Path')
    # axs[1].set_xlabel('Point Index')
    # axs[1].set_ylabel('Heading (radians)')
    # axs[1].legend()
    
    # Plot the curvature using the circle through three points method
    axs[1].plot(curvatureL, label='Left line Curvature')
    axs[1].plot(curvatureR, label='Right line Curvature')
    axs[1].plot(curvatureLocal, label='Local line Curvature')
    axs[1].set_title('Curvature using circle through three points method')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Curvature')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylim([-2, 2])

    # Plot the curvature (kappa)
    axs[2].plot(np.arange(len(kappaL)), kappaL, label='Curvature (kappa) left')
    axs[2].plot(np.arange(len(kappaR)), kappaR, label='Curvature (kappa) right')
    axs[2].plot(np.arange(len(kappaLocal)), kappaLocal, label='Curvature (kappa) Local')
    axs[2].set_title('Curvature (kappa) along the Path using calc_head_curv_num')
    axs[2].set_xlabel('Point Index')
    axs[2].set_ylabel('Curvature (1/m)')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()
    
def plot_Poly_and_curvature(n):
    local_track = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_"+ str(n) +".npy")
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    left_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_"+ str(n) +".npy")
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")

    try:
        scan_dir = f"/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_0.npy"
        scans = np.load(scan_dir)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)
    
    # Extract x and y coordinates
    scan_xs, scan_ys = scans[n+1] * np.array([coses, sines])
    left_x, left_y = left_line[:, 0], left_line[:, 1]
    right_x, right_y = right_line[:, 0], right_line[:, 1]
    local_x, local_y = local_track[:, 0], local_track[:, 1]
    
    poptL, pcovL = curve_fit(func, left_x, left_y)
    poptR, pcovR = curve_fit(func, right_x, right_y)
    poptLocal, pcovLocal = curve_fit(func, local_x, local_y)
    left_y_new = func(left_x, *poptL)
    right_y_new = func(right_x, *poptR)
    local_y_new = func(local_x, *poptLocal)
    


    #Calculate the curvature using the calc_head_curv_num function
    # Calculate element lengths (distances between consecutive points)
    pathL = np.vstack((left_x, left_y_new)).T
    pathR = np.vstack((right_x, right_y_new)).T
    pathLocal = np.vstack((local_x, local_y_new)).T
    el_lengthsL = np.sqrt(np.sum(np.diff(pathL, axis=0)**2, axis=1))
    el_lengthsR = np.sqrt(np.sum(np.diff(pathR, axis=0)**2, axis=1))
    el_lengthsLocal = np.sqrt(np.sum(np.diff(pathLocal, axis=0)**2, axis=1))
    
    # Ca;culate the curvature using cirlce through three points method
    radiiR, centersR, curvatureR = FeatureExtraction.calculate_circle_radius_and_center(pathR)
    radiiL, centersL, curvatureL = FeatureExtraction.calculate_circle_radius_and_center(pathL)
    radiiLocal, centersLocal, curvatureLocal = FeatureExtraction.calculate_circle_radius_and_center(pathLocal)

    # Call the calc_head_curv_num function
    psi, kappaL = calc_head_curv_num(
        path=pathL,
        el_lengths=el_lengthsL,
        is_closed=False,
        stepsize_psi_preview=0.1,
        stepsize_psi_review=0.1,
        stepsize_curv_preview=0.2,
        stepsize_curv_review=0.2,
        calc_curv=True
    )
    psi, kappaR = calc_head_curv_num(
        path=pathR,
        el_lengths=el_lengthsR,
        is_closed=False,
        stepsize_psi_preview=0.1,
        stepsize_psi_review=0.1,
        stepsize_curv_preview=0.2,
        stepsize_curv_review=0.2,
        calc_curv=True
    )
    psiLocal, kappaLocal = calc_head_curv_num(
        path=pathLocal,
        el_lengths=el_lengthsLocal,
        is_closed=False,
        stepsize_psi_preview=0.1,
        stepsize_psi_review=0.1,
        stepsize_curv_preview=0.2,
        stepsize_curv_review=0.2,
        calc_curv=True
    )
    
    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot the coordinates
    axs[0].plot(left_x, left_y, label=f'Left Line ()', marker='o')
    axs[0].plot(right_x, right_y, label=f'Right Line ()', marker='o')
    axs[0].plot(local_x, local_y, label=f'Local Line ()', marker='o')
    axs[0].plot(left_x, left_y_new, 'r-', 
            label='Lfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptL))
    axs[0].plot(right_x, right_y_new, 'r-', 
            label='Rfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptR))
    axs[0].plot(local_x, local_y_new, 'r-', 
            label='Rfit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(poptR))
    # axs[0].plot(scan_xs, scan_ys, label=f'Scan Data', marker='o')
    axs[0].set_title('Left and Right Line Coordinates Over Time for n = ' + str(n))
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')

    # # Plot the heading (psi)
    # axs[1].plot(np.arange(len(psi)), psi, label='Heading (psi)')
    # axs[1].set_title('Heading (psi) along the Path')
    # axs[1].set_xlabel('Point Index')
    # axs[1].set_ylabel('Heading (radians)')
    # axs[1].legend()
    
    # Plot the curvature using the circle through three points method
    axs[1].plot(curvatureL, label='Left line Curvature')
    axs[1].plot(curvatureR, label='Right line Curvature')
    axs[1].plot(curvatureLocal, label='Local line Curvature')
    axs[1].set_title('Curvature using circle through three points method')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Curvature')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylim([-1, 1])

    # Plot the curvature (kappa)
    axs[2].plot(np.arange(len(kappaL)), kappaL, label='Curvature (kappa) left')
    axs[2].plot(np.arange(len(kappaR)), kappaR, label='Curvature (kappa) right')
    axs[2].plot(np.arange(len(kappaLocal)), kappaLocal, label='Curvature (kappa) Local')
    axs[2].set_title('Curvature (kappa) along the Path using calc_head_curv_num')
    axs[2].set_xlabel('Point Index')
    axs[2].set_ylabel('Curvature (1/m)')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_ylim([-1, 1])

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()
    
def plot_boundaries_once(n):
    boundaries = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundaries_"+ str(n) +".npy")
    bound_extension = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/boundExtension_"+ str(n) +".npy")
    pass

def plot_boundaries_animation():

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
        right_x, right_y = Bound_data[frame][:, 0], Bound_data[frame][:, 1]
        left_x, left_y = Bound_data[frame][:, 2], Bound_data[frame][:, 3]

        # Check if Bound_ext_data[frame] is not empty
        if Bound_ext_data[frame].size > 0:
            right_ext_x, right_ext_y = Bound_ext_data[frame][:, 0], Bound_ext_data[frame][:, 1]
            left_ext_x, left_ext_y = Bound_ext_data[frame][:, 2], Bound_ext_data[frame][:, 3]
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

def plot_lines_animation():
  
    # Load the boundary files
    local_track_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)

    # # Preload the data
    local_data = [np.load(file) for file in local_track_files]
    left_data = [np.load(file) for file in left_files]
    right_data = [np.load(file) for file in right_files]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # # Plot initial data to set up the plot objects
    local_line, = ax.plot([], [], 'o-', label='Local Track')
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
        local_x, local_y = local_data[frame][:, 0], local_data[frame][:, 1]
        left_x, left_y = left_data[frame][:, 0], left_data[frame][:, 1]
        right_x, right_y = right_data[frame][:, 0], right_data[frame][:, 1]
        
        local_line.set_data(local_x, local_y)
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)

        # Update the title to indicate the current frame
        ax.set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=50)

    plt.show()

def plot_lines_animation_with_polyfit():
    # Load the boundary files
    local_track_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)

    # Preload the data
    local_data = [np.load(file) for file in local_track_files]
    left_data = [np.load(file) for file in left_files]
    right_data = [np.load(file) for file in right_files]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot initial data to set up the plot objects
    left_line, = ax.plot([], [], 'o-', label='Left Line')
    right_line, = ax.plot([], [], 'o-', label='Right Line')
    local_line, = ax.plot([], [], 'o-', label='Local Track')
    left_poly, = ax.plot([], [], 'r-', label='Left Polyfit')
    right_poly, = ax.plot([], [], 'g-', label='Right Polyfit')
    local_poly, = ax.plot([], [], 'b-', label='Local Polyfit')

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
        local_x, local_y = local_data[frame][:, 0], local_data[frame][:, 1]
        
        # Update line data
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)
        local_line.set_data(local_x, local_y)

        # Fit polynomials
        try:
            poptL, _ = curve_fit(func, left_x, left_y)
            poptR, _ = curve_fit(func, right_x, right_y)
            poptLocal, _ = curve_fit(func, local_x, local_y)
            left_poly.set_data(left_x, func(left_x, *poptL))
            right_poly.set_data(right_x, func(right_x, *poptR))
            local_poly.set_data(local_x, func(local_x, *poptLocal))
        except Exception as e:
            print(f"Error fitting polynomials: {e}")
            left_poly.set_data([], [])
            right_poly.set_data([], [])

        # Update the title to indicate the current frame
        ax.set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=50)

    plt.show()

def plot_lines_and_curvature_animation():
    
     # Load the boundary files
    local_track_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)
    try:
        scan_files = sorted(glob.glob("/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_*.npy"), key=numerical_sort)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)

    # # Preload the data
    local_track_data = [np.load(file) for file in local_track_files]
    left_data = [np.load(file) for file in left_files]
    right_data = [np.load(file) for file in right_files]
    scan_data = [np.load(file) for file in scan_files]

    # Create the figure and axes
    # fig, ax = plt.subplots(figsize=(12, 8))
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # # Plot initial data to set up the plot objects
    left_line, = axs[0].plot([], [], 'o-', label='Left Line')
    right_line, = axs[0].plot([], [], 'o-', label='Right Line')
    local_track_line, = axs[0].plot([], [], 'o-', label='Local Track')
    ThreeP_curveL, = axs[1].plot([],[], label='Curvature Left')
    ThreeP_curveR, = axs[1].plot([],[], label='Curvature Right')
    ThreeP_curveLocal, = axs[1].plot([],[], label='Curvature Local')
    Kappa_curveL, = axs[2].plot([],[], label='Curvature (kappa) Left')
    Kappa_curveR, = axs[2].plot([], [],label='Curvature (kappa) Right')
    Kappa_curveLocal, = axs[2].plot([], [],label='Curvature (kappa) Local')
    
     # Plot the coordinates
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].legend()
    axs[0].grid(True)
     # Set fixed axis limits based on initial data range (adjust according to your data range)
    axs[0].set_xlim([-1, 20])  # Example limits, adjust according to your data range
    axs[0].set_ylim([-10, 10])  # Example limits, adjust according to your data range
    # axs[0].axis('equal')

    # Plot the curvature using the circle through three points method
    axs[1].set_title('Curvature using circle through three points method')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Curvature')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylim([-1, 1])
    axs[1].set_xlim([0, 50])  # Example limits, adjust according to your data range

    # Plot the curvature (kappa)
    axs[2].set_title('Curvature (kappa) along the Path using calc_head_curv_num')
    axs[2].set_xlabel('Point Index')
    axs[2].set_ylabel('Curvature (1/m)')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_ylim([-1, 1])
    axs[2].set_xlim([0, 50])  # Example limits, adjust according to your data range

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)

    # Define the update function
    def update(frame):
        # Update data in plot objects
        left_x, left_y = left_data[frame][:, 0], left_data[frame][:, 1]
        right_x, right_y = right_data[frame][:, 0], right_data[frame][:, 1]
        local_x, local_y = local_track_data[frame][:, 0], local_track_data[frame][:, 1]
        pathL = np.vstack((left_x, left_y)).T
        pathR = np.vstack((right_x, right_y)).T
        pathLocal = np.vstack((local_x, local_y)).T
        # LocalLine = local_track_data[frame][:, 0:2]
        
        # Calculate the curvature using cirlce through three points method
        radiiR, centersR, curvatureR = FeatureExtraction.calculate_circle_radius_and_center(pathR)
        radiiL, centersL, curvatureL = FeatureExtraction.calculate_circle_radius_and_center(pathL)
        radiiLocal, centersLocal, curvatureLocal = FeatureExtraction.calculate_circle_radius_and_center(pathLocal)

        #Calculate the curvature using the calc_head_curv_num function
        # Calculate element lengths (distances between consecutive points)
        el_lengthsL = np.sqrt(np.sum(np.diff(pathL, axis=0)**2, axis=1))
        el_lengthsR = np.sqrt(np.sum(np.diff(pathR, axis=0)**2, axis=1))
        el_lengthsLocal = np.sqrt(np.sum(np.diff(pathLocal, axis=0)**2, axis=1))

        # Call the calc_head_curv_num function
        psi, kappaL = calc_head_curv_num(
            path=pathL,
            el_lengths=el_lengthsL,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )
        psi, kappaR = calc_head_curv_num(
            path=pathR,
            el_lengths=el_lengthsR,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )
        psiLocal, kappaLocal = calc_head_curv_num(
            path=pathLocal,
            el_lengths=el_lengthsLocal,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )      
        
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)
        local_track_line.set_data(local_x, local_y)
        ThreeP_curveL.set_data(np.arange(len(curvatureL)), curvatureL)
        ThreeP_curveR.set_data(np.arange(len(curvatureR)),curvatureR)
        ThreeP_curveLocal.set_data(np.arange(len(curvatureLocal)),curvatureLocal)
        Kappa_curveL.set_data(np.arange(len(kappaL)), kappaL)
        Kappa_curveR.set_data(np.arange(len(kappaR)), kappaR)
        Kappa_curveLocal.set_data(np.arange(len(kappaLocal)), kappaLocal)

        # Update the title to indicate the current frame
        axs[0].set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=100)

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()

def plot_Poly_and_curvature_animation():
    
     # Load the boundary files
    local_track_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/local_map_*.npy"), key=numerical_sort)
    right_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_*.npy"), key=numerical_sort)
    left_files = sorted(glob.glob("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line2_*.npy"), key=numerical_sort)
    try:
        scan_files = sorted(glob.glob("/home/ruan/Documents/f1tenth_benchmarks/Logs/LocalMPCC/RawData_mu60/ScanLog_gbr_*.npy"), key=numerical_sort)
        print("Scan data found")
    except:
        print("No scan data found")

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)

    # # Preload the data
    local_track_data = [np.load(file) for file in local_track_files]
    left_data = [np.load(file) for file in left_files]
    right_data = [np.load(file) for file in right_files]
    scan_data = [np.load(file) for file in scan_files]

    # Create the figure and axes
    # fig, ax = plt.subplots(figsize=(12, 8))
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # # Plot initial data to set up the plot objects
    left_line, = axs[0].plot([], [], 'o-', label='Left Line')
    right_line, = axs[0].plot([], [], 'o-', label='Right Line')
    local_track_line, = axs[0].plot([], [], 'o-', label='Local Track')
    left_poly, = axs[0].plot([], [], 'r-', label='Left Polyfit')
    right_poly, = axs[0].plot([], [], 'g-', label='Right Polyfit')
    local_poly, = axs[0].plot([], [], 'b-', label='Local Polyfit')
    ThreeP_curveL, = axs[1].plot([],[], label='Curvature Left')
    ThreeP_curveR, = axs[1].plot([],[], label='Curvature Right')
    ThreeP_curveLocal, = axs[1].plot([],[], label='Curvature Local')
    Kappa_curveL, = axs[2].plot([],[], label='Curvature (kappa) Left')
    Kappa_curveR, = axs[2].plot([], [],label='Curvature (kappa) Right')
    Kappa_curveLocal, = axs[2].plot([], [],label='Curvature (kappa) Local')
    
     # Plot the coordinates
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].legend()
    axs[0].grid(True)
     # Set fixed axis limits based on initial data range (adjust according to your data range)
    axs[0].set_xlim([-1, 20])  # Example limits, adjust according to your data range
    axs[0].set_ylim([-10, 10])  # Example limits, adjust according to your data range
    # axs[0].axis('equal')

    # Plot the curvature using the circle through three points method
    axs[1].set_title('Curvature using circle through three points method')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Curvature')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylim([-1, 1])
    axs[1].set_xlim([0, 50])  # Example limits, adjust according to your data range

    # Plot the curvature (kappa)
    axs[2].set_title('Curvature (kappa) along the Path using calc_head_curv_num')
    axs[2].set_xlabel('Point Index')
    axs[2].set_ylabel('Curvature (1/m)')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_ylim([-1, 1])
    axs[2].set_xlim([0, 50])  # Example limits, adjust according to your data range

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)

    # Define the update function
    def update(frame):
        # Update data in plot objects
        left_x, left_y = left_data[frame][:, 0], left_data[frame][:, 1]
        right_x, right_y = right_data[frame][:, 0], right_data[frame][:, 1]
        local_x, local_y = local_track_data[frame][:, 0], local_track_data[frame][:, 1]
        
         # Fit polynomials
        try:
            poptL, _ = curve_fit(func, left_x, left_y)
            poptR, _ = curve_fit(func, right_x, right_y)
            poptLocal, _ = curve_fit(func, local_x, local_y)
            left_y_new = func(left_x, *poptL)
            right_y_new = func(right_x, *poptR)
            local_y_new = func(local_x, *poptLocal)
            left_poly.set_data(left_x, left_y_new)
            right_poly.set_data(right_x, right_y_new)
            local_poly.set_data(local_x, local_y_new )
        except Exception as e:
            print(f"Error fitting polynomials: {e}")
            left_poly.set_data([], [])
            right_poly.set_data([], [])
            
        pathL = np.vstack((left_x, left_y_new)).T
        pathR = np.vstack((right_x, right_y_new)).T
        pathLocal = np.vstack((local_x, local_y_new)).T
        # LocalLine = local_track_data[frame][:, 0:2]
        
       
        
        # Calculate the curvature using cirlce through three points method
        radiiR, centersR, curvatureR = FeatureExtraction.calculate_circle_radius_and_center(pathR)
        radiiL, centersL, curvatureL = FeatureExtraction.calculate_circle_radius_and_center(pathL)
        radiiLocal, centersLocal, curvatureLocal = FeatureExtraction.calculate_circle_radius_and_center(pathLocal)

        #Calculate the curvature using the calc_head_curv_num function
        # Calculate element lengths (distances between consecutive points)
        el_lengthsL = np.sqrt(np.sum(np.diff(pathL, axis=0)**2, axis=1))
        el_lengthsR = np.sqrt(np.sum(np.diff(pathR, axis=0)**2, axis=1))
        el_lengthsLocal = np.sqrt(np.sum(np.diff(pathLocal, axis=0)**2, axis=1))

        # Call the calc_head_curv_num function
        psi, kappaL = calc_head_curv_num(
            path=pathL,
            el_lengths=el_lengthsL,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )
        psi, kappaR = calc_head_curv_num(
            path=pathR,
            el_lengths=el_lengthsR,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )
        psiLocal, kappaLocal = calc_head_curv_num(
            path=pathLocal,
            el_lengths=el_lengthsLocal,
            is_closed=False,
            stepsize_psi_preview=0.1,
            stepsize_psi_review=0.1,
            stepsize_curv_preview=0.2,
            stepsize_curv_review=0.2,
            calc_curv=True
        )      
        
        left_line.set_data(left_x, left_y)
        right_line.set_data(right_x, right_y)
        local_track_line.set_data(local_x, local_y)
        ThreeP_curveL.set_data(np.arange(len(curvatureL)), curvatureL)
        ThreeP_curveR.set_data(np.arange(len(curvatureR)),curvatureR)
        ThreeP_curveLocal.set_data(np.arange(len(curvatureLocal)),curvatureLocal)
        Kappa_curveL.set_data(np.arange(len(kappaL)), kappaL)
        Kappa_curveR.set_data(np.arange(len(kappaR)), kappaR)
        Kappa_curveLocal.set_data(np.arange(len(kappaLocal)), kappaLocal)
        
        # Update the title to indicate the current frame
        axs[0].set_title(f'Left and Right Line Coordinates (Frame {frame})')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(left_data), repeat=False, interval=100)

    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', close_event)
    plt.show()  
   
def plot_points_and_circles_together(n):
    right_line_data = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_" + str(n) + ".npy")
    radii, centers, curvature = FeatureExtraction.calculate_circle_radius_and_center(right_line_data)

    right_x, right_y = right_line_data[:, 0], right_line_data[:, 1]
    print(right_x)
    print(right_y)
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot initial data to set up the plot objects
    right_line_plot, = ax.plot([], [], 'o-', label='Right Line')

    # Set labels, title, legend, and grid
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Right Line Coordinates and Curvature Over Time')
    ax.legend()
    ax.grid(True)

    ax.set_xlim(-15, 15)  # Example limits, adjust according to your data range
    ax.set_ylim(-15, 15)
    # Set equal scaling
    # ax.set_aspect('equal', 'box')

    # Define the update function
    def update(frame):
        # Update data in plot objects
        right_line_plot.set_data(right_x, right_y)

        # Update the title to indicate the current frame
        ax.set_title(f'Right Line Coordinates (Frame {frame})')

        radius, (x_center, y_center) = radii[frame], centers[frame]

        circle = plt.Circle((x_center, y_center), radius, color='red', fill=False, label='Circle')
        ax.add_patch(circle)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(curvature), repeat=False, interval=200)
    fig.canvas.mpl_connect('key_press_event', close_event)

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
        curvatures = []

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

            # Determine the sign of curvature based on the orientation of the triangle
            # Sign of the area A will determine the direction of the turn:
            # A > 0: counterclockwise (positive curvature)
            # A < 0: clockwise (negative curvature)
            curvature = 1.0 / radius
            curvature = curvature if A > 0 else -curvature

            radii.append(radius)
            centers.append((x_center, y_center))
            curvatures.append(curvature)

        return np.array(radii), np.array(centers), np.array(curvatures)

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
    

def main():
    
    points = np.array([
    [-1.13979656, -1.1504081],
    [-0.88752856, -1.08038959],
    [-0.64219188, -1.05320218],
    [-0.38959808, -1.0280293],
    [-0.13795937, -0.96584606],
    [0.11108701, -0.96568564]
    ])

    # n = 200
    # n = 400
    # n = 85
    # n = 240
    # n = 415
    # n = 395 # Very noisy
    # n = 140
    n = 600
    # n = 40
    right_line = np.load("Logs/LocalMPCC/RawData_mu60/LocalMapData_mu60/line1_"+ str(n) +".npy")
    
    
    radii, centers, curvature = FeatureExtraction.calculate_circle_radius_and_center(right_line)
    # print("Radii:", radii)
    # print("Centers:", centers)
    # print("Curvature:", len(curvature))
    # print("Right Line:", len(right_line))

    # PrintDataArray(n)
    # plot_lines_once(n)
    # plot_Polyfit(n)
    # plot_lines_and_curvature(n)
    # plot_Poly_and_curvature(n)
    # plot_boundaries_once(n)
    # plot_boundaries_animation()
    # plot_lines_animation()
    plot_lines_animation_with_polyfit()
    # plot_lines_and_curvature_animation()
    # plot_Poly_and_curvature_animation()
    # plot_points_and_circles(right_line, radii, centers)
    # plot_curvature(curvature, right_line)  #Not working
    # plot_points_and_circles_together(n)
    
    # # Sample path data
    # x_list, y_list = right_line[:, 0], right_line[:, 1]
  

    # # Simulate heading (in radians)
    # heading_list = np.arctan2(np.gradient(y_list), np.gradient(x_list))

    # # Simulate curvature
    # curvature = np.gradient(heading_list) / np.gradient(np.sqrt(np.gradient(x_list)**2 + np.gradient(y_list)**2))

    # # Plot the original path
    # plt.plot(x_list, y_list, label="Original Path")

    # # Plot the curvature using the function
    # plot_curvature(x_list, y_list, heading_list, curvature, k=0.001, c="r-", label="Curvature Plot")

    # # Show the plot
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Curvature Plot Example")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__':
     main()