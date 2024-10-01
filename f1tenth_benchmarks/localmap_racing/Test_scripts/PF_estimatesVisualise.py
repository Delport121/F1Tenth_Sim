# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # Get the current script's directory
# script_dir = os.path.dirname(os.path.abspath(__file__))
# print(script_dir)

# # Move two directories back
# parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))
# print(parent_dir)

# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)

import os
import numpy as np
import matplotlib.pyplot as plt
from f1tenth_benchmarks.utils.MapData import MapData
from f1tenth_benchmarks.utils.track_utils import CentreLine

map_name = "aut"
map_data = MapData(map_name)
track = CentreLine(map_name)

track.widths = track.widths
l1 = track.path[:, :2] + track.nvecs * track.widths[:, 0][:, None]
l2 = track.path[:, :2] - track.nvecs * track.widths[:, 1][:, None]

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move to the directory of the file (two directories back, then into Logs/...):
npy_file_path = os.path.join(script_dir, "../../../Logs/FullStackPP/RawData_full_stack_pp/cf_estimates_aut_2.npy")
npy_file_path1 = os.path.join(script_dir, "../../../Logs/FullStackPP/RawData_full_stack_pp/pf_estimates_aut_2.npy")

# Load the NumPy file
data = np.load(npy_file_path)  # Assuming the data shape is (N, 3) where N is the number of samples
data1 = np.load(npy_file_path1)  

# Extract x, y, and orientation (theta)
x = data[:, 0]
y = data[:, 1]
theta = data[:, 2]  # Orientation angle in radians
x1 = data1[:, 0]
y1 = data1[:, 1]
theta1 = data1[:, 2]  # Orientation angle in radians

# Create the plot
plt.figure(figsize=(8, 8))
plt.plot(track.path[:, 0], track.path[:, 1], '--', linewidth=2, color='black', label='Track Centerline')
plt.plot(l1[:, 0], l1[:, 1], color='green', label='Track Boundary')
plt.plot(l2[:, 0], l2[:, 1], color='green')
plt.plot(x1, y1, label="Particle Filter Path")
plt.plot(x, y, label="Curvature Filter Path")


# Plot the orientation as arrows
arrow_length = 0.1  # Adjust the length of the orientation arrows
for i in range(len(x)):
    plt.arrow(x[i], y[i], arrow_length * np.cos(theta[i]), arrow_length * np.sin(theta[i]), 
              head_width=0.05, head_length=0.1, fc='red', ec='red')
    
for i in range(len(x1)):
    plt.arrow(x1[i], y1[i], arrow_length * np.cos(theta1[i]), arrow_length * np.sin(theta1[i]), 
              head_width=0.05, head_length=0.1, fc='blue', ec='blue')


plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Object Path with Orientation')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Ensure equal scaling on both axes
plt.show()

Timestep_file_path = os.path.join(script_dir, "../../../Logs/FullStackPP/RawData_full_stack_pp/cf_steptimes_aut_1.npy")
Time_file_path1 = os.path.join(script_dir, "../../../Logs/FullStackPP/RawData_full_stack_pp/pf_steptimes_aut_1.npy")

# Load the NumPy file
Timedata = np.load(Timestep_file_path)  
Timedata1 = np.load(Time_file_path1)  
print(Timedata)
print(Timedata1)


plt.plot(Timedata, label="Curvature Filter Time")
plt.plot( Timedata1, label="Particel Filter Time")
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
plt.ylim([0, 0.04])
plt.title('Time for each step')
plt.legend()
plt.grid(True)
# plt.axis('equal')  # Ensure equal scaling on both axes
plt.show()
