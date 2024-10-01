import numpy as np
import matplotlib.pyplot as plt
import ICP

# Sample local maps as numpy arrays (x, y, theta)
map_1 = np.array([[0, 0, 0], [1, 1, 0.1], [2, 2, 0.2], [3, 3, 0.3], [4, 4, 0.4]], dtype=np.float32)
map_2 = map_1 + np.array([0.5, 0.5, 0.05])  # Shift and rotate slightly for testing

# Extract only x, y for ICP
points_map_1 = map_1[:, :2]
points_map_2 = map_2[:, :2]

# Initial guess for transformation
initial_transformation = np.eye(3)

# Run ICP algorithm
T, distances, iterations = ICP.ICP(points_map_2, points_map_1, init_pose=initial_transformation, max_iterations=100)

# Apply the transformation to map_2
transformed_points_map_2 = np.dot(T, np.vstack((points_map_2.T, np.ones((1, points_map_2.shape[0])))))

# Plot the original and transformed points
plt.figure(figsize=(8, 8))
plt.scatter(points_map_1[:, 0], points_map_1[:, 1], color='blue', label='Map 1 (Target)')
plt.scatter(points_map_2[:, 0], points_map_2[:, 1], color='red', label='Map 2 (Before Alignment)')
plt.scatter(transformed_points_map_2[0, :], transformed_points_map_2[1, :], color='green', label='Map 2 (After Alignment)', marker='x')

plt.legend()
plt.title('ICP Scan Matching')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()

# Print the cost (mean distance after alignment)
cost = np.mean(distances)
print("Cost of matching the two scans:", cost)
