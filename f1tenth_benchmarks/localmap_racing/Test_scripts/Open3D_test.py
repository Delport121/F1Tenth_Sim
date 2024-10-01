import open3d as o3d
import numpy as np

# Create two sample 2D point clouds and add a third z-coordinate (set to 0)
points_src = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0]], dtype=np.float32)
points_tgt = points_src + np.array([0.5, 0.5, 0])  # Shift the second cloud slightly in the x and y directions

# Convert to Open3D point clouds
pcd_src = o3d.geometry.PointCloud()
pcd_src.points = o3d.utility.Vector3dVector(points_src)

pcd_tgt = o3d.geometry.PointCloud()
pcd_tgt.points = o3d.utility.Vector3dVector(points_tgt)

# Apply ICP
threshold = 1.0
trans_init = np.eye(4)  # Initial transformation
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_src, pcd_tgt, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Display results
print("Has converged:", reg_p2p.inlier_rmse)
print("Transformation matrix:\n", reg_p2p.transformation)

# Visualize the alignment
pcd_src.transform(reg_p2p.transformation)
o3d.visualization.draw_geometries([pcd_src, pcd_tgt])

