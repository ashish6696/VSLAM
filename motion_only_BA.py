import numpy as np
import g2o

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, cam, fixed=False):
        q = g2o.Quaternion(pose[:3, :3])
        t = pose[:3, 3]
        sbacam = g2o.SBACam(q, t)
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()

# Define camera parameters
class CameraParameters:
    def __init__(self, fx, fy, cx, cy, baseline=0.1):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline

def convert_array_to_list(array):
    # Ensure the input array has shape N * 2
    if array.shape[1] != 2:
        raise ValueError("Input array should have shape N * 2")
    
    # Convert each row of the array to a NumPy array of size 2
    result_list = [np.array(row) for row in array]
    
    return result_list

def main():

    # Create an instance of the BundleAdjustment class
    ba_optimizer = BundleAdjustment()

    # Define camera parameters
    cam_params = CameraParameters(fx=500, fy=500, cx=320, cy=240, baseline=0.1)

    # Simulated camera poses, 3D points, and their projections
    camera_poses = [np.eye(4), np.eye(4), np.eye(4)]
    points_3d = np.random.rand(10, 3) * 10
    print("points_3d: ", points_3d)
    projections = [(0, i, np.random.rand(2)) for i in range(10)] + [(1, i, np.random.rand(2)) for i in range(10)] + [(2, i, np.random.rand(2)) for i in range(10)]

    # Add camera poses and points to the optimizer
    for i, pose in enumerate(camera_poses):
        ba_optimizer.add_pose(i, pose, cam_params, fixed=False)
    for i, point in enumerate(points_3d):
        ba_optimizer.add_point(i, point, fixed=False)

    # Add edges (projections) between camera poses and points
    for pose_id, point_id, measurement in projections:
        ba_optimizer.add_edge(point_id, pose_id, measurement)

    # Perform optimization
    ba_optimizer.optimize(max_iterations=50)

    # Retrieve optimized camera poses and points
    optimized_poses = [ba_optimizer.get_pose(i) for i in range(len(camera_poses))]
    optimized_points = [ba_optimizer.get_point(i) for i in range(len(points_3d))]

    # Print optimized camera poses and points
    print("Optimized Camera Poses:")
    for pose in optimized_poses:
        print("pose:", pose.rotation().rotation_matrix(), pose.position())

    print("\nOptimized 3D Points:")
    for point in optimized_points:
        print("point:", point)

if __name__ == "__main__":
    main()
