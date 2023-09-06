import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt 
from motion_only_BA import BundleAdjustment, CameraParameters



from lib.visualization import plotting
from lib.visualization.video import play_trip
from dataload import load_data
from feature_OptFlow  import feature_matcher
from intial_pose_E import get_pose
from triangulate import point3D
# from motion_only_BA import BundleAdjustment, CameraParameters, convert_array_to_list
from tqdm import tqdm

def extract_intrinsics(intrinsic_matrix):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    return fx, fy, cx, cy

def bundle_adjustment(ba_optimizer, q1, q2, mappoint, cur_pose, pre_pose, cam_params):
    l = len(q1)

    projections = [(0, i, q1[i]) for i in range(l)] + [(1, i, q2[i]) for i in range(l)] 
    ba_optimizer.add_pose(1, cur_pose, cam_params, fixed=False)
    ba_optimizer.add_pose(0, pre_pose, cam_params, fixed=True)
    for i, point in enumerate(mappoint):
        ba_optimizer.add_point(i, point, fixed=True)
    # Add edges (projections) between camera poses and points
    for pose_id, point_id, measurement in projections:
        ba_optimizer.add_edge(point_id, pose_id, measurement)

    # Perform optimization
    ba_optimizer.optimize(max_iterations=50)

    # Retrieve optimized camera poses and points
    optimized_pose = [ba_optimizer.get_pose(i) for i in range(2)]
    rot = optimized_pose[1].rotation().rotation_matrix()
    tra = optimized_pose[1].position()
    T = np.eye(4)
    T[:3, :3] = rot 
    T[:3, 3] = tra
    cur_pose = T
    return cur_pose



def main():
    data_dir = '/Users/ashish.garg1/Downloads/VisualSLAM-main/kitti2'  # Try KITTI_sequence_2 too
    K, P, gt_poses, images = load_data(data_dir)

    ba_optimizer = BundleAdjustment()

    fx, fy, cx, cy = extract_intrinsics(K)

    cam_params = CameraParameters(fx, fy, cx, cy, baseline=0.1)
    # play_trip(vo.images)   # Comment out to not play the trip
    
    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            res ,q1, q2 = feature_matcher(i,images)
            transf = get_pose(q1, q2, K, P)
            pre_pose = cur_pose
            cur_pose = np.matmul(pre_pose, np.linalg.inv(transf))
            mappoint = point3D(P, pre_pose, cur_pose, q1, q2)
            cur_pose = bundle_adjustment(ba_optimizer, q1, q2, mappoint, cur_pose, pre_pose, cam_params)

            
            # print ("\nGround truth pose:\n" + str(gt_pose))
            # print ("\n Current pose:\n" + str(cur_pose))
            # print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
  
    
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")
    

if __name__ == "__main__":
    main()
