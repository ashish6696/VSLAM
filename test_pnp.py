import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt 
import sys


from lib.visualization import plotting
from lib.visualization.video import play_trip


from dataloader.dataload import load_data
from features.feature_OptFlow  import feature_matcher
from intialization.intial_pose_E import get_pose
from triangulate import point3D, net_triangulation_function
from plot_reprojection import plot_images_with_reprojection
from pnp_both import optimize_poses
# from motion_only_BA import BundleAdjustment, CameraParameters, convert_array_to_list
from tqdm import tqdm




def main():
    data_dir = '/Users/ashish.garg1/Downloads/slam/Ashish/kitti2'  # Try KITTI_sequence_2 too
    K, P, gt_poses, images = load_data(data_dir)


    # play_trip(vo.images)   # Comment out to not play the trip
    
    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(gt_poses, unit="pose")):
        if i == 0:
            # print("gt_pose",gt_pose)
            cur_pose = gt_pose
        else:
            res ,q1, q2 = feature_matcher(i,images)
            transf = get_pose(q1, q2, K, P)
            pre_pose = cur_pose
            cur_pose = np.matmul(pre_pose, np.linalg.inv(transf))
            print("keypoint shape",q1.shape)
            mappoint,f1,f2 = point3D(P, pre_pose, cur_pose, q1, q2)
            print("mappoint shape",mappoint.shape)
            draw=False
            if i ==33:
                draw=True
            # print("pre_pose",pre_pose)
            # pre_pose,cur_pose=optimize_poses(P, mappoint, pre_pose, cur_pose, f1, f2, images[i-1], images[i], draw)

            

            
           

            
            # print ("\nGround truth pose:\n" + str(gt_pose))
            # print ("\n Current pose:\n" + str(cur_pose))
            # print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
  
    
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")
    print(mappoint.shape)

if __name__ == "__main__":
    main()
