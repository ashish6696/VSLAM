import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt 

from lib.visualization import plotting
from lib.visualization.video import play_trip
from dataload import load_data
from feature_OptFlow  import feature_matcher
from intial_pose_E import get_pose
from tqdm import tqdm







def main():
    data_dir = '/Users/ashish.garg1/Downloads/VisualSLAM-main/kitti_data'  # Try KITTI_sequence_2 too
    K, P, gt_poses, images = load_data(data_dir)


    # play_trip(vo.images)   # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            res ,q1, q2 = feature_matcher(i,images)
            try:
                transf = get_pose(q1, q2, K, P)
            except:
                print("i", i)
                continue
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            print ("\nGround truth pose:\n" + str(gt_pose))
            print ("\n Current pose:\n" + str(cur_pose))
            print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
  
    
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()
