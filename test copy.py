import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt 



from lib.visualization import plotting
from lib.visualization.video import play_trip

from dataloader.dataload import load_data
from features.feature_OptFlow  import feature_matcher
from intialization.intial_pose_E import get_pose

from triangulate import point3D
from plot import visualize_frustums, compute_fov
from plot4 import main_plot
from plot5 import visualize_tracks_angles


# from motion_only_BA import BundleAdjustment, CameraParameters, convert_array_to_list
from tqdm import tqdm
from scipy.optimize import least_squares

import numpy as np

def _form_transf(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
def pose_to_stacked_vector(pose):

    
    # Extract rotation matrix and translation vector
    R = pose[:3, :3]
    t = pose[:3, 3][:, np.newaxis]  # Ensure t is a column vector

    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)

    # Ensure rvec is a column vector
    if rvec.shape[1] != 1:
        rvec = rvec[:, np.newaxis]

    # Stack rotation vector and translation vector
    stacked_vector = np.vstack((rvec, t))
    
    return stacked_vector

def reprojection_residuals(vcurr_pose, q2, Q ,P):
     # Get the rotation vector
    r = vcurr_pose[:3]
    # Create the rotation matrix from the rotation vector
    R, _ = cv2.Rodrigues(r)
    # Get the translation vector
    t = vcurr_pose[3:]
    # Create the transformation matrix from the rotation matrix and translation vector
    curr_pose = _form_transf(R, t)
    # Create the projection matrix for the i-1'th image and i'th image
    projection = np.matmul(P, curr_pose)
    # Make the 3D points homogenize
    ones = np.ones((q2.shape[0], 1))
    Q = np.hstack([Q, ones])

    # Project 3D points from i'th image to i-1'th image
    q_pred = Q.dot(projection.T)
    # Un-homogenize
    q_pred = q_pred[:, :2].T / q_pred[:, 2]

    # Calculate the residuals
    residuals = np.array([q_pred - q2.T]).flatten()
    return residuals

def estimate_pose(curr_pose, q2, Q ,P ,max_iter=100):
    early_termination_threshold = 5

    # Initialize the min_error and early_termination counter
    min_error = float('inf')
    early_termination = 0
    curr_pose = pose_to_stacked_vector(curr_pose)

    for _ in range(max_iter):
        # Choose 6 random feature points
        sample_idx = np.random.choice(range(q2.shape[0]), 6)
        sample_q2, sample_Q= q2[sample_idx], Q[sample_idx]

        # Make the start guess
        in_guess = curr_pose.squeeze()
        print(in_guess)
        # Perform least squares optimization
        opt_res = least_squares(reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                args=(sample_q2,sample_Q, P))

        # Calculate the error for the optimized transformation
        error = reprojection_residuals(opt_res.x, q2, Q ,P)
        error = error.reshape((Q.shape[0] , 2))
        error = np.sum(np.linalg.norm(error, axis=1))

        # Check if the error is less the the current min error. Save the result if it is
        if error < min_error:
            min_error = error
            out_pose = opt_res.x
            early_termination = 0
            # curr_pose = out_pose
        else:
            early_termination += 1
        if early_termination == early_termination_threshold:
            # If we have not fund any better result in early_termination_threshold iterations
            break

    # Get the rotation vector
    r = out_pose[:3]
    # Make the rotation matrix
    R, _ = cv2.Rodrigues(r)
    # Get the translation vector
    t = out_pose[3:]
    # Make the transformation matrix
    transformation_matrix = _form_transf(R, t)
    return transformation_matrix







def main():
    data_dir = '/Users/ashish.garg1/Downloads/VisualSLAM-main/kitti2'  # Try KITTI_sequence_2 too
    K, P, gt_poses, images = load_data(data_dir)


    # play_trip(vo.images)   # Comment out to not play the trip
    
    gt_path = []
    estimated_path = []
    estimate_pose_l = []
    gt_pose_l = []
    for i, gt_pose in enumerate(tqdm(gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:

            res ,q1, q2 = feature_matcher(i,images)
            
            transf = get_pose(q1, q2, K, P)
            
            pre_pose = cur_pose
            
            cur_pose = np.matmul(pre_pose, np.linalg.inv(transf))
            
            mappoint,q1,q2 = point3D(P, pre_pose, cur_pose, q1, q2)
            # resu=reprojection_residuals(cur_pose, q2, mappoint ,P)
            # cur_pose = estimate_pose(cur_pose, q2, mappoint ,P)

            

            # cur_pose,inliners = simple_pnp_ransac(mappoint,q2,K)
            # _,cur_pose,inliners = solve_pnp_with_epnp_ransac(mappoint,q2,K)

            # tranf = estimate_pose(pre_pose, cur_pose, q1, q2, mappoint)
            # print ("\nGround truth pose:\n" + str(gt_pose))
            # print ("\n Current pose:\n" + str(cur_pose))
            print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_pose_l.append((gt_pose[:3,:3],gt_pose[:3,3]))
        estimate_pose_l.append((cur_pose[:3,:3],cur_pose[:3,3]))
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
    
    # print("images",images[0].shape)
    # print("q1",q1.shape)
    # print("q2",q2.shape)
    # print("q1[0]",q1[0:5])
    # print("K",K)
    # print("P",P)
    # print("transf",transf)
    print("pre_pose",pre_pose)
    print("cur_pose",cur_pose)
    




    print("mappoint",mappoint.shape)
    print("mappoint0",mappoint[0:10])
    # print("resu",resu)
    main_plot(images[i-1],images[i],pre_pose,cur_pose,q1,q2,K)
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")
    visualize_tracks_angles(gt_pose_l, estimate_pose_l)
    # print(mappoint.shape)
    visualize_frustums(pre_pose, cur_pose, 90, 90, 0.5, 100, mappoint)

if __name__ == "__main__":
    main()
