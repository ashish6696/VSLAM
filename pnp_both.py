import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from plot_reprojection import plot_images_with_reprojection

def reproject_3D_point(P, X):
    # Project the 3D point using the projection matrix
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    x = P @ X.T
    # Convert from homogeneous to Cartesian coordinates
    x = x[:2] / x[2]
    return x

def compute_reprojection_error(params, P, map_point_3D, keypoints):
    """Compute reprojection error for the given pose."""
    rvec, tvec = params[:3], params[3:6]
    R, _ = cv2.Rodrigues(rvec)
    pose = np.hstack((R, tvec.reshape(3, 1)))
    # pose = np.vstack((pose, np.ones((1,pose.shape[0]))))
    last_row = np.array([[0, 0, 0, 1]])
    pose = np.vstack((pose, last_row))
    # print("P",P)
    # print("pose",pose)
    projected = reproject_3D_point(P @ pose, map_point_3D)
    return np.linalg.norm(keypoints - projected.T, axis=1)

def optimize_pose(P, map_point_3D, pose_initial, keypoints):
    # print("pose shape",pose_initial.shape)
    rvec_initial, _ = cv2.Rodrigues(pose_initial[:3, :3])
    params_initial = np.hstack((rvec_initial.ravel(), pose_initial[:, 3]))
    print("inital residual",compute_reprojection_error(params_initial ,P,map_point_3D,keypoints))
    result = least_squares(compute_reprojection_error, params_initial,verbose=2, x_scale='jac', ftol=1e-4, method='trf', max_nfev=50, args=(P, map_point_3D, keypoints))
    print("final residual",compute_reprojection_error(result.x ,P,map_point_3D,keypoints))
    optimized_rvec, optimized_tvec = result.x[:3], result.x[3:6]
    optimized_R, _ = cv2.Rodrigues(optimized_rvec)
    return np.column_stack((optimized_R, optimized_tvec))

def optimize_poses(P, map_point_3D, prepose, currpose, prekeypoint, currkeypoint, preimage=None, currimage=None, draw=False):
    
    optimized_prepose = optimize_pose(P, map_point_3D, prepose, prekeypoint)
    
    optimized_currpose = optimize_pose(P, map_point_3D, currpose, currkeypoint)
    last_row = np.array([[0, 0, 0, 1]])

    optimized_prepose = np.vstack((optimized_prepose, last_row))
    optimized_currpose = np.vstack((optimized_currpose, last_row))
    # If draw is True, visualize using plot_images_with_reprojection
    if draw and preimage is not None and currimage is not None:
        plot_images_with_reprojection(P, map_point_3D, optimized_prepose, optimized_currpose, prekeypoint, currkeypoint, preimage, currimage)
    # optimize_prepose= prepose
    return optimized_prepose, optimized_currpose

# Remember to also include the definition of the `plot_images_with_reprojection` function as mentioned in previous steps.
