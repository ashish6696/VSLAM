import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt 
import math

from lib.visualization import plotting
from lib.visualization.video import play_trip
from dataload import load_data
from feature import feature_matcher
from intial_pose_E import get_pose
from tqdm import tqdm



def select_HorF(preMatchedPoints, currMatchedPoints):


    ###### H = CPP --> Intializer.cc --> Compute H12
    ###### maskH = CPP --> Intializer.cc --> CheckHomography,ReprojectionError 
    ###### H,maskH,scoreH = CPP ---> Intializer.cc --> FindHomography
    # Compute homography and evaluate reconstruction  --> used for planar queries
    H, maskH = cv2.findHomography(preMatchedPoints, currMatchedPoints, cv2.RANSAC, 5.0)
    scoreH = maskH.sum()
    # maskH = np.array([[1], [0], [1], ....]) 1 means inlier, 0 means outlier :: maskH[0, 0] is 1
    # Example: https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    # findHomography: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780



    # Compute fundamental matrix and evaluate reconstruction  --> used for non-planar queries
    F, maskF = cv2.findFundamentalMat(preMatchedPoints, currMatchedPoints, cv2.FM_RANSAC, 3.0)
    # FM_RANSAC: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
    scoreF = maskF.sum()

    return H, maskH, scoreH, True

    # Select the model based on a heuristic
    ratio = scoreH / (scoreH + scoreF)
    ratioThreshold = 0.45
    if ratio > ratioThreshold:
        return H, maskH, scoreH, True
    else:
        return F, maskF, scoreF, False
    
def _form_transf(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
    

# def select_best_solutionH(outR, outT, prematchpoints, currmatchpoints, K):
    # print("K: ", K)
    # print("prematchpoints: ", prematchpoints[0])
    best_error = float('inf')
    best_R = None
    best_T = None

    for i in range(4):
        R = outR[i]
        T = outT[i]

        # Create projection matrices P1 and P2
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, T))

        # Triangulate points
        points_4d_homogeneous = cv2.triangulatePoints(P1, P2, prematchpoints.T, currmatchpoints.T)

        # Convert homogeneous coordinates to 3D
        points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]

        # Calculate and accumulate reprojection error
        total_reprojection_error = 0

        for j in range(len(points_3d[0])):
            # Reproject points using R and T
            point_proj1 = np.dot(K, np.dot(R, points_3d[:, j]))  # Adjust indices if needed
            point_proj1 /= point_proj1[2]
            point_proj2 = np.dot(K, np.dot(np.eye(3), points_3d[:, j]) + T)  # Adjust indices if needed
            point_proj2 /= point_proj2[2]

            #################### don't know exact mm_per_pixel assume = 20 ####################
            mm_per_pixel = 20
            point_proj1 = point_proj1 / mm_per_pixel
            point_proj2 = point_proj2 / mm_per_pixel

            # Calculate squared reprojection error
            print("point_proj1: ", point_proj1[:2])
            print("prematchpoints[j]: ", prematchpoints[j])
            point_error = np.sum((point_proj1[:2].reshape(-1, 1) - prematchpoints[j])**2) + np.sum((point_proj2[:2].reshape(-1, 1) - currmatchpoints[j])**2)
            total_reprojection_error += point_error

        # Compare and store the best solution
        if total_reprojection_error < best_error:
            best_error = total_reprojection_error
            best_R = R
            best_T = T

    return best_R, best_T

def select_best_solutionH(outR, outT, prematchpoints, currmatchpoints, K, P1):
    """
    Decompose the Essential matrix

    Parameters
    ----------
    E (ndarray): Essential matrix
    q1 (ndarray): The good keypoints matches position in i-1'th image
    q2 (ndarray): The good keypoints matches position in i'th image

    Returns
    -------
    right_pair (list): Contains the rotation matrix and translation vector
    """
    R1,R2,R3,R4 = outR
    t1,t2,t3,t4 = outT
    T1 = _form_transf(R1,np.ndarray.flatten(t1))
    T2 = _form_transf(R2,np.ndarray.flatten(t2))
    T3 = _form_transf(R3,np.ndarray.flatten(t3))
    T4 = _form_transf(R4,np.ndarray.flatten(t4))
    transformations = [T1, T2, T3, T4]
    
    # Homogenize K
    K = np.concatenate(( K, np.zeros((3,1)) ), axis = 1)

    # List of projections
    projections = [K @ T1, K @ T2, K @ T3, K @ T4]

    np.set_printoptions(suppress=True)

    # print ("\nTransform 1\n" +  str(T1))
    # print ("\nTransform 2\n" +  str(T2))
    # print ("\nTransform 3\n" +  str(T3))
    # print ("\nTransform 4\n" +  str(T4))

    positives = []
    for P, T in zip(projections, transformations):
        hom_Q1 = cv2.triangulatePoints(P1, P, prematchpoints.T,currmatchpoints.T)
        hom_Q2 = T @ hom_Q1
        # Un-homogenize
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

        total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
        relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                    np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
        positives.append(total_sum + relative_scale)
        


    print("positives: ", positives)
    max = np.argmax(positives)
    if (max == 2):
        # print(-t)
        return R3, np.ndarray.flatten(t3)
    elif (max == 3):
        # print(-t)
        return R4, np.ndarray.flatten(t4)
    elif (max == 0):
        # print(t)
        return R1, np.ndarray.flatten(t1)
    elif (max == 1):
        # print(t)
        return R2, np.ndarray.flatten(t2)
    








 




    

def get_poseH(H, K,prematchpoints, currmatchpoints,P):
    # tform = H
    retval, outR, outT, outN = cv2.decomposeHomographyMat(H, K)
    # decomposeHomographyMat: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga396afb6411b30758cf6b19d7c1ed5bb5

    best_R, best_T = select_best_solutionH(outR, outT, prematchpoints, currmatchpoints, K,P)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = best_R
    T[:3, 3] = best_T.ravel()
    return T
    


def main():
    data_dir = '/Users/ashish.garg1/Downloads/VisualSLAM-main/kitti2'  # Try KITTI_sequence_2 too
    K, P, gt_poses, images = load_data(data_dir)



    # play_trip(vo.images)   # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            res ,q1, q2 = feature_matcher(i,images)
            matrix, mask, score, isHomography = select_HorF(q1, q2)
            if isHomography:
                transf = get_poseH(matrix, K , q1, q2,  P)
                
            else:
                transf = get_poseF(q1, q2, K, P)
            # transf = get_pose(q1, q2, K, P)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            print ("\nGround truth pose:\n" + str(gt_pose))
            print ("\n Current pose:\n" + str(cur_pose))
            print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
  
    
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()

