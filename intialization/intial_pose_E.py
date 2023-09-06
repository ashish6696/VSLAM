
import numpy as np
import cv2
import sys



sys.path.append("/Users/ashish.garg1/Downloads/slam/Ashish")
from tqdm import tqdm
from dataloader.dataload import load_data
from features.feature_OptFlow import feature_matcher







def _form_transf(R, t):
    """
    Makes a transformation matrix from the given rotation matrix and translation vector

    Parameters
    ----------
    R (ndarray): The rotation matrix
    t (list): The translation vector

    Returns
    -------
    T (ndarray): The transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T




    # This function should detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
    # The descriptors should then be matched using the class flann object (knnMatch with k=2)
    # Remove the matches not satisfying Lowe's ratio test
    # Return a list of the good matches for each image, sorted such that the n'th descriptor in image i matches the n'th descriptor in image i-1
    # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    pass



def decomp_essential_mat( E, q1, q2,K,P1):
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


    R1, R2, t = cv2.decomposeEssentialMat(E)
    T1 = _form_transf(R1,np.ndarray.flatten(t))
    T2 = _form_transf(R2,np.ndarray.flatten(t))
    T3 = _form_transf(R1,np.ndarray.flatten(-t))
    T4 = _form_transf(R2,np.ndarray.flatten(-t))
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
        hom_Q1 = cv2.triangulatePoints(P1, P, q1.T, q2.T)
        hom_Q2 = T @ hom_Q1
        # Un-homogenize
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

        total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
        relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                    np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
        positives.append(total_sum + relative_scale)
        

    # Decompose the Essential matrix using built in OpenCV function
    # Form the 4 possible transformation matrix T from R1, R2, and t
    # Create projection matrix using each T, and triangulate points hom_Q1
    # Transform hom_Q1 to second camera using T to create hom_Q2
    # Count how many points in hom_Q1 and hom_Q2 with positive z value
    # Return R and t pair which resulted in the most points with positive z

    max = np.argmax(positives)
    if (max == 2):
        # print(-t)
        return R1, np.ndarray.flatten(-t)
    elif (max == 3):
        # print(-t)
        return R2, np.ndarray.flatten(-t)
    elif (max == 0):
        # print(t)
        return R1, np.ndarray.flatten(t)
    elif (max == 1):
        # print(t)
        return R2, np.ndarray.flatten(t)
    


def get_pose( q1, q2,K,P1):
    # q1 (ndarray): The good keypoints matches position in i-1'th image
    # q2 (ndarray): The good keypoints matches position in i'th image


 


    Essential, mask = cv2.findEssentialMat(q1, q2, K)
    # print ("\nEssential matrix:\n" + str(Essential))

    R, t = decomp_essential_mat(Essential, q1, q2,K,P1)

    return _form_transf(R,t)
    # transformation_matrix (ndarray (4,3)): The transformation matrix



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
            res, q1, q2 = feature_matcher(i,images)
            transf = get_pose(q1, q2, K,P)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            print ("\nGround truth pose:\n" + str(gt_pose))
            print ("\n Current pose:\n" + str(cur_pose))
            print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
  
    


if __name__ == "__main__":
    main()
