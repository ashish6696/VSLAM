import cv2
import numpy as np

def pnp_ransac(known_3d_points, image_2d_points, camera_matrix, dist_coeffs, ransac_threshold=2.0, max_reprojection_error=8.0, max_iterations=2000):
    """
    Perform PnP estimation with RANSAC using OpenCV.

    Parameters:
    - known_3d_points: List of 3D points in world coordinates. Each point should be a 3-element list or array.
    - image_2d_points: List of 2D points in image coordinates. Each point should be a 2-element list or array.
    - camera_matrix: Camera intrinsic matrix (3x3).
    - dist_coeffs: Distortion coefficients (5x1).
    - ransac_threshold: RANSAC inlier threshold.
    - max_reprojection_error: Maximum allowed reprojection error for inliers.
    - max_iterations: Maximum number of RANSAC iterations.

    Returns:
    - success: Boolean indicating if PnP estimation was successful.
    - pose: The camera pose as a 4x4 transformation matrix.
    """

    # Convert input points to NumPy arrays
    known_3d_points = np.array(known_3d_points, dtype=np.float32)
    image_2d_points = np.array(image_2d_points, dtype=np.float32)

    # Perform PnP with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(known_3d_points, image_2d_points, camera_matrix, dist_coeffs,
                                                      reprojectionError=max_reprojection_error,
                                                      confidence=0.99, iterationsCount=max_iterations,
                                                      flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create the 4x4 transformation matrix [R | t]
        pose = np.hstack((R, tvec))
        pose = np.vstack((pose, [0, 0, 0, 1]))
    else:
        pose = np.eye(4)  # Return an identity matrix if PnP fails
        
    return success, pose
