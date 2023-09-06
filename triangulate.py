import cv2
import numpy as np





def point3D(P, pre_pose, cur_pose, q1, q2, max_depth=100, max_reproj_err=15.0):
    pre_proj = np.dot(P, pre_pose)
    cur_proj = np.dot(P, cur_pose)

    # Triangulate points from i-1'th image
    q1 = q1.reshape(-1, 1, 2)
    q2 = q2.reshape(-1, 1, 2)

    mappoint = cv2.triangulatePoints(pre_proj, cur_proj, q1, q2)
    mappoint = np.transpose(mappoint[:3] / mappoint[3])
    
    mask = []

    for i in range(len(mappoint)):
        # Check positive depth for both camera frames
        depth_pre = np.dot(pre_pose[:3, :3], mappoint[i]) + pre_pose[:3, 3]
        depth_cur = np.dot(cur_pose[:3, :3], mappoint[i]) + cur_pose[:3, 3]

        # Ensure the Z component (depth) is positive
        if depth_pre[2] <= 0 or depth_cur[2] <= 0:
            mask.append(False)
            continue
        
        # Reprojection Error for the first image
        proj_pt1 = np.dot(pre_proj, np.append(mappoint[i], 1))
        proj_pt1 /= proj_pt1[2]
        reproj_err1 = np.linalg.norm(proj_pt1[:2] - q1[i])

        # Reprojection Error for the second image
        proj_pt2 = np.dot(cur_proj, np.append(mappoint[i], 1))
        proj_pt2 /= proj_pt2[2]
        reproj_err2 = np.linalg.norm(proj_pt2[:2] - q2[i])

        # # Check against criteria
        # if (depth_pre[2] <= max_depth and depth_cur[2] <= max_depth
        #     and reproj_err1 <= max_reproj_err and reproj_err2 <= max_reproj_err):
        # # if depth_pre[2] <= max_depth and depth_cur[2] <= max_depth:
        #     mask.append(True)
        # else:
        #     mask.append(False)
        mask.append(True)
    mappoint = mappoint[mask]
    q1 = q1[mask]
    q2 = q2[mask]

    q1 = q1.reshape(-1, 2)
    q2 = q2.reshape(-1, 2)

    return mappoint, q1, q2



######################## add limit to depth as we are far from camera #############################


def triangulate_points(P1, P2, points1, points2):
    # P1 & P2 - (3,4)
    # point1/2- (N,2)
    triangulated_points = []
    for x1, x2 in zip(points1, points2):
        A = np.vstack([
            x1[0] * P1[2, :] - P1[0, :],
            x1[1] * P1[2, :] - P1[1, :],
            x2[0] * P2[2, :] - P2[0, :],
            x2[1] * P2[2, :] - P2[1, :]
        ])
        _, _, V = np.linalg.svd(A)
        X = V[-1]  # Keeping this in homogeneous coordinates for now.
        triangulated_points.append(X)
    return np.array(triangulated_points)

def is_depth_positive(P, X):
    x_cam = np.linalg.inv(P[:, :3]) @ (X[:3] / X[3] - P[:, 3])
    return x_cam[2] > 0

def is_angle_valid(P1, P2, X, threshold=np.radians(1)):
    C1 = -np.linalg.inv(P1[:, :3]) @ P1[:, 3]
    C2 = -np.linalg.inv(P2[:, :3]) @ P2[:, 3]
    ray1 = X[:3] / X[3] - C1
    ray2 = X[:3] / X[3] - C2
    cos_angle = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2))
    return np.arccos(cos_angle) > threshold

# def robust_triangulation_ransac(P1, P2, points1, points2, threshold=1.0, max_iterations=1000, early_termination_ratio=0.9):

#     num_points = len(points1)
#     best_inliers = np.zeros(num_points, dtype=bool)
#     best_inlier_count = 0

#     for _ in range(max_iterations):
#         random_indices = np.random.choice(num_points, size=8, replace=False)
#         subset_points1 = points1[random_indices]
#         subset_points2 = points2[random_indices]
#         # Triangulate the randomly selected points
#         triangulated_subset = triangulate_points(P1, P2, subset_points1, subset_points2)

#         # Project the triangulated points back to image coordinates for all points
#         ones_column = np.ones((triangulated_subset.shape[0], 1))
#         projected_points1_all = (P1 @ np.hstack((triangulated_subset[:,:3], ones_column)).T).T
#         projected_points2_all = (P2 @ np.hstack((triangulated_subset[:,:3], ones_column)).T).T


#         # Convert from homogeneous to Cartesian coordinates
#         projected_points1_all = projected_points1_all[:, :2] / projected_points1_all[:, 2, None]
#         projected_points2_all = projected_points2_all[:, :2] / projected_points2_all[:, 2, None]

#         # Instead of computing error for all points, compute only for the subset.
#         error1 = np.linalg.norm(subset_points1 - projected_points1_all, axis=1)
#         error2 = np.linalg.norm(subset_points2 - projected_points2_all, axis=1)

#         inliers = (error1 < threshold) & (error2 < threshold)

#         current_inlier_count = np.sum(inliers)
        
#         if current_inlier_count > best_inlier_count:
#             best_inlier_count = current_inlier_count
#             best_inliers = inliers

#             # Early termination check
#             if best_inlier_count/num_points > early_termination_ratio:
#                 break

#     # Triangulate using all best inliers
#     triangulated_best = triangulate_points(P1, P2, points1[best_inliers], points2[best_inliers])

#     return points1[best_inliers], points2[best_inliers], triangulated_best


def compute_reprojection_error(P, x, X):
    x_projected = P @ X
    x_projected /= x_projected[2] # Convert to inhomogeneous coordinates
    return np.linalg.norm(x_projected[:2] - x[:2])


def net_triangulation_function(P, pre_pose, cur_pose, points1, points2, ransac_threshold=1.0, ransac_iterations=1000, reprojection_threshold=15.0):
    P1 = np.dot(P, pre_pose)
    P2 = np.dot(P, cur_pose)
    # Robust triangulation using RANSAC
    # inliers1, inliers2, triangulated = robust_triangulation_ransac(
    #     P1, P2, points1, points2, ransac_threshold, ransac_iterations)
    triangulated = triangulate_points(P1,P2,points1,points2)
    inliers1 = points1
    inliers2 = points2

    # Filtering triangulated points
    valid_points = []
    valid_keypoints1 = []
    valid_keypoints2 = []
    for i, X in enumerate(triangulated):
        error1 = compute_reprojection_error(P1, inliers1[i], X)
        error2 = compute_reprojection_error(P2, inliers2[i], X)
        # if is_depth_positive(P1, X) and is_depth_positive(P2, X) and is_angle_valid(P1, P2, X):
        if is_depth_positive(P1, X) and is_depth_positive(P2, X) and error1 < reprojection_threshold and error2 < reprojection_threshold:
            valid_points.append(X)
            valid_keypoints1.append(inliers1[i])
            valid_keypoints2.append(inliers2[i])

    return np.array(valid_points), np.array(valid_keypoints1), np.array(valid_keypoints2)


# Let's break down the size/shape of each returned array from the `net_triangulation_function`.

# 1. **`valid_points`** (returned as `np.array(valid_points)`):
#   - **Type**: numpy array
#   - **Shape**: (M, 4), where M is the number of valid 3D points after filtering. 
#   - his is a list of 3D points in homogeneous coordinates that passed checks (depth consistency and triangulation angle).

# 2. **`valid_keypoints1`** (returned as `np.array(valid_keypoints1)`):
#   - **Type**: numpy array
#   - **Shape**: (M, 2)
#   - **Description**: This array represents the keypoints from the first image that correspond to the valid 3D map points. 
#      Each keypoint is given by its x and y coordinates in the image, usually in pixels. 
#      Since each valid 3D point corresponds to a keypoint from each image, the number of keypoints here matches the number of `valid_points`.

# 3. **`valid_keypoints2`** (returned as `np.array(valid_keypoints2)`):
#   - **Type**: numpy array
#   - **Shape**: (M, 2)

# In summary, the function returns three arrays. All have the same number of rows M (which is the number of valid 3D points), 
# but they differ in the number of columns, reflecting the dimensionality of their respective data (3D points vs 2D keypoints).


