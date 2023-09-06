import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def triangulate_points(P1, P2, pts1, pts2):
    """Triangulate points from two views."""
    points_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points = points_hom / points_hom[3]
    return points[:3].T

def filter_points(points3D, C1, C2, threshold=100):
    """Filter out points that are too far from the camera centers."""
    midpoint = 0.5 * (C1 + C2)
    distances = np.linalg.norm(points3D - midpoint, axis=1)
    return points3D[distances < threshold]

def draw_camera(ax, pose, K, img_shape, color='b'):
    """Draw the camera as a frustum."""
    h, w = img_shape
    # Define the four corners of the image in pixel
    corners_img = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]  # Close the loop
    ])

    # Convert the corners to normalized image coordinates
    f = (K[0, 0] + K[1, 1]) / 2  # average focal length
    cx, cy = K[0, 2], K[1, 2]
    corners_norm_img = (corners_img - [cx, cy]) / f
    print("Shape of corners_norm_img:", corners_norm_img.shape)
    # Convert to 3D camera coordinates (Z=1 plane)
    corners_cam = np.hstack([corners_norm_img, np.ones((4, 1))])*10

    # Convert to world coordinates
    R, t = pose[:3, :3], pose[:3, 3]
    corners_world = (R @ corners_cam.T + t.reshape(3, 1)).T

    # Draw the frustum lines
    for i in range(4):
        ax.plot3D(*zip(t, corners_world[i]), color=color)

    # Draw the base rectangle
    ax.plot3D(*corners_world.T, color=color)


def visualize_cameras_points_and_matches(img1, img2, pose1, pose2, points3D, q1, q2, intrinsics):
    """Visualize cameras, 3D points, and matches on images."""
    fig = plt.figure(figsize=(20, 10))
    
    # 3D Visualization
    ax3D = fig.add_subplot(2, 1, 1, projection='3d')
    
    # Camera centers
    C1 = -np.linalg.inv(pose1[:3, :3]) @ pose1[:3, 3]
    C2 = -np.linalg.inv(pose2[:3, :3]) @ pose2[:3, 3]
    
    # Filter points
    points3D_filtered = filter_points(points3D, C1, C2)

    # Draw cameras as frustums
    draw_camera(ax3D, pose1, intrinsics, img1.shape, 'b')
    draw_camera(ax3D, pose2, intrinsics, img2.shape, 'r')
    
    # Plot 3D points
    ax3D.scatter(points3D_filtered[:, 0], points3D_filtered[:, 1], points3D_filtered[:, 2], c='g', marker='.', s=5, label="3D Points")

    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Y')
    ax3D.set_zlabel('Z')
    ax3D.legend()

    # 2D Visualization
    ax1 = fig.add_subplot(2, 2, 3)
    ax1.imshow(img1, cmap='gray')
    ax1.scatter(q1[:, 0], q1[:, 1], c='r', marker='o')
    ax1.set_title('Camera 1 Features')

    ax2 = fig.add_subplot(2, 2, 4)
    ax2.imshow(img2, cmap='gray')
    ax2.scatter(q2[:, 0], q2[:, 1], c='b', marker='o')
    ax2.set_title('Camera 2 Features')

    plt.tight_layout()
    plt.show()

def main_plot(img1, img2, pose1, pose2, q1, q2, intrinsics):
    # Create projection matrices for both cameras
    P1 = intrinsics @ pose1[:3]
    P2 = intrinsics @ pose2[:3]
    print("pose1",pose1)
    print("pose2",pose2)
    
    # Triangulate points
    points3D = triangulate_points(P1, P2, q1, q2)
    
    # Visualize
    visualize_cameras_points_and_matches(img1, img2, pose1, pose2, points3D, q1, q2, intrinsics)


