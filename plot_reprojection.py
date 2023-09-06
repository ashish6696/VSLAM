import cv2
import numpy as np
import matplotlib.pyplot as plt

def reproject_3D_point(P, X):
    # Project the 3D point using the projection matrix
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    x = P @ X.T
    # Convert from homogeneous to Cartesian coordinates
    x = x[:2] / x[2]
    return x

def plot_images_with_reprojection(P, map_point_3D, prepose, curpose, prekeypoint, currkeypoint, preimage, currimage):
    P1 = np.dot(P, prepose)
    P2 = np.dot(P, curpose)
    # Compute reprojections
    reprojected_pre = reproject_3D_point(P1, map_point_3D)
    reprojected_curr = reproject_3D_point(P2, map_point_3D)

    # Plot
    reprojected_pre = reprojected_pre.T  # Transpose to shape (288, 2)
    reprojected_curr = reprojected_curr.T  # Transpose to shape (288, 2)

    print( "shape:", np.shape(prekeypoint), "type:", type(prekeypoint))
    print("shape:", np.shape(reprojected_pre), "type:", type(reprojected_pre))

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))


    # Display preimage with keypoints and reprojections
    axes[0].imshow(preimage, cmap='gray')
    axes[0].scatter(prekeypoint[:, 0], prekeypoint[:, 1], c='red', marker='x', s=10)
    axes[0].scatter(reprojected_pre[:, 0], reprojected_pre[:, 1], c='blue', marker='x', s=10)
    axes[0].set_title("Preimage")

    # Display currimage with keypoints and reprojections
    axes[1].imshow(currimage, cmap='gray')
    axes[1].scatter(currkeypoint[:, 0], currkeypoint[:, 1], c='red', marker='x', s=10)
    axes[1].scatter(reprojected_curr[:, 0], reprojected_curr[:, 1], c='blue', marker='x', s=10)
    axes[1].set_title("Currimage")
    
    plt.tight_layout()
    plt.show()