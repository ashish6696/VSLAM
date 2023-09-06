import numpy as np
import cv2

def reprojection_residuals(dof, image, keypoints, mappoints, P, draw=False):
    
    # Convert DOF to absolute pose matrix
    R, _ = cv2.Rodrigues(dof[:3])
    t = dof[3:].reshape(3, 1)
    absolute_pose = np.hstack([R, t])
    
    # Compute the projection matrix
    projection = P.dot(absolute_pose)

    # Add homogeneous coordinate to mappoints
    ones = np.ones((mappoints.shape[0], 1))
    mappoints_h = np.hstack([mappoints, ones])

    # Compute the predicted keypoints
    keypoints_pred = mappoints_h.dot(projection.T)
    
    # Unhomogenize the keypoints
    keypoints_pred = (keypoints_pred[:, :2] / keypoints_pred[:, 2:3])
    
    # Calculate residuals
    residuals = (keypoints_pred - keypoints).flatten()

    if draw:
        for pt1, pt2 in zip(keypoints, keypoints_pred):
            cv2.circle(image, tuple(pt1.astype(int)), 3, (0, 255, 0), -1)  # Actual in green
            cv2.circle(image, tuple(pt2.astype(int)), 3, (0, 0, 255), -1)  # Predicted in red
            cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (255, 0, 0)) # Line connecting them in blue

        for i, res in enumerate(residuals.reshape(-1, 2)):
            cv2.putText(image, str(np.linalg.norm(res)), tuple(keypoints[i].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow("Reprojection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return residuals
