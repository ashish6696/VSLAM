import numpy as np
import cv2
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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

def draw_camera(pose, color, scale=10):
    """Generate lines and points to visualize a camera frustum."""
    R, t = pose[:3, :3], pose[:3, 3]

    # Define the camera frustum (a small pyramid)
    frustum_points = np.array([
        [0, 0, 0],
        [scale * 0.5, scale * 0.4, scale],
        [scale * 0.5, -scale * 0.4, scale],
        [-scale * 0.5, -scale * 0.4, scale],
        [-scale * 0.5, scale * 0.4, scale]
    ])
    
    frustum_points = (R @ frustum_points.T).T + t
    lines = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

    # Extracting the line start and end points for plotting
    x_lines, y_lines, z_lines = [], [], []
    for start, end in lines:
        x_lines.extend([frustum_points[start, 0], frustum_points[end, 0], None])
        y_lines.extend([frustum_points[start, 1], frustum_points[end, 1], None])
        z_lines.extend([frustum_points[start, 2], frustum_points[end, 2], None])

    return x_lines, y_lines, z_lines

def visualize_cameras_points(img1, img2, pose1, pose2, points3D, q1, q2, intrinsics):
    """Visualize cameras, 3D points, and matches using Plotly."""
    
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.4, 0.3, 0.3],
        subplot_titles=('3D Visualization', 'Camera 1 Features', 'Camera 2 Features'),
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}, {'type': 'xy'}]]
    )

    # Filter points
    points3D_filtered = filter_points(points3D, pose1[:3, 3], pose2[:3, 3])

    # Draw camera frustums
    x_lines1, y_lines1, z_lines1 = draw_camera(pose1, 'blue')
    fig.add_trace(go.Scatter3d(x=x_lines1, y=y_lines1, z=z_lines1, mode='lines', line=dict(color='blue')), row=1, col=1)

    x_lines2, y_lines2, z_lines2 = draw_camera(pose2, 'red')
    fig.add_trace(go.Scatter3d(x=x_lines2, y=y_lines2, z=z_lines2, mode='lines', line=dict(color='red')), row=1, col=1)

    # Plot 3D points
    fig.add_trace(go.Scatter3d(x=points3D_filtered[:, 0], y=points3D_filtered[:, 1], z=points3D_filtered[:, 2], mode='markers', marker=dict(size=3, color='green')), row=1, col=1)

    # 2D Visualizations of features on images
    fig.add_trace(go.Image(z=img1), row=1, col=2)
    fig.add_trace(go.Scatter(x=q1[:, 0], y=q1[:, 1], mode='markers', marker=dict(color='red'), name="Features Cam1"), row=1, col=2)
    
    fig.add_trace(go.Image(z=img2), row=1, col=3)
    fig.add_trace(go.Scatter(x=q2[:, 0], y=q2[:, 1], mode='markers', marker=dict(color='blue'), name="Features Cam2"), row=1, col=3)

    fig.update_layout(title_text="Cameras, Points, and Features Visualization")
    fig.show()

def transform_to_camera1_frame(points3D, pose1, pose2):
    """Transforms points and Camera 2 pose to Camera 1's coordinate frame."""
    R1, t1 = pose1[:3, :3], pose1[:3, 3]
    R2, t2 = pose2[:3, :3], pose2[:3, 3]
    
    # Transform 3D points
    transformed_points = (R1.T @ (points3D.T - t1.reshape(3, 1))).T
    
    # Transform Camera 2 pose
    transformed_R2 = R1.T @ R2
    transformed_t2 = R1.T @ (t2 - t1)
    transformed_pose2 = np.eye(4)
    transformed_pose2[:3, :3] = transformed_R2
    transformed_pose2[:3, 3] = transformed_t2
    
    return transformed_points, transformed_pose2

def visualize_cameras_points_adjusted(img1, img2, pose1, pose2, points3D, q1, q2, intrinsics):
    """Visualize cameras, 3D points, and matches using Plotly in Camera 1's coordinate frame."""
    
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.4, 0.3, 0.3],
        subplot_titles=('3D Visualization', 'Camera 1 Features', 'Camera 2 Features'),
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}, {'type': 'xy'}]]
    )

    # Transform points and Camera 2 to Camera 1's frame
    points3D_cam1, transformed_pose2 = transform_to_camera1_frame(points3D, pose1, pose2)
    
    # Filter points
    points3D_filtered = filter_points(points3D_cam1, np.zeros(3), transformed_pose2[:3, 3])

    # Draw Camera 1 (at origin since it's our reference frame)
    x_lines1, y_lines1, z_lines1 = draw_camera(np.eye(4), 'blue')  # Identity matrix because it's our reference
    fig.add_trace(go.Scatter3d(x=x_lines1, y=y_lines1, z=z_lines1, mode='lines', line=dict(color='blue')), row=1, col=1)

    # Draw transformed Camera 2
    x_lines2, y_lines2, z_lines2 = draw_camera(transformed_pose2, 'red')
    fig.add_trace(go.Scatter3d(x=x_lines2, y=y_lines2, z=z_lines2, mode='lines', line=dict(color='red')), row=1, col=1)

    # Plot 3D points
    fig.add_trace(go.Scatter3d(x=points3D_filtered[:, 0], y=points3D_filtered[:, 1], z=points3D_filtered[:, 2], mode='markers', marker=dict(size=3, color='green')), row=1, col=1)

    # 2D Visualizations of features on images
    fig.add_trace(go.Image(z=img1), row=1, col=2)
    fig.add_trace(go.Scatter(x=q1[:, 0], y=q1[:, 1], mode='markers', marker=dict(color='red'), name="Features Cam1"), row=1, col=2)
    
    fig.add_trace(go.Image(z=img2), row=1, col=3)
    fig.add_trace(go.Scatter(x=q2[:, 0], y=q2[:, 1], mode='markers', marker=dict(color='blue'), name="Features Cam2"), row=1, col=3)

    fig.update_layout(title_text="Cameras, Points, and Features Visualization in Camera 1's Frame")
    fig.show()

def main_plot(img1_shape, img2_shape, pose1, pose2, q1, q2, intrinsics):
    # Create projection matrices for both cameras
    P1 = intrinsics @ pose1[:3]
    P2 = intrinsics @ pose2[:3]
    
    # Triangulate points
    points3D = triangulate_points(P1, P2, q1, q2)
    
    # Visualize
    # visualize_cameras_points(img1_shape, img2_shape, pose1, pose2, points3D, q1, q2, intrinsics)
    visualize_cameras_points_adjusted(img1_shape, img2_shape, pose1, pose2, points3D, q1, q2, intrinsics)