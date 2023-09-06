import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def euler_from_rotation_matrix(R):
    """Extract Yaw, Pitch, Roll from a rotation matrix."""
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        print("R:", R)
        raise ValueError(f"Invalid rotation matrix provided: {R}")
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.rad2deg(np.array([x, y, z]))

def draw_camera_frustum(ax, pose, scale=1.0):
    """Draws a camera frustum in the 3D plotly figure."""
    R, t = pose[:3, :3], pose[:3, 3]
    
    # Camera points: define the camera center and a simple pyramid frustum
    points = np.array([
        [0, 0, 0],  # Camera center
        [scale, scale, scale],
        [scale, -scale, scale],
        [-scale, -scale, scale],
        [-scale, scale, scale]
    ])
    
    # Transform the points using the camera pose
    transformed_points = (R @ points.T).T + t

    # Define the lines that compose the frustum
    lines = [
        [transformed_points[0], transformed_points[1]],
        [transformed_points[0], transformed_points[2]],
        [transformed_points[0], transformed_points[3]],
        [transformed_points[0], transformed_points[4]],
        [transformed_points[1], transformed_points[2]],
        [transformed_points[2], transformed_points[3]],
        [transformed_points[3], transformed_points[4]],
        [transformed_points[4], transformed_points[1]]
    ]
    
    # Add the lines to the figure
    for line in lines:
        line = np.array(line)
        ax.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines', line=dict(color='blue')))

def visualize_tracks_angles(list1, list2):
    # Create subplots
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('3D Tracks', 'Yaw', 'Pitch', 'Roll'),
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                               [{'type': 'scatter'}, {'type': 'scatter'}]])


    # Extract translations and rotations for both lists
    translations1 = np.array([t for _, t in list1])
    rotations1 = [R for R, _ in list1]
    
    translations2 = np.array([t for _, t in list2])
    rotations2 = [R for R, _ in list2]
    
    # Plot 3D camera poses
    for i in range(len(translations1)):
        pose1 = np.eye(4)
        pose1[:3, :3] = rotations1[i]
        pose1[:3, 3] = translations1[i]
        draw_camera_frustum(fig, pose1)

    for i in range(len(translations2)):
        pose2 = np.eye(4)
        pose2[:3, :3] = rotations2[i]
        pose2[:3, 3] = translations2[i]
        draw_camera_frustum(fig, pose2, scale=1.5)  # Slightly larger frustum for visualization clarity
    
    # Extract yaw, pitch, roll for both lists
    yaws1, pitches1, rolls1 = zip(*[euler_from_rotation_matrix(R) for R in rotations1])
    yaws2, pitches2, rolls2 = zip(*[euler_from_rotation_matrix(R) for R in rotations2])

    time_axis = np.arange(len(list1))

    # Plot Yaw, Pitch, Roll for list 1
    fig.add_trace(go.Scatter(x=time_axis, y=yaws1, mode='lines', name='Yaw 1'), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_axis, y=pitches1, mode='lines', name='Pitch 1'), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_axis, y=rolls1, mode='lines', name='Roll 1'), row=2, col=2)

    # Plot Yaw, Pitch, Roll for list 2
    fig.add_trace(go.Scatter(x=time_axis, y=yaws2, mode='lines', name='Yaw 2', line=dict(dash='dot')), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_axis, y=pitches2, mode='lines', name='Pitch 2', line=dict(dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_axis, y=rolls2, mode='lines', name='Roll 2', line=dict(dash='dot')), row=2, col=2)

    # Display
    fig.show()


