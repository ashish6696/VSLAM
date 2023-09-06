import numpy as np
import plotly.graph_objs as go

def rotation_to_euler(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def visualize_poses(poses):
    rolls, pitches, yaws, times = [], [], [], []
    fig_3d = go.Figure()

    for i, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]
        roll, pitch, yaw = rotation_to_euler(R)
        
        rolls.append(np.degrees(roll))
        pitches.append(np.degrees(pitch))
        yaws.append(np.degrees(yaw))
        times.append(i)
        
        # Plotting camera as a frustum
        camera_size = 0.2
        dir = np.array([0, 0, 1])
        # vertices of the pyramid
        v = np.array([[0, 0, 0], [camera_size, camera_size, camera_size], [-camera_size, camera_size, camera_size], [-camera_size, -camera_size, camera_size], [camera_size, -camera_size, camera_size]])
        vertices = v @ R.T + t
        
        # connecting the vertices to form the pyramid
        lines = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        for start, end in lines:
            fig_3d.add_trace(go.Scatter3d(x=[vertices[start, 0], vertices[end, 0]], 
                                          y=[vertices[start, 1], vertices[end, 1]], 
                                          z=[vertices[start, 2], vertices[end, 2]], 
                                          mode='lines'))

    # Plotting roll, pitch, yaw with time
    fig_rpy = go.Figure()
    fig_rpy.add_trace(go.Scatter(x=times, y=rolls, mode='lines', name='Roll'))
    fig_rpy.add_trace(go.Scatter(x=times, y=pitches, mode='lines', name='Pitch'))
    fig_rpy.add_trace(go.Scatter(x=times, y=yaws, mode='lines', name='Yaw'))
    fig_rpy.update_layout(title='Roll, Pitch, Yaw vs Time', xaxis_title='Time', yaxis_title='Degrees')
    fig_rpy.show()
    
    # Displaying the 3D plot
    fig_3d.show()

# Sample data
pose1 = np.eye(4)
pose2 = np.array([
    [0, -1, 0, 2],
    [1, 0, 0, 0],
    [0, 0, 1, 2],
    [0, 0, 0, 1]
])
poses = [pose1, pose2]

visualize_poses(poses)
