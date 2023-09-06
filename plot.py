import numpy as np
import matplotlib.pyplot as plt

def visualize_frustums(pose1, pose2, fov_x, fov_y, z_near_plane, z_far_plane, points, fig_size=(8, 8)):
    
    def render_frustum(points, ax, color='r'):
        line_indices = [
            [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6],
            [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
        ]
        
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        zs = [point[2] for point in points]
        buffer = 1

        ax.set_xlim([min(xs)-buffer, max(xs)+buffer])
        ax.set_ylim([min(ys)-buffer, max(ys)+buffer])
        ax.set_zlim([min(zs)-buffer, max(zs)+buffer])

        for idx_pair in line_indices:
            line = np.transpose([points[idx_pair[0]], points[idx_pair[1]]])
            ax.plot(line[0], line[2], line[1], color)
        
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.autoscale_view()

    def render_axis(pose, ax, length=1):
        origin = pose[:3, 3]
        x_axis = origin + length * pose[:3, 0]
        y_axis = origin + length * pose[:3, 1]
        z_axis = origin + length * pose[:3, 2]
        
        ax.quiver(origin[0], origin[2], origin[1], x_axis[0]-origin[0], x_axis[2]-origin[2], x_axis[1]-origin[1], color='c')
        ax.quiver(origin[0], origin[2], origin[1], y_axis[0]-origin[0], y_axis[2]-origin[2], y_axis[1]-origin[1], color='m')
        ax.quiver(origin[0], origin[2], origin[1], z_axis[0]-origin[0], z_axis[2]-origin[2], z_axis[1]-origin[1], color='y')

    def get_perspective_mat(fov_x_deg, fov_y_deg, z_near, z_far):
        fov_x_rad = fov_x_deg * np.pi / 180
        fov_y_rad = fov_y_deg * np.pi / 180
        f_x = 1 / np.tan(fov_x_rad / 2)
        f_y = 1 / np.tan(fov_y_rad / 2)
        return np.array([
            [f_x, 0, 0, 0],
            [0, f_y, 0, 0],
            [0, 0, (z_far + z_near) / (z_near - z_far), 2 * z_far * z_near / (z_near - z_far)],
            [0, 0, -1, 0]
        ])

    def get_view_matrix_from_pose(pose_matrix):
        return np.linalg.inv(pose_matrix)

    points_clip = np.array([
        [-1, -1,  1, 1], [1, -1,  1, 1], [-1, 1,  1, 1], [1, 1,  1, 1],
        [-1, -1, -1, 1], [1, -1, -1, 1], [-1,  1, -1, 1], [1,  1, -1, 1]
    ], dtype=float)

    M_wv1 = get_view_matrix_from_pose(pose1)
    M_vc = get_perspective_mat(fov_x, fov_y, z_near_plane, z_far_plane)
    M_vw1 = np.linalg.inv(M_wv1)
    M_cv = np.linalg.inv(M_vc)
    
    
    # pointcv= [(np.matmul(M_cv, pc) / np.matmul(M_cv, pc)[3]) for pc in points_clip]
    # pointcv= [i[2]=-i[2] for i in pointcv]
    # points_world1 = [np.matmul(M_vw1, pointcv)

    M_wv1 = get_view_matrix_from_pose(pose1)
    M_vw1= np.linalg.inv(M_wv1)
    points_world1 = [np.matmul(M_vw1, np.matmul(M_cv, pc) / np.matmul(M_cv, pc)[3]) for pc in points_clip]
    M_wv2 = get_view_matrix_from_pose(pose2)
    M_vw2 = np.linalg.inv(M_wv2)
    points_world2 = [np.matmul(M_vw2, np.matmul(M_cv, pc) / np.matmul(M_cv, pc)[3]) for pc in points_clip]

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")
    render_frustum(points_world1, ax, color='r')
    render_frustum(points_world2, ax, color='b')
    render_axis(pose1, ax)
    render_axis(pose2, ax)

    xs, ys, zs = zip(*points)
    ax.scatter(xs, zs, ys, marker='o', color='g', s=30)

    ax.set_title("Camera Frustums, 3D Points, and Coordinate Systems in World Space")
    plt.show()

pose1 = np.eye(4)
pose2 = np.array([
    [1, 0, 0, 2],
    [0, 1, 0, 0],
    [0, 0, 1, 2],
    [0, 0, 0, 1]
])
fov_x = 70
fov_y = 45
z_near_plane = 0.5
z_far_plane = 3
points = np.array([[2, 1, 1], [2, -1, 1], [2, 0, 2]])
fig_size = (8, 8)

visualize_frustums(pose1, pose2, fov_x, fov_y, z_near_plane, z_far_plane, points, fig_size)





