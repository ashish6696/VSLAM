import os
import numpy as np
import cv2


def load_data( data_dir):
    K, P = load_calib(os.path.join(data_dir, 'calib.txt'))
    gt_poses = load_poses(os.path.join(data_dir, 'poses.txt'))
    images = load_images(os.path.join(data_dir, 'image_l'))
    return K, P, gt_poses, images
    # K , P is numpy array
    # gt_poses is list of numpy array
    # images is list of numpy array




def load_calib(filepath):
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
    return K, P

def load_poses(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

def load_images(filepath):
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]


def main():
    data_dir = '/Users/ashish.garg1/Downloads/VisualSLAM-main/kitti_data'  # Try KITTI_sequence_2 too
    K, P, gt_poses, images = load_data(data_dir)
    print("K: ", K)
    print("P: ", P)
    # gt_poses is list
    print("gt_poses: ", type(gt_poses))
    print("gt_poses: ", len(gt_poses))
    print("gt_poses: ", gt_poses[0])
    # images is list
    print("images: ", type(images))
    print("images: ", len(images))
    print("images: ", images[0].shape)
    # image is numpy array
    print("images: ", type(images[0]))





if __name__ == "__main__":
    main()
