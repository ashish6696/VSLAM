import numpy as np
import cv2
from matplotlib import pyplot as plt 
import sys
import os

import sys

sys.path.append("/Users/ashish.garg1/Downloads/slam/Ashish")
from dataloader.dataload import load_data 

def get_tiled_keypoints(img, fastFeatures ,tile_h=10, tile_w=20):

    # img (ndarray): The image to find keypoints in. Shape (height, width)
    # tile_h (int): The tile height
    # tile_w (int): The tile width

    def get_kps(x, y):
        # Get the image tile
        impatch = img[y:y + tile_h, x:x + tile_w]

        # Detect keypoints
        keypoints = fastFeatures.detect(impatch)

        # Correct the coordinate for the point
        for pt in keypoints:
            pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

        # Get the 10 best keypoints
        if len(keypoints) > 10:
            keypoints = sorted(keypoints, key=lambda x: -x.response)
            return keypoints[:10] # ---------------------------------------------- only one keypoint is saved at the moment
        return keypoints
    # Get the image height and width
    h, w, *_ = img.shape

    # Get the keypoints for each of the tiles
    kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

    # Flatten the keypoint list
    kp_list_flatten = np.concatenate(kp_list)
    return kp_list_flatten
    # kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)


def track_keypoints(img1, img2, kp1, max_error=4):

    # img1 (ndarray): i-1'th image. Shape (height, width)
    # img2 (ndarray): i'th image. Shape (height, width)
    # kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
    # max_error (float): The maximum acceptable error

    # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
    trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

    lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
    
    # Use optical flow to find tracked counterparts
    trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **lk_params)

    # Convert the status vector to boolean so we can use it as a mask
    trackable = st.astype(bool)

    # Create a maks there selects the keypoints there was trackable and under the max error
    under_thresh = np.where(err[trackable] < max_error, True, False)

    # Use the mask to select the keypoints
    trackpoints1 = trackpoints1[trackable][under_thresh]
    trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

    # Remove the keypoints there is outside the image
    h, w = img1.shape
    in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
    trackpoints1 = trackpoints1[in_bounds]
    trackpoints2 = trackpoints2[in_bounds]
    

    
    return trackpoints1, trackpoints2
    # trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
    # trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)

def feature_matcher(i,images,draw=False):
        fastFeatures = cv2.FastFeatureDetector_create()
        preimg=images[i-1]
        currimg=images[i]
        prekeypoints=get_tiled_keypoints(preimg,fastFeatures)
        preMatchedPoints, currMatchedPoints = track_keypoints(preimg, currimg, prekeypoints)
        return True,preMatchedPoints, currMatchedPoints


def feature_matcher1(i,images,draw=False):
    # Orb feature detector and descriptor
    scaleFactor = 1.2
    numLevels = 8
    numPoints = 1000

    # Orb feature matcher
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=4500)


    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures=numPoints, scaleFactor=scaleFactor, nlevels=numLevels)
    # numPoints: The maximum number of features to retain
    # other parameters: https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html
    # also has option for harris detector or FAST detector


    # Find putative feature matches
    # OTHEROPTION = cbf = cv2.BFMatcher()
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    
    # Detect keypoints and compute descriptors
    prekeypoints, preFeatures = orb.detectAndCompute(images[i - 1], None)
    currkeypoints, currFeatures = orb.detectAndCompute(images[i], None)
    # To understand kp data https://amroamroamro.github.io/mexopencv/matlab/cv.ORB.detectAndCompute.html

    
    # Convert keypoints to a list of points
    prekeypoints_list = [kp.pt for kp in prekeypoints]
    currkeypoints_list = [kp.pt for kp in currkeypoints]
    # To understand kp data https://amroamroamro.github.io/mexopencv/matlab/cv.ORB.detectAndCompute.html


    matches = flann.knnMatch(preFeatures, currFeatures, k=2)
    # Select matcher, matcher data : https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    #OTHEROPTION = matches = bf.knnMatch(preFeatures, currFeatures, k=2)
    
    
     # Apply ratio test to filter good matches
    good_matches = []
    for pair in matches:
        try:
            m, n = pair
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)
        except ValueError:
            pass
        
            
    
    if i == 1 :
        # If not enough matches are found, check the next frame
        minMatches = 100
        if len(good_matches) < minMatches:
            return False, None, None
    
    # Draw matches
    if draw:
        img3 = cv2.drawMatches(images[i-1], prekeypoints, images[i], currkeypoints,good_matches, None, flags=2)
        # https://docs.opencv.org/4.x/d4/d5d/group__features2d__draw.html
        plt.imshow(img3)
        plt.show()


    preMatchedPoints = np.float32([ prekeypoints_list[m.queryIdx] for m in good_matches ])
    currMatchedPoints = np.float32([ currkeypoints_list[m.trainIdx] for m in good_matches ])
    # queryIdx: This refers to the index of a keypoint in the query (source) image. In other words, it's an index that points to a keypoint from the image for which you want to find matches.
    # trainIdx: This refers to the index of a keypoint in the train (destination) image. It's an index that points to a keypoint in the image you are trying to match the keypoints against.
    # When you perform feature matching, you are looking for pairs of keypoints between two images that correspond to the same feature or point in the scene. 
    # The queryIdx and trainIdx values help you identify which keypoints in the source and destination images are being matched.

    return True,preMatchedPoints, currMatchedPoints
    # MatchPoints is numpy array of size (N,2) where N is number of good matched points (760,2)
    # 2 is u,v coordinates of matched points






def main():
    data_dir = '/Users/ashish.garg1/Downloads/slam/Ashish/kitti2'  # Try KITTI_sequence_2 too
    K, P ,gt_poses, images= load_data(data_dir)
    for i in range(len(images)):
        print("i: ", i)
        res,preMatchedPoints, currMatchedPoints= feature_matcher(i,images)
    print(type(preMatchedPoints))
    print(preMatchedPoints.shape)


if __name__ == "__main__":
    main()
