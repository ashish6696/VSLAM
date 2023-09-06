import numpy as np
import cv2
from matplotlib import pyplot as plt 
import sys

sys.path.append("/Users/ashish.garg1/Downloads/slam/Ashish")
from dataloader.dataload import load_data





def feature_matcher(i,images,draw=False):
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

    # Extract indices of the good matches
    pre_matched_indices = [m.queryIdx for m in good_matches]
    curr_matched_indices = [m.trainIdx for m in good_matches]

    # Extract corresponding descriptors
    preMatchedDescriptors = preFeatures[pre_matched_indices]
    currMatchedDescriptors = currFeatures[curr_matched_indices]



    preMatchedPoints = np.float32([ prekeypoints_list[m.queryIdx] for m in good_matches ])
    currMatchedPoints = np.float32([ currkeypoints_list[m.trainIdx] for m in good_matches ])
    # queryIdx: This refers to the index of a keypoint in the query (source) image. In other words, it's an index that points to a keypoint from the image for which you want to find matches.
    # trainIdx: This refers to the index of a keypoint in the train (destination) image. It's an index that points to a keypoint in the image you are trying to match the keypoints against.
    # When you perform feature matching, you are looking for pairs of keypoints between two images that correspond to the same feature or point in the scene. 
    # The queryIdx and trainIdx values help you identify which keypoints in the source and destination images are being matched.



    return True,preMatchedPoints, currMatchedPoints
    # return True,preMatchedPoints, currMatchedPoints, preMatchedDescriptors, currMatchedDescriptors



    # MatchPoints is numpy array of size (N,2) where N is number of good matched points (760,2)
    # 2 is u,v coordinates of matched points
    # MatchDescriptor is numpy array of size (N,M) where M is size of descriptor





def main():
    data_dir = '/Users/ashish.garg1/Downloads/VisualSLAM-main/kitti2'  # Try KITTI_sequence_2 too
    K, P ,gt_poses, images= load_data(data_dir)
    for i in range(len(images)):
        print("i: ", i)
        res,preMatchedPoints, currMatchedPoints= feature_matcher(i,images,True)
    print(type(preMatchedPoints))
    print(preMatchedPoints.shape)


if __name__ == "__main__":
    main()
