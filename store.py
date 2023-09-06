
# image view set = https://www.mathworks.com/help/vision/ref/imageviewset.html
# world point set = https://www.mathworks.com/help/vision/ref/worldpointset.html
class image_view_set:
    def __init__(self):
        self.key_dict = {}
 
    def addview(ViewId, AbsPose , Points, Features): 
        # 2 view set add
        # absolute camera pose
        # keypoint, feature points, descriptors
        # vSetKeyFrames = addView(rigidtform3d, Points=prePoints,Features=preFeatures.Features);
        self.key_dict[str("View",ViewId)] = {"AbsPose":AbsPose, "Points":Points, "Features":Features}
        return self.key_dict

    def addconnection(preViewId, currViewId, relPose, Matches=indexPairs):
        # 1 connnection add
        # relative camera pose
        # feature matching --->index pairs
        # loop closure connection
        # vSetKeyFrames = addConnection(preViewId, currViewId, relPose, Matches=indexPairs);
        self.key_dict[str(preViewId,"to",currViewId)] = {"RelPose":relPose, "Matches":Matches}
        return self.key_dict

class world_point_set:
    def __init__(self):
        self.point_dict = {}
        self.newPointIdx = {}
 
    def addcorrespondence(PointIdx, ViewId, indexPairs):
        # 2 correspondence add
        # 3d to 2d correspondence
        # mapPointSet   = addCorrespondences(mapPointSet, preViewId, newPointIdx, indexPairs(:,1));
        self.point_dict[str("Point",PointIdx)] = {"View": ViewId, "indexPairs":indexPairs}


    def addworldpoint():
        # 1 world point
        # 3d world point
        # [mapPointSet, newPointIdx] = addWorldPoints(mapPointSet, xyzWorldPoints);
        # self.point_dict[str("Point",PointIdx)] = {"View": ViewId, "indexPairs":indexPairs}
        # self.newPointIdx[str("mapPointSet",mapPointSet)] = {"newPointIdx":newPointIdx, "xyzWorldPoints":xyzWorldPoints}



