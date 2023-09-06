

def add_map_point(map_points, point_id, position, descriptor, observed_keyframes=None, correspondences=None):
    """
    Add a MapPoint to the map_points dictionary.
    """
    if point_id in map_points:
        print(f"MapPoint with ID {point_id} already exists.")
        return False

    map_point = {
        'point_id': point_id,
        'position': np.array(position),
        'descriptor': descriptor,
        'observed_keyframes': observed_keyframes if observed_keyframes else [],
        'correspondences': correspondences if correspondences else []  # List to store the 3D-to-2D correspondences
    }
    map_points[point_id] = map_point
    return True

def add_correspondence_to_mappoint(map_points, point_id, keyframe_id, feature_index):
    """
    Add a new correspondence to the MapPoint's correspondences list.
    """
    if point_id not in map_points:
        print(f"MapPoint with ID {point_id} not found.")
        return False
    
    correspondence = {
        'PointIndex': point_id,
        'keyframeId': keyframe_id,
        'FeatureIndex': feature_index
    }
    
    map_points[point_id]['correspondences'].append(correspondence)
    return True

def remove_world_point(map_points, point_id):
    """
    Remove a specific MapPoint.
    """
    if point_id not in map_points:
        print(f"MapPoint with ID {point_id} not found.")
        return False
    del map_points[point_id]
    return True

def update_world_point(map_points, point_id, position=None, descriptor=None, observed_keyframes=None):
    """
    Update attributes of a specific MapPoint.
    """
    if point_id not in map_points:
        print(f"MapPoint with ID {point_id} not found.")
        return False

    if position is not None:
        map_points[point_id]['position'] = np.array(position)

    if descriptor is not None:
        map_points[point_id]['descriptor'] = descriptor

    if observed_keyframes is not None:
        map_points[point_id]['observed_keyframes'] = observed_keyframes
    
    return True

def remove_correspondence(map_points, point_id, keyframe_id):
    """
    Remove a correspondence for a specific MapPoint and keyframe.
    """
    if point_id not in map_points:
        print(f"MapPoint with ID {point_id} not found.")
        return False
    
    correspondences = map_points[point_id]['correspondences']
    for idx, correspondence in enumerate(correspondences):
        if correspondence['keyframeId'] == keyframe_id:
            del correspondences[idx]
            return True

    print(f"Correspondence with keyframe ID {keyframe_id} not found for MapPoint {point_id}.")
    return False

def update_correspondence(map_points, point_id, keyframe_id, new_feature_index):
    """
    Update the feature index for a specific correspondence of a MapPoint and keyframe.
    """
    if point_id not in map_points:
        print(f"MapPoint with ID {point_id} not found.")
        return False
    
    correspondences = map_points[point_id]['correspondences']
    for correspondence in correspondences:
        if correspondence['keyframeId'] == keyframe_id:
            correspondence['FeatureIndex'] = new_feature_index
            return True

    print(f"Correspondence with keyframe ID {keyframe_id} not found for MapPoint {point_id}.")
    return False


# Initialize a dictionary to store MapPoints
map_points = {}
# Sample usage
add_map_point(map_points, 1, [1,2,3], "descriptor_val")
add_correspondence_to_mappoint(map_points, 1, 101, 25)
add_correspondence_to_mappoint(map_points, 1, 102, 30)
