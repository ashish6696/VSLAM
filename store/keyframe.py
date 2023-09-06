import numpy as np
import cv2

def add_connection(connections, source_keyframe, destination_keyframe, relative_pose, information_matrix, matches):
    connection = {
        'source_keyframe': source_keyframe,
        'destination_keyframe': destination_keyframe,
        'relative_pose': relative_pose,
        'information_matrix': information_matrix,
        'matches': matches
    }
    connections.append(connection)

def add_keyframe(keyframes, keyframe_id, timestamp, pose, image_matrix, keypoints, descriptors):
    if keyframe_id in keyframes:
        print(f"Keyframe with ID {keyframe_id} already exists.")
        return False

    keyframe = {
        'keyframe_id': keyframe_id,
        'timestamp': timestamp,
        'pose': np.array(pose),
        'image_matrix': image_matrix,
        'keypoints': keypoints,
        'descriptors': descriptors,
        'connections': []
    }
    keyframes[keyframe_id] = keyframe
    return True

def update_keyframe(keyframes, keyframe_id, **kwargs):
    """
    Update the keyframe with the given keyframe_id.
    Use kwargs to specify which attributes to update with new values.
    For example, to update the timestamp and pose: 
    update_keyframe(keyframes, 1, timestamp=new_timestamp, pose=new_pose)
    """
    if keyframe_id not in keyframes:
        print(f"Keyframe with ID {keyframe_id} not found.")
        return False

    for attribute, value in kwargs.items():
        if attribute in keyframes[keyframe_id]:
            if attribute == "pose" or attribute == "relative_pose" or attribute == "information_matrix":
                keyframes[keyframe_id][attribute] = np.array(value)
            else:
                keyframes[keyframe_id][attribute] = value
        else:
            print(f"Attribute {attribute} not found in keyframe.")
    return True

def delete_keyframe(keyframes, keyframe_id):
    """
    Delete the keyframe with the given keyframe_id.
    delete_keyframe(keyframes, 1)
    """
    if keyframe_id not in keyframes:
        print(f"Keyframe with ID {keyframe_id} not found.")
        return False
    
    del keyframes[keyframe_id]
    return True

def add_connection(keyframes, source_keyframe_id, destination_keyframe_id, relative_pose=None, information_matrix=None, matches=None):
    """
    Add a bidirectional connection between source_keyframe_id and destination_keyframe_id.
    """
    if source_keyframe_id not in keyframes:
        print(f"Keyframe with ID {source_keyframe_id} not found.")
        return False
    if destination_keyframe_id not in keyframes:
        print(f"Keyframe with ID {destination_keyframe_id} not found.")
        return False

    # Set default values if arguments are None
    if relative_pose is None:
        relative_pose = np.eye(4)  # 4x4 identity transformation matrix
    if information_matrix is None:
        information_matrix = np.eye(6)
    if matches is None:
        matches = []

    # Connection from source to destination
    connection_to_dest = {
        'source_keyframe': source_keyframe_id,
        'destination_keyframe': destination_keyframe_id,
        'relative_pose': relative_pose,
        'information_matrix': information_matrix,
        'matches': np.array(matches)
    }
    
    # Connection from destination to source
    # Note: You might need to invert the relative pose or handle the information matrix and matches appropriately
    inverted_relative_pose = np.linalg.inv(relative_pose)  # Inverting the transformation matrix
    connection_to_source = {
        'source_keyframe': destination_keyframe_id,
        'destination_keyframe': source_keyframe_id,
        'relative_pose': inverted_relative_pose,
        'information_matrix': information_matrix,  # This might need adjustments
        'matches': np.array(matches)  # This might need adjustments
    }
    
    keyframes[source_keyframe_id]['connections'].append(connection_to_dest)
    keyframes[destination_keyframe_id]['connections'].append(connection_to_source)
    return True


keyframes = {}

# Example of usage:
# Load an image into 'img_matrix' using cv2.imread
img_matrix = cv2.imread("path_to_image.jpg")
add_keyframe(keyframes, 1, 12345678.9, [0,0,0,1,0,0,0], img_matrix, [(100,150), (200,250)], ["binary1", "binary2"])

add_connection(keyframes[1]['connections'], 1, 2, [0,0,0,1,0,0,1], np.eye(6), [[0, 1], [1, 2], [2, 3]])

print(f"Number of Keyframes: {len(keyframes)}")
print(f"Total Number of Connections for Keyframe 1: {len(keyframes[1]['connections'])}")

# Now, let's update the timestamp and pose of keyframe 1:
new_timestamp = 12345679.0
new_pose = [0,0,1,1,0,0,0]
update_keyframe(keyframes, 1, timestamp=new_timestamp, pose=new_pose)

print(f"Updated Timestamp for Keyframe 1: {keyframes[1]['timestamp']}")
print(f"Updated Pose for Keyframe 1: {keyframes[1]['pose']}")
