import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    p2 = np.sum(desc1 * desc1, axis = 1, keepdims=True) * np.ones((desc1.shape[0], desc2.shape[0]))
    q2 = np.sum(desc2 * desc2, axis = 1, keepdims=True).transpose() * np.ones((desc1.shape[0], desc2.shape[0]))
    pq = desc1 @ desc2.transpose()
    distances = p2 - 2*pq + q2
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m, 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        match_1 = np.arange(q1)
        match_2 = np.argmin(distances, axis=1)
        matches = np.array([match_1, match_2]).transpose()

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        match_1 = np.arange(q1)
        match_2 = np.argmin(distances, axis=1)
        matches_12 = np.array([match_1, match_2]).transpose()
        match_1 = np.arange(q2)
        match_2 = np.argmin(distances, axis=0)
        matches_21 = np.array([match_2, match_1]).transpose()
        if q1 > q2:
            mutual = matches_12[matches_21[:, 0], 1] == matches_21[:, 1]
            matches = matches_21[mutual]
        else:
            mutual = matches_21[matches_12[:, 1], 0] == matches_12[:, 0]
            matches = matches_12[mutual]

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        match_1 = np.arange(q1)
        match_2 = np.argmin(distances, axis=1)
        distances_sorted = np.sort(distances, axis=1)
        valid = distances_sorted[:,0] / distances_sorted[:,1] < 0.5
        matches = np.array([match_1, match_2])[:, valid].transpose()
    else:
        raise NotImplementedError
    return matches

