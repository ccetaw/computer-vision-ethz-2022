import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    # TODO: Filter out keypoints that are too close to the edges
    # Filter out keypoints that are in 5% margin
    edge_rate = 0.05
    margin_left = int(edge_rate * img.shape[1])
    margin_right = int((1-edge_rate) * img.shape[1])
    margin_top = int(edge_rate * img.shape[0])
    margin_bottom = int((1-edge_rate) * img.shape[0])
    in_margin = (margin_left < keypoints[:, 0]) * (keypoints[:, 0] < margin_right) * (margin_top < keypoints[:, 1]) * (keypoints[:, 1] < margin_bottom)
    keypoints_f = keypoints[in_margin]
    return keypoints_f

# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

