import numpy as np
import copy
import cv2

def draw_keypoints(img, keypoints, color=(0, 0, 255), thickness=2):
    if len(img.shape) == 2:
        img = img[:,:,None].repeat(3, 2)
    if keypoints is None:
        raise ValueError("Error! Keypoints should not be None")
    keypoints = np.array(keypoints)
    for p in keypoints.tolist():
        pos_x, pos_y = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (pos_x, pos_y), thickness, color, -1)
    return img

def draw_segments(img, segments, color=(255, 0, 0), thickness=2):
    for s in segments:
        p1 = (int(round(s[0])), int(round(s[1])))
        p2 = (int(round(s[2])), int(round(s[3])))
        cv2.line(img, p1, p2, color, thickness)
    return img

def plot_image_with_keypoints(fname_out, img, keypoints, color=(0, 0, 255), thickness=2):
    img = copy.deepcopy(img)
    img_keypoints = draw_keypoints(img, keypoints, color=color, thickness=thickness)
    cv2.imwrite(fname_out, img_keypoints)
    print("[LOG] Number of keypoints: {0}. Writing keypoints visualization to {1}".format(keypoints.shape[0], fname_out))

def plot_image_pair_with_matches(fname_out, img1, keypoints1, img2, keypoints2, matches):
    # construct full image
    import pdb
    assert img1.shape[0] == img2.shape[0]
    assert img1.shape[1] == img2.shape[1]
    h, w = img1.shape[0], img1.shape[1]
    img = np.concatenate([img1, img2], 1)
    img = img[:,:,None].repeat(3, 2)
    img = draw_keypoints(img, keypoints1, color=(0, 0, 255), thickness=2)
    img = draw_keypoints(img, keypoints2 + np.array([w, 0])[None,:], color=(0, 0, 255), thickness=2)
    segments = []
    segments.append(keypoints1[matches[:,0]])
    segments.append(keypoints2[matches[:,1]] + np.array([w, 0])[None,:])
    segments = np.concatenate(segments, axis=1)
    img = draw_segments(img, segments, color=(255, 0, 0), thickness=1)
    cv2.imwrite(fname_out, img)
    print("[LOG] Number of matches: {0}. Writing matches visualization to {1}".format(matches.shape[0], fname_out))


