import os
import cv2
import pdb

from functions.extract_harris import extract_harris
from functions.extract_descriptors import filter_keypoints, extract_patches
from functions.match_descriptors import match_descriptors
from functions.vis_utils import plot_image_with_keypoints, plot_image_pair_with_matches

# constants
HARRIS_SIGMA = 1.0
HARRIS_K = 0.05
HARRIS_THRESH = 1e-5
MATCHING_RATIO_TEST_THRESHOLD = 0.5

def main_detection():
    IMG_NAME1 = "images/blocks.jpg"
    IMG_NAME2 = "images/house.jpg"

    # Harris corner detection
    img1 = cv2.imread(IMG_NAME1, cv2.IMREAD_GRAYSCALE)
    corners1, C1 = extract_harris(img1, HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)
    plot_image_with_keypoints(os.path.basename(IMG_NAME1[:-4]) + "_harris.png", img1, corners1)

    img2 = cv2.imread(IMG_NAME2, cv2.IMREAD_GRAYSCALE)
    corners2, C2 = extract_harris(img2, HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)
    plot_image_with_keypoints(os.path.basename(IMG_NAME2[:-4]) + "_harris.png", img2, corners2)

def main_matching():
    IMG_NAME1 = "images/I1.jpg"
    IMG_NAME2 = "images/I2.jpg"

    # Harris corner detection
    img1 = cv2.imread(IMG_NAME1, cv2.IMREAD_GRAYSCALE)
    corners1, C1 = extract_harris(img1, HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)
    plot_image_with_keypoints(os.path.basename(IMG_NAME1[:-4]) + "_harris.png", img1, corners1)

    img2 = cv2.imread(IMG_NAME2, cv2.IMREAD_GRAYSCALE)
    corners2, C2 = extract_harris(img2, HARRIS_SIGMA, HARRIS_K, HARRIS_THRESH)
    plot_image_with_keypoints(os.path.basename(IMG_NAME2[:-4]) + "_harris.png", img2, corners2)

    # Extract descriptors
    corners1 = filter_keypoints(img1, corners1, patch_size=9)
    desc1 = extract_patches(img1, corners1, patch_size=9)
    corners2 = filter_keypoints(img2, corners2, patch_size=9)
    desc2 = extract_patches(img2, corners2, patch_size=9)
    # Matching
    matches_ow = match_descriptors(desc1, desc2, "one_way")
    plot_image_pair_with_matches("match_ow.png", img1, corners1, img2, corners2, matches_ow)
    matches_mutual = match_descriptors(desc1, desc2, "mutual")
    plot_image_pair_with_matches("match_mutual.png", img1, corners1, img2, corners2, matches_mutual)
    matches_ratio = match_descriptors(desc1, desc2, "ratio", ratio_thresh=MATCHING_RATIO_TEST_THRESHOLD)
    plot_image_pair_with_matches("match_ratio.png", img1, corners1, img2, corners2, matches_ratio)

def main():
    main_detection()
    pdb.set_trace() # Enter c to continue to matching, q to exit.
    main_matching()

if __name__ == "__main__":
    main()

