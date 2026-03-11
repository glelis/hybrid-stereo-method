
import cv2
import cv2.xfeatures2d
import numpy as np
from natsort import natsorted
from utils import *


def compute_descriptors(imGray):
    """
    Computes SIFT keypoints and descriptors for a grayscale image.

    Args:
        imGray: A grayscale image represented as a NumPy array.

    Returns:
        keypoints: A list of detected keypoints.
        descriptors: A NumPy array containing the computed descriptors.
    """

    # Create SIFT object for keypoint detection and descriptor calculation
    sift = cv2.SIFT.create()
    # Detect keypoints and compute descriptors using SIFT
    keypoints, descriptors = sift.detectAndCompute(imGray, None)

    # Print number of detected keypoints and descriptor array shape
    print(f"keypoints: {len(keypoints)}, descriptors: {descriptors.shape}")

    return keypoints, descriptors


def create_matcher(trees, checks):
    """
    Creates a cv2.FlannBasedMatcher object for feature matching.

    Args:
        trees: Number of trees in the KD-Tree data structure used for fast nearest neighbor search.
        checks: Number of checks performed during the matching search. Higher values improve accuracy but increase processing time.

    Returns:
        matcher: A cv2.FlannBasedMatcher object configured for feature matching using the FLANN algorithm.
    """

    # Define KD-Tree index type
    FLANN_INDEX_KDTREE = 0

    # Parameters for building the KD-Tree index
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)

    # Parameters for the matching search
    search_params = dict(checks=checks)

    # Create matcher object using defined parameters
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    return matcher


def find_good_matches_loc(matcher, keypoints1, descriptors1, keypoints2, descriptors2, factor):
    """
    Finds good quality matches between two sets of keypoints and descriptors, and returns their locations.

    Args:
        matcher: A cv2.FlannBasedMatcher object configured for feature matching.
        keypoints1: A list of keypoints from the first image.
        descriptors1: A NumPy array containing the descriptors of the first image.
        keypoints2: A list of keypoints from the second image.
        descriptors2: A NumPy array containing the descriptors of the second image.
        factor: A threshold factor used to filter ambiguous matches. Lower factor results in stricter filtering.

    Returns:
        good_matches: A list of good quality matches between the two images.
        points1: A NumPy array containing the coordinates of the corresponding keypoints in the first image.
        points2: A NumPy array containing the coordinates of the corresponding keypoints in the second image.
    """

    # Find the two nearest neighbors for each descriptor in the first image
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Initialize a list to store good matches
    good_matches = []

    # Apply Lowe's ratio test to filter ambiguous matches
    for m, n in matches:
        if (
            m.distance < factor * n.distance
        ):  # Keep matches where the distance to the nearest neighbor is significantly smaller than the distance to the second nearest neighbor
            good_matches.append(m)

    # Extract coordinates of corresponding keypoints in both images
    points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(
        -1, 1, 2
    )
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(
        -1, 1, 2
    )

    return good_matches, points1, points2


def apply_homography(img1, img2, points1, points2):
    """
    Aligns 'img1' with 'img2' using a homography transformation calculated from corresponding points.

    Args:
        img1: The image to be aligned.
        img2: The reference image for alignment.
        points1: A NumPy array containing the coordinates of points in 'img1'.
        points2: A NumPy array containing the coordinates of corresponding points in 'img2'.

    Returns:
        aligned_img: 'img1' aligned with 'img2' using the homography transformation.
    """

    # Get dimensions of the reference image
    height, width, channels = img2.shape

    # Compute homography matrix using RANSAC for robustness against outliers
    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Apply perspective transformation to 'img1' using the computed homography
    aligned_img = cv2.warpPerspective(img1, homography, (width, height))

    return aligned_img


def align_im1_to_im2(img1, img2):
    """
    Aligns 'img1' with 'img2' using feature matching and homography.

    Args:
        img1: The image to be aligned.
        img2: The reference image for alignment.

    Returns:
        imMatches: An image showing the matches found between the two images.
        aligned_img: 'img1' aligned with 'img2'.
    """

    # Convert images to grayscale
    img1Gray = img1
    img2Gray = img2

    # Compute keypoints and descriptors for both images
    keypoints1, descriptors1 = compute_descriptors(img1Gray)
    keypoints2, descriptors2 = compute_descriptors(img2Gray)

    # Create matcher object for feature matching
    matcher = create_matcher(trees=5, checks=50)

    # Find good quality matches and their locations
    good_matches, points1, points2 = find_good_matches_loc(
        matcher, keypoints1, descriptors1, keypoints2, descriptors2, factor=0.80
    )

    # Draw matches found in an image
    imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)

    # Apply homography to align 'img2' with 'img1'
    aligned_img = apply_homography(img2, img1, points2, points1)

    return imMatches, aligned_img


def main_align(base_path):
    """
    Main function that performs image alignment using a global reference image.

    Args:
        base_path: The base directory containing the input images.

    """

    img_path = base_path + "imagens/"
    save_path = base_path + "output/align_images/aligned/"
    match_path = base_path + "output/align_images/match_save/"

    # Find all files in the image folder and sort them naturally
    all_files = find_all_files(img_path)
    all_files = natsorted(all_files)
    print(all_files)

    if len(all_files) == 0:
        print("No images found.")
        return

    # Use the middle image as the global reference
    ref_idx = len(all_files) // 2
    reference_img_path = img_path + all_files[ref_idx]
    print(f"Reading global reference image: {reference_img_path}")
    reference_img = read_image(reference_img_path)
    
    # Save the reference image directly to the aligned folder
    ref_save_as = "align_" + str(ref_idx) + ".jpg"
    save_image(save_path, ref_save_as, reference_img, 0, 255)

    # Iterate over the files, aligning each image with the reference image
    for i in range(len(all_files)):
        if i == ref_idx:
            continue
            
        target_img_path = img_path + all_files[i]
        match_save_as = "matches_" + str(i) + ".jpg"
        align_save_as = "align_" + str(i) + ".jpg"

        print("Reading a target image : ", target_img_path)
        target_img = read_image(target_img_path)

        print("Aligning image to global reference ...")
        imMatches, aligned_img = align_im1_to_im2(reference_img, target_img)

        print("Saving a feature matching image : ", match_path)
        save_image(match_path, match_save_as, imMatches, 0, 255)

        print("Saving an aligned image : ", save_path)
        save_image(save_path, align_save_as, aligned_img, 0, 255)

        print("\n")
