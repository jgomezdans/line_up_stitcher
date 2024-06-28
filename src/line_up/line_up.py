import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple


class ImageStitcher:
    def __init__(
        self, image_paths: List[Union[Path, str]], normalize: bool = False
    ):
        """
        Initialise the ImageStitcher class

        Args:
            image_paths (List[Union[Path, str]]): List of paths to 
            image files.
            normalize (bool): Whether to normalize images before
            processing. Default is False.
        """
        self.image_paths = [Path(path) for path in image_paths]
        self.normalize = normalize
        self.homographies = []

    def load_image(self, path: Path, grayscale: bool = True) -> np.ndarray:
        """
        Load an image from the specified path. You may want to override this
        method with your own loader.

        Args:
            path (Path): Path to the image file.
            greyscale (bool): Whether to load the image in grey. 
                Default is True.

        Returns:
            np.ndarray: Loaded image.
        """
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        return cv2.imread(str(path), flag)

    def detect_and_compute_keypoints(self, image: np.ndarray):
        """
        Detect keypoints and compute descriptors using ORB.

        Args:
            image (np.ndarray): Input image.

        Returns:
            tuple: Keypoints and descriptors.
        """
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray):
        """
        Match descriptors using BFMatcher.

        Args:
            desc1 (np.ndarray): Descriptors from the first image.
            desc2 (np.ndarray): Descriptors from the second image.

        Returns:
            list: Sorted list of matches.
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        return sorted(matches, key=lambda x: x.distance)

    def extract_points(self, matches, kp1, kp2):
        """
        Extract matched points from keypoints.

        Args:
            matches (list): List of matched keypoints.
            kp1 (list): Keypoints from the first image.
            kp2 (list): Keypoints from the second image.

        Returns:
            tuple: Points from the first and second image.
        """
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        return points1, points2

    def compute_homography(self, points1: np.ndarray, points2: np.ndarray):
        """
        Compute homography matrix using RANSAC.Note that in some cases, you
        may want to use a different method for finding the homography if
        RANSAC fails. Also, you may want to filter out poorly fitted points
        that might degrade the whole fitting.

        Args:
            points1 (np.ndarray): Points from the first image.
            points2 (np.ndarray): Points from the second image.

        Returns:
            tuple: Homography matrix and mask.
        """
        return cv2.findHomography(points2, points1, cv2.RANSAC)

    def warp_image(
        self, image: np.ndarray, homography: np.ndarray, shape: tuple
    ):
        """
        Warp image using the homography matrix.

        Args:
            image (np.ndarray): Image to be warped.
            homography (np.ndarray): Homography matrix.
            shape (tuple): Desired output shape.

        Returns:
            np.ndarray: Warped image.
        """
        return cv2.warpPerspective(image, homography, shape)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image to the range [0, 255].

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Normalized image.
        """
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_image.astype(np.uint8)

    def process_images(self):
        """
        Process the list of images, stacking them into a single
        superposed image.

        Returns:
            np.ndarray: The final superposed image.
        """
        if not self.image_paths:
            raise ValueError("No images provided.")

        # Load the first image
        base_image = self.load_image(self.image_paths[0], grayscale=False)
        if self.normalize:
            base_image = self.normalize_image(base_image)

        height, width, _ = base_image.shape
        superposed_image = base_image.copy()

        # Process each subsequent image
        for path in self.image_paths[1:]:
            next_image = self.load_image(path, grayscale=False)
            if self.normalize:
                next_image = self.normalize_image(next_image)

            keypoints1, descriptors1 = self.detect_and_compute_keypoints(
                cv2.cvtColor(superposed_image, cv2.COLOR_BGR2GRAY)
            )
            keypoints2, descriptors2 = self.detect_and_compute_keypoints(
                cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
            )

            matches = self.match_descriptors(descriptors1, descriptors2)
            points1, points2 = self.extract_points(
                matches, keypoints1, keypoints2
            )
            homography, _ = self.compute_homography(points1, points2)

            self.homographies.append(homography)
            warped_image = self.warp_image(
                next_image, homography, (width, height)
            )
            superposed_image = cv2.addWeighted(
                superposed_image, 0.5, warped_image, 0.5, 0
            )

        return superposed_image

    def save_homographies(self, file_path: Union[Path, str]):
        """
        Save the computed homographies to a file.

        Args:
            file_path (Union[Path, str]): Path to the file where 
                homographies will be saved.
        """
        file_path = Path(file_path)
        np.save(file_path, self.homographies)

    def load_homographies(self, file_path: Union[Path, str]):
        """
        Load homographies from a file.

        Args:
            file_path (Union[Path, str]): Path to the file from 
                where homographies will be loaded.
        """
        file_path = Path(file_path)
        self.homographies = np.load(file_path, allow_pickle=True).tolist()
