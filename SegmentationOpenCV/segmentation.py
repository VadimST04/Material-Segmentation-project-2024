import cv2
import numpy as np


class Segmentation:
    """
    A class used to perform image segmentation and generate binary masks for different classes.

    The Segmentation class provides methods to:
    - Generate binary masks for specific classes from an expected mask.
    - Perform segmentation on an input image to create masks for each class based on color thresholding in the HSV color space.

    Attributes:
    EMPTY_THRESHOLD (int): Threshold value used to identify empty spaces in the HSV color space.
    THREAD_THRESHOLD (int): Threshold value used to identify threads in the HSV color space.
    """
    EMPTY_THRESHOLD = 200
    THREAD_THRESHOLD = 100

    def get_binary_answer_mask(self, expected_mask, class_number):
        """
        Generate a binary mask for a specific class from the expected mask.

        This method takes an expected mask and a class number as input and returns a binary mask where
        pixels belonging to the specified class are set to 1 and all other pixels are set to 0.

        :param expected_mask: The input expected mask where different classes are represented by different integer values.
        :param class_number: The class number to generate the mask for:
                             0 for gray substance, 1 for threads, and 2 for empty space.
        :return: A binary mask with the same shape as the expected mask, where 1 represents the pixels
                 belonging to the specified class, and 0 represents all other pixels.
        """
        wh_img = np.zeros(shape=expected_mask.shape, dtype=np.uint8)

        match class_number:
            case 0:  # gray substance
                wh_img[expected_mask == 1] = 0
                wh_img[expected_mask == 2] = 0
                wh_img[expected_mask == 0] = 1
            case 1:  # threads
                wh_img[expected_mask == 1] = 1
                wh_img[expected_mask == 2] = 0
                wh_img[expected_mask == 0] = 0
            case 2:  # empty space
                wh_img[expected_mask == 1] = 0
                wh_img[expected_mask == 2] = 1
                wh_img[expected_mask == 0] = 0

        return wh_img

    def get_segmented_answer_mask_by_each_class(self, expected_mask):
        """
        Generate binary masks for each class from the expected mask.

        This method returns a tuple of binary masks for each class (gray substance, threads, and empty space)
        based on the input expected mask.

        :param expected_mask: The input expected mask where different classes are represented by different integer values.
        :return: A tuple containing three binary masks (one for each class: gray substance, threads, and empty space).
        """
        answer_mask1_gray_substance = self.get_binary_answer_mask(expected_mask, 0)
        answer_mask2_threads = self.get_binary_answer_mask(expected_mask, 1)
        answer_mask3_empty_space = self.get_binary_answer_mask(expected_mask, 2)

        return answer_mask1_gray_substance, answer_mask2_threads, answer_mask3_empty_space

    def create_binary_mask(self, img, class_number):
        """
        Creates a mask based on color thresholding in the HSV color space.

        This method takes an image and the class number as input parameters.
        It converts the image to the HSV color space, applies thresholding,
        and generates a binary mask according to the specified class.

        :param img: Input image in BGR format.
        :param class_number: Numeric index of the class: 0 for gray substance, 1 for threads, and 2 for empty space.
        :return: Binary mask represented as a numpy array, where 1 denotes pixels belonging to the class,
                 and 0 denotes all other pixels.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        res_mask = np.zeros(shape=img.shape, dtype=np.uint8)
        _, thresh1 = cv2.threshold(hsv[:, :, 2], self.EMPTY_THRESHOLD, 255, cv2.THRESH_BINARY)  # empty
        _, thresh2 = cv2.threshold(hsv[:, :, 2], self.THREAD_THRESHOLD, 255, cv2.THRESH_BINARY)  # threads
        match class_number:
            case 0:  # gray substance
                res_mask[thresh1 == 0] = 1
                res_mask[thresh2 == 0] = 0
            case 1:  # threads
                res_mask[thresh1 != 0] = 1
                res_mask[thresh2 == 0] = 0
            case 2:  # empty space
                res_mask[thresh1 != 0] = 0
                res_mask[thresh2 == 0] = 1

        return res_mask

    def segmentation_by_each_class(self, img):
        """
        Perform segmentation for each class in the input image.

        This method generates binary masks for each class (gray substance, threads, and empty space)
        by applying color thresholding in the HSV color space.

        :param img: Input image in BGR format.
        :return: A tuple containing three binary masks (one for each class: gray substance, threads, and empty space).
        """
        mask1_gray_substance = self.create_binary_mask(img, 0)
        mask2_threads = self.create_binary_mask(img, 1)
        mask3_empty_space = self.create_binary_mask(img, 2)

        return mask1_gray_substance, mask2_threads, mask3_empty_space

    def create_ternary_mask(self, img):
        """
        Create a ternary mask based on color thresholding in the HSV color space.

        This method takes an image as input, converts it to the HSV color space, applies thresholding,
        and generates a ternary mask where pixels are labeled as 0, 1, or 2 based on the following criteria:
        - Pixels with values below the EMPTY_THRESHOLD are labeled as 1.
        - Pixels with values above the THREAD_THRESHOLD are labeled as 0.
        - Pixels with values between the EMPTY_THRESHOLD and THREAD_THRESHOLD are labeled as 2.

        :param img: Input image in BGR format.
        :return: Ternary mask represented as a numpy array, where pixels are labeled as 0, 1, or 2
                 based on the defined thresholds.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        res_mask = np.zeros(shape=img.shape, dtype=np.uint8)
        _, thresh1 = cv2.threshold(hsv[:, :, 2], self.EMPTY_THRESHOLD, 255, cv2.THRESH_BINARY)  # empty
        _, thresh2 = cv2.threshold(hsv[:, :, 2], self.THREAD_THRESHOLD, 255, cv2.THRESH_BINARY)  # threads

        res_mask[thresh1 != 0] = 1
        res_mask[thresh2 == 0] = 2

        # Map the ternary mask values to grayscale values
        color_map = {0: 0, 1: 125, 2: 255}
        for key, value in color_map.items():
            res_mask[res_mask == key] = value

        return res_mask
