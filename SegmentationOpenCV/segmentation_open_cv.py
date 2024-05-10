import os
import cv2
import numpy as np


class Segmentation:
    EMPTY_THRESHOLD = 200
    THREAD_THRESHOLD = 100

    def __init__(self, imgs_path='../DATA_PROJECT', masks_path='../MASKS'):
        self._imgs_path = imgs_path
        self._masks_path = masks_path

    @property
    def imgs_path(self):
        return self._imgs_path

    @imgs_path.setter
    def imgs_path(self, new_path):
        self._imgs_path = new_path

    @property
    def masks_path(self):
        return self._masks_path

    @masks_path.setter
    def masks_path(self, new_path):
        self._masks_path = new_path

    @staticmethod
    def dice_coefficient(mask1, mask2):
        """
        Calculate the Dice coefficient between two binary masks to  measure of the similarity
        between two masks.
        - The input masks should be binary arrays (0s and 1s), where 1 represents the segmented region.
        - The function calculates the Dice coefficient using the formula:
            dice = (2 * intersection) / (total_size)
            where 'intersection' is the sum of overlapping pixels between the two masks, and
            'total_size' is the sum of all pixels in both masks.
        - A value of 1 indicates perfect overlap, while a value of 0 indicates no overlap.
        :param mask1: the first binary mask
        :param mask2: the second binary mask
        :return: The Dice coefficient between the two masks, ranging from 0 to 1.
        """
        if mask1.shape != mask2.shape:
            return 'the input masks are not of the same shape'

        intersection = np.sum(mask1[mask2 == 1])
        total_size = np.sum(mask1) + np.sum(mask2)
        dice = (2.0 * intersection) / total_size
        return dice

    def _create_mask(self, img, class_number):
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

    def segmentation_by_class(self, class_number):
        """
        Perform segmentation evaluation for a specific class using Dice coefficient.

        This function evaluates the segmentation performance for a specified class using the Dice coefficient.
        It loads images and corresponding ground truth masks from predefined directories, applies segmentation
        based on HSV thresholding, and calculates the Dice coefficient between the segmented mask and
        the expected mask.

        - The function assumes that images are stored in the '../DATA_PROJECT/'
          directory and masks in the '../MASKS/' directory.
        - HSV thresholding is used for segmentation, with different thresholds applied based on the specified class.
        - Dice coefficient is calculated using the 'dice_coefficient' function, comparing the segmented mask
          with the expected mask for each image.
        - The returned Dice coefficient represents the similarity between the segmented regions and
          the ground truth.
        :param class_number: The class number for which segmentation evaluation is performed.
        :return: The mean Dice coefficient for the specified class across all images.
        """
        try:
            img_lst = [cv2.imread(f'{self._imgs_path}/{filename}') for filename in os.listdir(f'{self._imgs_path}')]
        except FileNotFoundError:
            return 'This path is incorrect!'
        try:
            expected_mask_lst = [cv2.imread(f'{self._masks_path}/{filename}') for filename in
                                 os.listdir(f'{self._masks_path}')]
        except FileNotFoundError:
            return 'This path is incorrect!'

        dice_coefficient_lst = []

        for img, expected_mask in zip(img_lst, expected_mask_lst):
            res_mask = self._create_mask(img, class_number)
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

            cv2.destroyAllWindows()

            dice_score = self.dice_coefficient(wh_img, res_mask)

            dice_coefficient_lst.append(dice_score)
            # print(dice_score)

        return np.mean(dice_coefficient_lst)

    def get_one_mask(self, img_path, class_number):
        """
        Generates a mask for a single image based on color thresholding in the HSV color space.

        This method takes the path to an image and the class number as input parameters.
        It loads the image, converts it to the HSV color space, and then applies thresholding
        to create a binary mask that corresponds to the specified class.

        :param img_path: String path to the image.
        :param class_number: Numeric index of the class: 0 for gray substance, 1 for threads, and 2 for empty space.
        :return: Binary mask represented as a numpy array, where 1 denotes pixels belonging to the class,
                 and 0 denotes all other pixels.
        """
        img = cv2.imread(img_path)
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


if __name__ == '__main__':
    sgmnt = Segmentation('../DATA_PROJECT', '../MASKS')
    dice_mean_gray_substance = sgmnt.segmentation_by_class(0)
    dice_mean_threads = sgmnt.segmentation_by_class(1)
    dice_mean_empty_space = sgmnt.segmentation_by_class(2)
    print(f'dice mean gray substance: {dice_mean_gray_substance}')
    print(f'dice mean threads: {dice_mean_threads}')
    print(f'dice mean empty space: {dice_mean_empty_space}')
