import cv2
import numpy as np
from segmentation import Segmentation


class DiceMetrics:
    """
    A class used to calculate the Dice coefficient for measuring the similarity between binary masks.

    The DiceMetrics class provides methods to:
    - Calculate the Dice coefficient between two binary masks.
    - Calculate the average Dice coefficient for each class in a given image
     and expected mask by performing segmentation.
    """

    def dice_coefficient(self, mask1, mask2):
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

    def get_dice_coefficient_by_each_class(self, img, expected_mask):
        """
        Calculate the average Dice coefficient for each class in the given image and expected mask.

        - The function uses a segmentation algorithm to obtain predicted masks for each class in the input image.
        - It also obtains the corresponding expected masks for each class from the expected mask.
        - The Dice coefficient is calculated for each pair of predicted and expected masks.
        - The function returns the average Dice coefficient across all classes.

        :param img: The input image for which the segmentation is performed.
        :param expected_mask: The expected mask containing the ground truth segmentations for each class.
        :return: The average Dice coefficient across all classes.
        """
        sgmt = Segmentation()
        masks = [msk for msk in sgmt.segmentation_by_each_class(img)]
        expected_masks = [e_msk for e_msk in sgmt.get_segmented_answer_mask_by_each_class(expected_mask)]

        dice_coefficients = []
        for mask, expected_mask in zip(masks, expected_masks):
            dice_coefficients.append(self.dice_coefficient(mask, expected_mask))

        # print(np.mean(dice_coefficients))
        return np.mean(dice_coefficients)
