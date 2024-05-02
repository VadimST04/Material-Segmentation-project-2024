import os
import cv2
import numpy as np

EMPTY_THRESHOLD = 200
THREAD_THRESHOLD = 100


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


def segmentation(class_number):
    """
    Perform segmentation evaluation for a specific class using Dice coefficient.

    This function evaluates the segmentation performance for a specified class using the Dice coefficient.
    It loads images and corresponding ground truth masks from predefined directories, applies segmentation
    based on HSV thresholding, and calculates the Dice coefficient between the segmented mask and the expected mask.

    - The function assumes that images are stored in the '../DATA_PROJECT/'
      directory and masks in the '../MASKS/' directory.
    - HSV thresholding is used for segmentation, with different thresholds applied based on the specified class.
    - Dice coefficient is calculated using the 'dice_coefficient' function, comparing the segmented mask
      with the expected mask for each image.
    - The returned Dice coefficient represents the similarity between the segmented regions and the ground truth.
    :param class_number: The class number for which segmentation evaluation is performed.
    :return: The mean Dice coefficient for the specified class across all images.
    """
    try:
        img_lst = [cv2.imread(f'../DATA_PROJECT/{filename}') for filename in os.listdir('../DATA_PROJECT')]
    except FileNotFoundError:
        return 'This path is incorrect!'
    try:
        expected_mask_lst = [cv2.imread(f'../MASKS/{filename}') for filename in os.listdir('../MASKS')]
    except FileNotFoundError:
        return 'This path is incorrect!'

    dice_coefficient_lst = []

    for img, expected_mask in zip(img_lst, expected_mask_lst):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        res_mask = np.zeros(shape=img.shape, dtype=np.uint8)
        wh_img = np.zeros(shape=expected_mask.shape, dtype=np.uint8)

        match class_number:
            case 0:
                wh_img[expected_mask == 1] = 0
                wh_img[expected_mask == 2] = 0
                wh_img[expected_mask == 0] = 1
            case 1:
                wh_img[expected_mask == 1] = 1
                wh_img[expected_mask == 2] = 0
                wh_img[expected_mask == 0] = 0
            case 2:
                wh_img[expected_mask == 1] = 0
                wh_img[expected_mask == 2] = 1
                wh_img[expected_mask == 0] = 0

        _, thresh1 = cv2.threshold(hsv[:, :, 2], EMPTY_THRESHOLD, 255, cv2.THRESH_BINARY)  # empty
        _, thresh2 = cv2.threshold(hsv[:, :, 2], THREAD_THRESHOLD, 255, cv2.THRESH_BINARY)  # threads
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

        cv2.destroyAllWindows()

        dice_score = dice_coefficient(wh_img, res_mask)

        dice_coefficient_lst.append(dice_score)
        # print(dice_score)

    return np.mean(dice_coefficient_lst)


if __name__ == '__main__':
    dice_mean_gray_substance = segmentation(0)
    dice_mean_threads = segmentation(1)
    dice_mean_empty_space = segmentation(2)
    print(f'dice mean gray substance: {dice_mean_gray_substance}')
    print(f'dice mean threads: {dice_mean_threads}')
    print(f'dice mean empty space: {dice_mean_empty_space}')
