import os
import numpy as np
import cv2

from dice_metrics import DiceMetrics

if __name__ == '__main__':
    print("===================for one===================")
    img = cv2.imread("../../DATA_PROJECT/2_3_1_R_cut_128_256.tif")
    expected_mask = cv2.imread("../../MASKS/2_3_1_R_cut_128_256.png")
    dm = DiceMetrics()

    # only for 2_3_1_R_cut_128_256.tif
    result_for_one = dm.get_dice_coefficient_by_each_class(img, expected_mask)
    print(result_for_one)

    # only for all
    print("===================for all===================")
    img_lst = [cv2.imread(f'../../DATA_PROJECT/{filename}') for filename in os.listdir(f'../../DATA_PROJECT')]
    expected_mask_lst = [cv2.imread(f'../../MASKS/{filename}') for filename in os.listdir(f'../../MASKS')]

    dice_coefs = []
    for i in range(len(img_lst)):
        dice_coefs.append(dm.get_dice_coefficient_by_each_class(img_lst[i], expected_mask_lst[i]))
    result_for_all = np.mean(dice_coefs)
    print(result_for_all)
