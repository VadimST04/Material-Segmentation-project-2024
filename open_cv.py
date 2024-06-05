import cv2
import base64
import numpy as np
from SegmentationOpenCV.segmentation import Segmentation


def open_cv(file_to_check):
    segmentation = Segmentation()
    image = np.frombuffer(file_to_check.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Process the image with segmentation
    ternary_mask = segmentation.create_ternary_mask(img)
    return ternary_mask
