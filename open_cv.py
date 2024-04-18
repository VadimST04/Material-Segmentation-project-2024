import cv2
import base64


def open_cv(file_to_check):
    image_path = './static/images/one.png'

    with open(image_path, 'rb') as image:
        encoded_string = base64.b64encode(image.read()).decode('utf-8')
        return encoded_string
