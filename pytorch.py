from PIL import Image
import torchvision.transforms as transforms
import base64


def pytorch(file_to_check):
    image_path = './static/images/two.jpg'

    with open(image_path, 'rb') as image:
        encoded_string = base64.b64encode(image.read()).decode('utf-8')
        return encoded_string
