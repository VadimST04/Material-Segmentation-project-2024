from SegmentationOpenCV.segmentation import Segmentation


def open_cv():
    image_path = './static/images/one.png'

    segmentation = Segmentation()
    with open(image_path, 'rb') as image:
        image = segmentation.create_ternary_mask(image)
        print(image)
        return image


open_cv()
