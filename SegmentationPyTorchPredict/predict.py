import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from SegmentationPyTorchPredict.utils.data_loading import BasicDataset
from SegmentationPyTorchPredict.unet import UNet

# Set up logging to a file
log_file_path = r'C:\LearningProjects\Material-Segmentation-project-2024\Material-Segmentation-project-2024\SegmentationPyTorchPredict\logs.txt'
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=log_file_path, filemode='w')


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    logging.info("Preprocessing the image for prediction")
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    logging.info("Running the prediction on the network")
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    logging.info("Prediction completed")
    return mask[0].long().squeeze().numpy()


def get_model_and_device(model_path='./SegmentationPyTorchPredict/CurrentModel.pth', n_classes=3, bilinear=False):
    logging.info("Initializing the UNet model")
    net = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_path}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    logging.info("Loading the state dictionary for the model")
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")
    return net, device, mask_values


def mask_to_image(mask: np.ndarray, mask_values):
    logging.info("Converting mask to image format")
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # Update the mask values if there are three classes
    if len(mask_values) == 3:
        mask_values = [1, 125, 255]

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def pytorch_image_models(image_file):
    net, device, mask_values = get_model_and_device()

    logging.info("Opening the input image")
    img = Image.open(image_file)

    logging.info("Predicting the mask for the image")
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=0.5,
                       out_threshold=0.5,
                       device=device)

    logging.info("Converting the mask to an image file")
    result = mask_to_image(mask, mask_values)
    return result