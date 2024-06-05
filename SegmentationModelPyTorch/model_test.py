import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp
import torchvision
import matplotlib.pyplot as plt


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMG_PATH = "../SegmentationPyTorchPredict/photo_2024-05-19_23-24-37.jpg"
CHECKPOINT_PATH = "./SegmentationModelPyTorch/mycheckpoint.pth.tar"
SAVE_DIR = "C:/LearningProjects/Material-Segmentation-project-2024/Material-Segmentation-project-2024/SegmentationModelPyTorch/test_predictions_no_dice/"

COLORS = [1, 125, 255]


class TestDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, os.path.basename(self.image_path)


def get_transforms():
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )
    return test_transform


def get_model(in_channels=3, out_channels=3):
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_channels,
    )
    return model


def load_checkpoint(checkpoint_path, model, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])


def apply_color_map(pred):
    color_mask = np.zeros((pred.shape[0], pred.shape[1]), dtype=np.uint8)

    for class_id, color in enumerate(COLORS):
        color_mask[pred == class_id] = color

    return Image.fromarray(color_mask)


def make_predictions(loader, model, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    loop = tqdm(loader)

    for data, img_names in loop:
        data = data.to(device=DEVICE)
        with torch.no_grad():
            preds = torch.argmax(torch.softmax(model(data), dim=1), dim=1).float()

        for pred, img_name in zip(preds, img_names):
            pred_np = pred.squeeze().cpu().numpy()
            pred_colored = apply_color_map(pred_np)
            base_name = os.path.basename(img_name).replace(".jpg", ".png")
            pred_path = os.path.join(save_dir, f"pred_{base_name}")
            pred_colored.save(pred_path)

    model.train()


def pytorch(file_to_check):
    test_transform = get_transforms()
    model = get_model(in_channels=3, out_channels=3).to(DEVICE)

    load_checkpoint(CHECKPOINT_PATH, model, DEVICE)

    temp_filename = 'temp_image.jpg'
    file_to_check.save(temp_filename)

    test_ds = TestDataset(image_path=temp_filename, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False)

    make_predictions(test_loader, model, SAVE_DIR)

    pred_image_path = os.path.join(SAVE_DIR, f"pred_{os.path.basename(temp_filename).replace('.jpg', '.png')}")

    os.remove(temp_filename)

    if not os.path.exists(pred_image_path):
        raise FileNotFoundError(f"Predicted image not found at {pred_image_path}")

    return pred_image_path


if __name__ == '__main__':
    file_to_check = Image.open(TEST_IMG_PATH)  # Open the image file here
    pred_image_path = pytorch(file_to_check)
    print(f"Predicted image saved at: {pred_image_path}")
