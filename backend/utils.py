import os

import cv2
import numpy as np
from PIL import Image
import torch


def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise


def save_image(img_tensor,image_name, c_name, save_dir):
    # Normalize the tensor: [1, C, H, W]
    img_tensor = (img_tensor.clone() + 1) * 0.5 * 255
    img_tensor = img_tensor.cpu().clamp(0, 255)

    # Convert to numpy
    array = img_tensor.detach().numpy().astype('uint8')[0]

    # Convert to PIL image (handling both grayscale and RGB)
    if array.shape[0] == 1:  # Grayscale
        im = Image.fromarray(array.squeeze(0), mode='L')
    else:  # RGB
        array = array.transpose(1, 2, 0)  # Reorder dimensions to HWC
        im = Image.fromarray(array)

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct the file name and path
    image_name = image_name.replace(".jpg", "")
    c_name = c_name.replace(".jpg", "")
    file_name = f"{image_name}-{c_name}.jpg"
    file_path = os.path.join(save_dir, file_name)

    # Save the image
    im.save(file_path)
    print(f"Image saved as {file_name} in {save_dir}")


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path))
