# test_dataset.py
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from dataset import ImageToImageDataset
from PIL import Image
import os

def denormalize(tensor): # for grid of patches
    tensor = (tensor * 0.5) + 0.5
    return tensor

def denormalize2(tensor): # for reconstructed image
    tensor = (tensor * 0.5) + 0.5 
    tensor = tensor.mul(255).clamp(0, 255).byte()
    return tensor

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

dataset = ImageToImageDataset(root_A='C:/Users/s_ani/Documents/Programming/deeplearning/pix2pixhd/wmnwm/w',
                              root_B='C:/Users/s_ani/Documents/Programming/deeplearning/pix2pixhd/wmnwm/nw',
                              transform=transform,
                              patch_size=256,
                              stride=128)

patches_A, patches_B, img_size_A, patch_sizes_A = dataset[0]

save_dir = "test_patches"
os.makedirs(save_dir, exist_ok=True)

nrow = (img_size_A[0] - 256) // 128 + 1

# save patches
torchvision.utils.save_image(denormalize(patches_A), os.path.join(save_dir, "input_patches.png"), nrow=nrow)
torchvision.utils.save_image(denormalize(patches_B), os.path.join(save_dir, "target_patches.png"), nrow=nrow)

# Reconstruct and save input image (w)
patches_A_pil = [transforms.ToPILImage()(denormalize2(patch)) for patch in patches_A]
reconstructed_img_A = dataset.reconstruct_image(patches_A_pil, img_size_A, patch_sizes_A, patch_size=256, stride=128)
reconstructed_img_A.save(os.path.join(save_dir, "reconstructed_input_w.png"))

# Reconstruct and save target image (nw)
patches_B_pil = [transforms.ToPILImage()(denormalize2(patch)) for patch in patches_B]
reconstructed_img_B = dataset.reconstruct_image(patches_B_pil, img_size_A, patch_sizes_A, patch_size=256, stride=128)
reconstructed_img_B.save(os.path.join(save_dir, "reconstructed_target_nw.png"))

print(f"Patches and reconstructed images saved in '{save_dir}'.")