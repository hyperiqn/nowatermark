import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class ImageToImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None, patch_size=256, stride=128):
        """
        Args:
            root_A (str): Directory with input images.
            root_B (str): Directory with target images.
            transform (callable, optional): Optional transform to be applied on a sample.
            patch_size (int): Size of the patches to extract.
            stride (int): Stride for extracting patches, allowing for overlapping patches.
        """
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride 

        self.images_A = sorted(os.listdir(root_A))
        self.images_B = sorted(os.listdir(root_B))
        assert len(self.images_A) == len(self.images_B), "Input and target directories must have the same number of images."

    def denormalize(self, tensor):
        return (tensor * 0.5) + 0.5

    def denormalize2(self, tensor):
        tensor = (tensor * 0.5) + 0.5
        tensor = tensor.mul(255).clamp(0, 255).byte()
        return tensor

    def __len__(self):
        return len(self.images_A)

    def __getitem__(self, idx):
        # Load images
        img_A = Image.open(os.path.join(self.root_A, self.images_A[idx])).convert('RGB')
        img_B = Image.open(os.path.join(self.root_B, self.images_B[idx])).convert('RGB')

        assert img_A.size == img_B.size, "Input and target images must have the same dimensions."

        patches_A, img_size_A, patch_sizes_A = self.split_into_patches(img_A, self.patch_size, self.stride)
        patches_B, img_size_B, patch_sizes_B = self.split_into_patches(img_B, self.patch_size, self.stride)

        if self.transform:
            patches_A = [self.transform(patch) for patch in patches_A]
            patches_B = [self.transform(patch) for patch in patches_B]

        patches_A = torch.stack(patches_A)
        patches_B = torch.stack(patches_B)

        return patches_A, patches_B, img_size_A, patch_sizes_A

    def split_into_patches(self, img, patch_size, stride):
        """Splits the image into patches with a specified stride (overlap), handling padding for edge patches."""
        width, height = img.size
        patches = []
        patch_sizes = [] 

        for i in range(0, height, stride):
            for j in range(0, width, stride):
                box = (j, i, min(j + patch_size, width), min(i + patch_size, height))
                patch = img.crop(box)

                patch_sizes.append(patch.size)
                
                if patch.size != (patch_size, patch_size):
                    padded_patch = Image.new('RGB', (patch_size, patch_size))
                    padded_patch.paste(patch, (0, 0))
                    patches.append(padded_patch)
                else:
                    patches.append(patch)

        return patches, img.size, patch_sizes 

    def reconstruct_image(self, patches, img_size, patch_sizes, patch_size=256, stride=128):
        """Reconstructs an image from overlapping patches by handling padded areas correctly."""
        width, height = img_size
        reconstructed_img = np.zeros((height, width, 3), dtype=np.float32)
        weight_matrix = np.zeros((height, width), dtype=np.float32)
        
        patch_idx = 0
        for i in range(0, height, stride):
            for j in range(0, width, stride):
                patch = np.array(patches[patch_idx]).astype(np.float32)
                actual_patch_width, actual_patch_height = patch_sizes[patch_idx]

                reconstructed_img[i:i+actual_patch_height, j:j+actual_patch_width, :] += patch[:actual_patch_height, :actual_patch_width, :]
                weight_matrix[i:i+actual_patch_height, j:j+actual_patch_width] += 1 
                
                patch_idx += 1

        reconstructed_img = reconstructed_img / np.maximum(weight_matrix[..., None], 1)

        return Image.fromarray(np.uint8(reconstructed_img))
