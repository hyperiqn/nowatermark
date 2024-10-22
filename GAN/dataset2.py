import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageToImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        """
        Args:
            root_A (str): Directory with input images.
            root_B (str): Directory with target images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

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
        img_A = Image.open(os.path.join(self.root_A, self.images_A[idx])).convert('RGB')
        img_B = Image.open(os.path.join(self.root_B, self.images_B[idx])).convert('RGB')

        assert img_A.size == img_B.size, "Input and target images must have the same dimensions."

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B
