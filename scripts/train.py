# train.py

import os
import time
import numpy as np
import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import save_image

from dataset import ImageToImageDataset
from generator import Generator 
from discriminator import Discriminator
from vgg_loss import VGGLoss


def main():
    # ======================
    # 1. Hyperparameters
    # ======================

    batch_size = 1 
    num_epochs = 100
    learning_rate = 0.0002
    beta1 = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================
    # 2. Data Transformations
    # =====================

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # =========================
    # 3. Load the Dataset
    # =========================
    val_output_dir = "validation_outputs"
    checkpoint_dir = "checkpoints"
    os.makedirs(val_output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = ImageToImageDataset(
        root_A='path_to_train_w', 
        root_B='path_to_train_nw', 
        transform=transform,
        patch_size=256,
        stride=128
    )

    val_dataset = ImageToImageDataset(
        root_A='path_to_val_w', 
        root_B='path_to_val_nw', 
        transform=transform,
        patch_size=256,
        stride=128
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # =========================
    # 4. Initialize Models
    # =========================

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # =========================
    # 5. Define Optimizers
    # =========================

    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # =========================
    # 6. Define Loss Functions and GradScaler
    # =========================

    criterionBCE = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()

    # =========================
    # 7. Training Loop
    # =========================

    for epoch in range(num_epochs):
        netG.train()
        netD.train()
        for i, data in enumerate(tqdm(train_loader, desc=f"epoch [{epoch+1}/{num_epochs}]")):
            patches_A, patches_B, img_size_A, patch_sizes_A = data

            patches_A = patches_A.squeeze(0)
            patches_B = patches_B.squeeze(0)

            patches_A = patches_A.to(device)
            patches_B = patches_B.to(device)

            # Discriminator update
            with torch.cuda.amp.autocast():
                # Real patches
                D_real = netD(patches_A, patches_B)
                loss_D_real = criterionBCE(D_real, torch.ones_like(D_real, device=device))

                # Fake patches
                fake_patches_B = netG(patches_A)
                D_fake = netD(patches_A, fake_patches_B.detach())
                loss_D_fake = criterionBCE(D_fake, torch.zeros_like(D_fake, device=device))

                # Total Discriminator Loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            optimizerD.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(optimizerD)
            scaler.update()

            # Generator update
            with torch.cuda.amp.autocast():
                # BCE Loss
                fake_patches_B = netG(patches_A)
                D_fake = netD(patches_A, fake_patches_B)
                loss_G_BCE = criterionBCE(D_fake, torch.ones_like(D_fake, device=device))

                # L1 Loss
                loss_G_L1 = criterionL1(fake_patches_B, patches_B)

                # VGG Loss
                loss_G_VGG = criterionVGG(fake_patches_B, patches_B)

                # Total Generator Loss
                loss_G = loss_G_BCE + (loss_G_L1 * 100) + (loss_G_VGG * 10)

            optimizerG.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(optimizerG)
            scaler.update()

            # Logging
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                    f"D Loss: {loss_D.item()}, G Loss: {loss_G.item()}")

        # Checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(netG.state_dict(), os.path.join(checkpoint_dir, f'netG_epoch_{epoch+1}.pth'))
            torch.save(netD.state_dict(), os.path.join(checkpoint_dir, f'netD_epoch_{epoch+1}.pth'))

    # =========================
    # 8. Validation
    # =========================

        netG.eval()
        with torch.no_grad():
            random_idx = random.randint(0, len(val_loader) - 1)
            val_data = list(val_loader)[random_idx]
            patches_A_val, _, img_size_val, patch_sizes_val = val_data
            patches_A_val = patches_A_val.squeeze(0).to(device)

            with torch.cuda.amp.autocast():
                generated_patches = netG(patches_A_val)

            generated_patches = generated_patches.cpu()
            generated_patches_pil = [transforms.ToPILImage()(val_dataset.denormalize2(patch)) for patch in generated_patches]
            reconstructed_image = val_dataset.reconstruct_image(generated_patches_pil, img_size_val, patch_sizes_val)
            reconstructed_image.save(os.path.join(val_output_dir, f"epoch_{epoch+1}_image.png"))  


    torch.save(netG.state_dict(), os.path.join(checkpoint_dir, 'netG_final.pth'))
    torch.save(netD.state_dict(), os.path.join(checkpoint_dir, 'netD_final.pth'))  


if __name__ == "__main__":
    main()


