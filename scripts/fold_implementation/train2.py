import os
import time
import numpy as np
import torch.amp
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import save_image

from dataset2 import ImageToImageDataset
from generator2 import Generator 
from discriminator2 import Discriminator
from vgg_loss2 import VGGLoss


def main():
    # ======================
    # 1. Hyperparameters
    # ======================
    batch_size = 1
    num_epochs = 20
    learning_rate = 0.0002
    beta1 = 0.9
    save_every = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================
    # 2. Data transformations
    # =====================
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # =========================
    # 3. Load dataset
    # =========================
    val_output_dir = "/kaggle/working/validation_outputs"
    checkpoint_dir = "/kaggle/working/checkpoints"
    os.makedirs(val_output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = ImageToImageDataset(
        root_A='/kaggle/input/watermark-dataset/selected_images/train/train', 
        root_B='/kaggle/input/watermark-dataset/selected_images/train/natural', 
        transform=transform
    )

    val_dataset = ImageToImageDataset(
        root_A='/kaggle/input/watermark-dataset/selected_images/val/train', 
        root_B='/kaggle/input/watermark-dataset/selected_images/val/natural', 
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # =========================
    # 4. Initialize Models
    # =========================
    netG = Generator(in_channels=3).to(device)
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
    criterionVGG = VGGLoss(device).to(device)
    d_scaler = torch.amp.GradScaler()
    g_scaler = torch.amp.GradScaler()

    # =========================
    # 7. Training Loop
    # =========================
    for epoch in range(num_epochs):
        netG.train()
        netD.train()
        loop = tqdm(train_loader, desc=f"epoch [{epoch+1}/{num_epochs}]")
        for i, (img_A, img_B) in enumerate(loop):
            img_A = img_A.to(device)
            img_B = img_B.to(device)

            # Discriminator update
            with torch.amp.autocast('cuda'):
                # Real images
                D_real = netD(img_A, img_B)
                loss_D_real = criterionBCE(D_real, torch.ones_like(D_real, device=device))

                # Fake images
                fake_img_B = netG(img_A)
                D_fake = netD(img_A, fake_img_B.detach())
                loss_D_fake = criterionBCE(D_fake, torch.zeros_like(D_fake, device=device))

                # Total Discriminator Loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            optimizerD.zero_grad()
            d_scaler.scale(loss_D).backward()
            d_scaler.step(optimizerD)
            d_scaler.update()

            # Generator update
            with torch.amp.autocast('cuda'):
                # BCE Loss
                fake_img_B = netG(img_A)
                D_fake = netD(img_A, fake_img_B)
                loss_G_BCE = criterionBCE(D_fake, torch.ones_like(D_fake, device=device))

                # L1 Loss
                loss_G_L1 = criterionL1(fake_img_B, img_B)

                # VGG Loss
                loss_G_VGG = criterionVGG(fake_img_B, img_B)

                # Total Generator Loss
                loss_G = loss_G_BCE + (loss_G_L1 * 100) + (loss_G_VGG * 10)

            optimizerG.zero_grad()
            g_scaler.scale(loss_G).backward()
            g_scaler.step(optimizerG)
            g_scaler.update()

            # Logging
            loop.set_postfix(
                l_D = loss_D.item(),
                l_G = loss_G.item(),
            )

            torch.cuda.empty_cache()
        
        torch.cuda.empty_cache()
        # Checkpoints
        if (epoch + 1) % save_every == 0:
            torch.save(netG.state_dict(), os.path.join(checkpoint_dir, f'netG_epoch_{epoch+1}.pth'))
            torch.save(netD.state_dict(), os.path.join(checkpoint_dir, f'netD_epoch_{epoch+1}.pth'))
        
    # =========================
    # 8. Validation
    # =========================
        netG.eval()
        val_loss_G_L1 = 0
        val_loss_G_VGG = 0
        val_loss_G_BCE = 0
        num_val_batches = len(val_loader)
        val_loop = tqdm(val_loader, desc=f"Validation [{epoch+1}/{num_epochs}]", leave=False)
        with torch.no_grad():
            random_idx = random.randint(0, len(val_loader) - 1)
            img_A_val, _ = list(val_loader)[random_idx]
            img_A_val = img_A_val.to(device)
            with torch.amp.autocast(enabled=False, device_type=device.type):
                generated_img = netG(img_A_val.float())
            save_image(generated_img, os.path.join(val_output_dir, f"epoch_{epoch+1}_generated.png"))

            for img_A_val, img_B_val in val_loop:
                img_A_val = img_A_val.to(device)
                img_B_val = img_B_val.to(device)
                with torch.amp.autocast(enabled=False, device_type=device.type):
                    fake_img_B_val = netG(img_A_val.float())
                val_loss_G_L1 += criterionL1(fake_img_B_val, img_B_val).item()
                val_loss_G_VGG += criterionVGG(fake_img_B_val, img_B_val).item()
                D_fake_val = netD(img_A_val.float(), fake_img_B_val.float())
                val_loss_G_BCE += criterionBCE(D_fake_val, torch.ones_like(D_fake_val, device=device)).item()
                # average val losses
                val_loop.set_postfix(
                    BCE=val_loss_G_BCE / (val_loop.n + 1),
                    L1=val_loss_G_L1 / (val_loop.n + 1),
                    VGG=val_loss_G_VGG / (val_loop.n + 1),
                )
            print(f"BCE_Loss={val_loss_G_BCE / num_val_batches} \nL1_Loss={val_loss_G_L1 / num_val_batches} \nVGG_Loss={val_loss_G_VGG / num_val_batches}")
            torch.cuda.empty_cache()

    # =========================
    # 9. Save Model
    # =========================
    torch.save(netG.state_dict(), os.path.join(checkpoint_dir, 'netG_final.pth'))
    torch.save(netD.state_dict(), os.path.join(checkpoint_dir, 'netD_final.pth'))  


if __name__ == "__main__":
    main()
