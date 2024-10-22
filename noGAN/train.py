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

from dataset import ImageToImageDataset
from autoencoder import Generator 
from vgg_loss import VGGLoss


# =========================
# Compute Gradient Penalty
# =========================

def save_checkpoint(epoch, netG, optimizerG, checkpoint_dir, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(checkpoint_file, netG, optimizerG, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}.")
    return epoch


def main():
    # ======================
    # 1. Hyperparameters
    # ======================
    batch_size = 32 
    num_epochs = 50
    learning_rate = 0.0002
    beta1 = 0.7
    save_every = 5  
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
    val_output_dir = "/data/anirudh/watermark_removal/output_no_gan/val"
    checkpoint_dir = "/data/anirudh/watermark_removal/output_no_gan/checkpoints"
    checkpoint_path = None
    os.makedirs(val_output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = ImageToImageDataset(
        root_A='/data/anirudh/watermark_removal/CLWD_images/train/Watermark_image', 
        root_B='/data/anirudh/watermark_removal/CLWD_images/train/Watermark_free_image', 
        transform=transform
    )
    
    val_dataset = ImageToImageDataset(
        root_A='/data/anirudh/watermark_removal/CLWD_images/val/Watermark_image', 
        root_B='/data/anirudh/watermark_removal/CLWD_images/val/Watermark_free_image', 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # =========================
    # 4. Initialize Models
    # =========================
    netG = Generator(in_channels=3).to(device)

    # =========================
    # 5. Define Optimizers
    # =========================
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # =========================
    # 6. Define Loss Functions and GradScaler
    # =========================
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(device).to(device)
    g_scaler = torch.amp.GradScaler()

    # =========================
    # 7. Load checkpoint if available
    # =========================
    start_epoch = 0
    if checkpoint_path:
        start_epoch = load_checkpoint(checkpoint_path, netG, optimizerG, device)

    # =========================
    # 8. Training Loop
    # =========================
    for epoch in range(start_epoch, num_epochs):
        
        netG.train()
        loop = tqdm(train_loader, desc=f"epoch [{epoch+1}/{num_epochs}]")
        for i, (img_A, img_B) in enumerate(loop):
            img_A = img_A.to(device)
            img_B = img_B.to(device)

            with torch.amp.autocast('cuda'):
                fake_img_B = netG(img_A)

                # L1 Loss
                loss_G_L1 = criterionL1(fake_img_B, img_B)

                # VGG Loss
                loss_G_VGG = criterionVGG(fake_img_B, img_B)

                # Total Generator Loss
                loss_G = (loss_G_L1 * 100) + (loss_G_VGG * 10)

            optimizerG.zero_grad()
            g_scaler.scale(loss_G).backward()
            g_scaler.step(optimizerG)
            g_scaler.update()

            # Logging
            loop.set_postfix(
                l1_loss = loss_G_L1.item(),
                vgg_loss = loss_G_VGG.item(),
            )

        
        if (epoch + 1) % save_every == 0:
            save_checkpoint(epoch + 1, netG, optimizerG, checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
        
        # =========================
        # 9. Validation
        # =========================
        netG.eval()
        val_loss_G_L1 = 0
        val_loss_G_VGG = 0
        num_val_batches = len(val_loader)
        val_loop = tqdm(val_loader, desc=f"Validation [{epoch+1}/{num_epochs}]", leave=False)
        with torch.no_grad():
            random_idx = random.randint(0, len(val_loader) - 1)
            img_A_val, img_B_val = list(val_loader)[random_idx]
            img_A_val = img_A_val.to(device)
            img_B_val = img_B_val.to(device)
            with torch.amp.autocast(enabled=False, device_type=device.type):
                generated_img = netG(img_A_val.float())
            save_image(img_A_val, os.path.join(val_output_dir, f"epoch_{epoch+1}_original.png"))
            save_image(img_B_val, os.path.join(val_output_dir, f"epoch_{epoch+1}_ground.png"))
            save_image(generated_img, os.path.join(val_output_dir, f"epoch_{epoch+1}_generated.png"))

            for img_A_val, img_B_val in val_loop:
                img_A_val = img_A_val.to(device)
                img_B_val = img_B_val.to(device)
                with torch.amp.autocast(enabled=False, device_type=device.type):
                    fake_img_B_val = netG(img_A_val.float())
                val_loss_G_L1 += criterionL1(fake_img_B_val, img_B_val).item()
                val_loss_G_VGG += criterionVGG(fake_img_B_val, img_B_val).item()

                val_loop.set_postfix(
                    L1_loss=criterionL1(fake_img_B_val, img_B_val).item(),
                    VGG_loss=criterionVGG(fake_img_B_val, img_B_val).item(),
                )
            print(f"L1_Loss={val_loss_G_L1 / num_val_batches} \nVGG_Loss={val_loss_G_VGG / num_val_batches}")

    # =========================
    # 10. Save Final Model
    # =========================
    torch.save(netG.state_dict(), os.path.join(checkpoint_dir, 'netG_final.pth'))


if __name__ == "__main__":
    main()
