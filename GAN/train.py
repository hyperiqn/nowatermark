# train.py
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
from generator_segformer import GeneratorSU 
from discriminator import Discriminator
from vgg_loss import VGGLoss


def compute_gradient_penalty(discriminator, real_images, fake_images, input_images, device):
    batch_size, channels, height, width = real_images.shape
    epsilon = torch.rand((batch_size, 1, 1, 1), requires_grad=True).to(device)
    epsilon = epsilon.expand_as(real_images)
    interpolation = epsilon * real_images + (1 - epsilon) * fake_images
    interp_logits = discriminator(input_images, interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty


def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, checkpoint_dir, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict()
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(checkpoint_file, netG, netD, optimizerG, optimizerD, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}.")
    return epoch


def main():
    # ======================
    # 1. Hyperparameters
    # ======================
    batch_size = 32 
    num_epochs = 25
    learning_rate = 0.0002
    beta1 = 0.9
    lambda_gp = 10  
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
    val_output_dir = "/data/anirudh/watermark_removal/outputs/val"
    checkpoint_dir = "/data/anirudh/watermark_removal/outputs/checkpoints"
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
    netG = GeneratorSU(in_channels=3).to(device)
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
    # 7. Load checkpoint if available
    # =========================
    start_epoch = 0
    if checkpoint_path:
        start_epoch = load_checkpoint(checkpoint_path, netG, netD, optimizerG, optimizerD, device)

    # =========================
    # 8. Training Loop
    # =========================
    for epoch in range(start_epoch, num_epochs):
        
        netG.train()
        netD.train()
        loop = tqdm(train_loader, desc=f"epoch [{epoch+1}/{num_epochs}]")
        for i, (img_A, img_B) in enumerate(loop):
            img_A = img_A.to(device)
            img_B = img_B.to(device)

            # Discriminator Update(freeze generator)
            for param in netG.parameters():
                param.requires_grad = False
            for param in netD.parameters():
                param.requires_grad = True

            with torch.amp.autocast('cuda'):
                # Real images
                D_real = netD(img_A, img_B)
                loss_D_real = criterionBCE(D_real, torch.ones_like(D_real, device=device))

                # Fake images
                fake_img_B = netG(img_A)
                D_fake = netD(img_A, fake_img_B.detach())
                loss_D_fake = criterionBCE(D_fake, torch.zeros_like(D_fake, device=device))

                # Compute gradient penalty
                gradient_penalty = compute_gradient_penalty(netD, img_B, fake_img_B, img_A, device)

                loss_D = loss_D_real + loss_D_fake + lambda_gp * gradient_penalty

            optimizerD.zero_grad()
            d_scaler.scale(loss_D).backward()
            d_scaler.step(optimizerD)
            d_scaler.update()

            # Generator Update(freeze discriminator)
            for param in netD.parameters():
                param.requires_grad = False
            for param in netG.parameters():
                param.requires_grad = True

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
                loss_D = loss_D.item(),
                loss_G = loss_G.item(),
            )

        
        if (epoch + 1) % save_every == 0:
            save_checkpoint(epoch + 1, netG, netD, optimizerG, optimizerD, checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
        
        # =========================
        # 9. Validation
        # =========================
        netG.eval()
        val_loss_G_L1 = 0
        val_loss_G_VGG = 0
        val_loss_G_BCE = 0
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
                D_fake_val = netD(img_A_val.float(), fake_img_B_val.float())
                val_loss_G_BCE += criterionBCE(D_fake_val, torch.ones_like(D_fake_val, device=device)).item()

                val_loop.set_postfix(
                    BCE=criterionBCE(D_fake_val, torch.ones_like(D_fake_val, device=device)).item(),
                    L1=criterionL1(fake_img_B_val, img_B_val).item(),
                    VGG=criterionVGG(fake_img_B_val, img_B_val).item(),
                )
            print(f"BCE_Loss={val_loss_G_BCE / num_val_batches} \nL1_Loss={val_loss_G_L1 / num_val_batches} \nVGG_Loss={val_loss_G_VGG / num_val_batches}")

    # =========================
    # 10. Save Final Model
    # =========================
    torch.save(netG.state_dict(), os.path.join(checkpoint_dir, 'netG_final.pth'))
    torch.save(netD.state_dict(), os.path.join(checkpoint_dir, 'netD_final.pth'))  


if __name__ == "__main__":
    main()
