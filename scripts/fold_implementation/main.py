import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import final_train_loader, final_val_loader
from models.generator import SegFormerUNetGenerator
from models.discriminator import Discriminator
import os
from torchvision.utils import save_image
import random


disc_optical = Discriminator(in_channels=3).to("cuda")
disc_SAR = Discriminator(in_channels=1).to("cuda")
gen_optical_to_SAR = SegFormerUNetGenerator(in_channels=3, out_channels=1).to("cuda")
gen_SAR_to_optical = SegFormerUNetGenerator(in_channels=1, out_channels=3).to("cuda")
opt_disc = torch.optim.Adam(
    list(disc_optical.parameters()) + list(disc_SAR.parameters()),
    lr=0.0002,
    betas=(0.5, 0.999),
)
opt_gen = torch.optim.Adam(
    list(gen_optical_to_SAR.parameters()) + list(gen_SAR_to_optical.parameters()),
    lr=0.0002,
    betas=(0.5, 0.999),
)

L1 = nn.L1Loss()
MSE = nn.MSELoss()

g_scaler = torch.amp.GradScaler('cuda')
d_scaler = torch.amp.GradScaler('cuda')

EPOCHS = 100
SAVE_MODEL = True
CHECKPOINT_GEN_H = "gen_optical_to_SAR.pth"
CHECKPOINT_GEN_Z = "gen_SAR_to_optical.pth"
CHECKPOINT_CRITIC_H = "critic_optical.pth"
CHECKPOINT_CRITIC_Z = "critic_SAR.pth"

# Load models
for epoch in range(EPOCHS):
    loop = tqdm(final_train_loader)
    for idx, (SAR, optical) in enumerate(loop):
        optical = optical.to("cuda")
        SAR = SAR.to("cuda")

        # Train Discriminator SAR
        with torch.amp.autocast('cuda'):
            fake_SAR = gen_optical_to_SAR(optical)
            D_SAR_real = disc_SAR(SAR)
            D_SAR_fake = disc_SAR(fake_SAR.detach())
            D_SAR_real_loss = MSE(D_SAR_real, torch.ones_like(D_SAR_real))
            D_SAR_fake_loss = MSE(D_SAR_fake, torch.zeros_like(D_SAR_fake))
            D_SAR_loss = D_SAR_real_loss + D_SAR_fake_loss

        disc_optical.zero_grad()
        d_scaler.scale(D_SAR_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Discriminator Optical
        with torch.amp.autocast('cuda'):
            fake_optical = gen_SAR_to_optical(SAR)
            D_optical_real = disc_optical(optical)
            D_optical_fake = disc_optical(fake_optical.detach())
            D_optical_real_loss = MSE(D_optical_real, torch.ones_like(D_optical_real))
            D_optical_fake_loss = MSE(D_optical_fake, torch.zeros_like(D_optical_fake))
            D_optical_loss = D_optical_real_loss + D_optical_fake_loss

        disc_optical.zero_grad()
        d_scaler.scale(D_optical_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
       # Train Generator
        with torch.amp.autocast('cuda'):
    # Adversarial loss for both generators
            D_SAR_fake = disc_SAR(fake_SAR)
            D_optical_fake = disc_optical(fake_optical)

            # Adversarial loss for both SAR to optical and optical to SAR
            loss_G_SAR = MSE(D_SAR_fake, torch.ones_like(D_SAR_fake))  
            loss_G_optical = MSE(D_optical_fake, torch.ones_like(D_optical_fake))

            # Cycle consistency loss for both directions
            cycle_SAR = gen_optical_to_SAR(fake_optical)
            cycle_optical = gen_SAR_to_optical(fake_SAR)

            cycle_SAR_loss = L1(SAR, cycle_SAR)  # SAR -> optical -> SAR
            cycle_optical_loss = L1(optical, cycle_optical)  # optical -> SAR -> optical

            # Identity loss for both directions (optional)
            identity_SAR = gen_SAR_to_optical(SAR)
            identity_optical = gen_optical_to_SAR(optical)

            identity_SAR_loss = L1(SAR, identity_SAR)
            identity_optical_loss = L1(optical, identity_optical)

            # Final loss with equal weight on both directions
            G_loss = (
                loss_G_optical
                + loss_G_SAR
                + cycle_optical_loss * 10  # Equal weight on cycle losses
                + cycle_SAR_loss * 10
                + identity_optical_loss * 5  # Equal weight on identity losses
                + identity_SAR_loss * 5
            )

        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_path = f"saved_models/{epoch}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(gen_optical_to_SAR.state_dict(), f"{save_path}/gen_optical_to_SAR.pth")
            torch.save(gen_SAR_to_optical.state_dict(), f"{save_path}/gen_SAR_to_optical.pth")
            torch.save(disc_optical.state_dict(), f"{save_path}/disc_optical.pth")
            torch.save(disc_SAR.state_dict(), f"{save_path}/disc_SAR.pth")
            print("Model saved")
        loop.set_postfix(
            D_SAR_loss=D_SAR_loss.item(),
            D_optical_loss=D_optical_loss.item(),
            G_loss=G_loss.item(),
        )
    
    # Validation
    val_loop = tqdm(final_val_loader)
    random_idx = random.randint(0, len(final_val_loader) - 1)
    with torch.no_grad():  # Disable gradients for validation
        for idx, (SAR, optical) in enumerate(val_loop):
            optical = optical.to("cuda")
            SAR = SAR.to("cuda")

            with torch.amp.autocast('cuda'):
                # Generate fake SAR and optical images
                fake_SAR = gen_optical_to_SAR(optical)
                fake_optical = gen_SAR_to_optical(SAR)

                # Save a few generated images to track progress
                if idx == random_idx:  # Save images for the first batch
                    save_image_dir = f"generated_images/epoch_{epoch}"
                    if not os.path.exists(save_image_dir):
                        os.makedirs(save_image_dir)

                    # Save fake SAR and fake optical images
                    save_image(fake_SAR, f"{save_image_dir}/fake_SAR_epoch_{epoch}.png")
                    save_image(fake_optical, f"{save_image_dir}/fake_optical_epoch_{epoch}.png")

                # Discriminator predictions on real and fake images
                D_SAR_real = disc_SAR(SAR)
                D_SAR_fake = disc_SAR(fake_SAR)
                D_optical_real = disc_optical(optical)
                D_optical_fake = disc_optical(fake_optical)

                # Compute generator loss on fake images (adversarial loss)
                loss_G_SAR = MSE(D_SAR_fake, torch.ones_like(D_SAR_fake))
                loss_G_optical = MSE(D_optical_fake, torch.ones_like(D_optical_fake))

                # Cycle consistency loss
                cycle_SAR = gen_optical_to_SAR(fake_optical)
                cycle_optical = gen_SAR_to_optical(fake_SAR)
                cycle_SAR_loss = L1(SAR, cycle_SAR)
                cycle_optical_loss = L1(optical, cycle_optical)

                # Identity loss (optional)
                identity_SAR = gen_SAR_to_optical(SAR)
                identity_optical = gen_optical_to_SAR(optical)
                identity_SAR_loss = L1(SAR, identity_SAR)
                identity_optical_loss = L1(optical, identity_optical)

                # Final generator loss for validation
                G_loss_val = (
                    loss_G_optical
                    + loss_G_SAR
                    + cycle_optical_loss * 10
                    + cycle_SAR_loss * 10
                    + identity_optical_loss * 5
                    + identity_SAR_loss * 5
                )

                # Discriminator losses (SAR and optical)
                D_SAR_real_loss = MSE(D_SAR_real, torch.ones_like(D_SAR_real))
                D_SAR_fake_loss = MSE(D_SAR_fake, torch.zeros_like(D_SAR_fake))
                D_SAR_loss_val = D_SAR_real_loss + D_SAR_fake_loss

                D_optical_real_loss = MSE(D_optical_real, torch.ones_like(D_optical_real))
                D_optical_fake_loss = MSE(D_optical_fake, torch.zeros_like(D_optical_fake))
                D_optical_loss_val = D_optical_real_loss + D_optical_fake_loss

            # Print losses for monitoring during validation
            val_loop.set_postfix(
                D_SAR_loss_val=D_SAR_loss_val.item(),
                D_optical_loss_val=D_optical_loss_val.item(),
                G_loss_val=G_loss_val.item(),
            )

    # Save model checkpoints

    if SAVE_MODEL:
        save_model_dir = "saved_models"
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        torch.save(gen_optical_to_SAR.state_dict(), f"{save_model_dir}/{CHECKPOINT_GEN_H}")
        torch.save(gen_SAR_to_optical.state_dict(), f"{save_model_dir}/{CHECKPOINT_GEN_Z}")
        torch.save(disc_optical.state_dict(), f"{save_model_dir}/{CHECKPOINT_CRITIC_H}")
        torch.save(disc_SAR.state_dict(), f"{save_model_dir}/{CHECKPOINT_CRITIC_Z}")
        print("Model saved")

    print("Epoch completed")
    print("Model saved")

    
        