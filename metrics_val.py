# metrics_val.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import numpy as np
from dataset import ImageToImageDataset
from generator_segformer import Generator
import os

def compute_ssim(img1, img2):
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)
    height, width, _ = img1.shape
    win_size = min(7, height, width)
    return ssim(img1, img2, win_size=win_size, channel_axis=-1)

def main():

    batch_size = 32 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])


    val_dataset = ImageToImageDataset(
        root_A='/data/anirudh/watermark_removal/CLWD_images/val/Watermark_image', 
        root_B='/data/anirudh/watermark_removal/CLWD_images/val/Watermark_free_image', 
        transform=transform
    )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    checkpoint_path = "/data/anirudh/watermark_removal/output_no_gan/checkpoints/netG_final.pth"
    netG = Generator(in_channels=3).to(device)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    netG.eval()

    total_ssim = 0
    total_psnr = 0
    num_images = 0

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc="Calculating SSIM and PSNR")
        for i, (img_A, img_B) in enumerate(val_loop):
            img_A = img_A.to(device)
            img_B = img_B.to(device)

            fake_img_B = netG(img_A)

            img_B_denorm = (img_B * 0.5) + 0.5 
            fake_img_B_denorm = (fake_img_B * 0.5) + 0.5

            for j in range(img_A.size(0)):
                img_B_single = img_B_denorm[j]
                fake_img_B_single = fake_img_B_denorm[j]

                ssim_val = compute_ssim(fake_img_B_single, img_B_single)
                total_ssim += ssim_val

                psnr_val = psnr(fake_img_B_single, img_B_single, data_range=1.0)
                total_psnr += psnr_val.item()

                num_images += 1

            val_loop.set_postfix(
                avg_ssim=total_ssim / num_images,
                avg_psnr=total_psnr / num_images,
            )

    avg_ssim = total_ssim / num_images
    avg_psnr = total_psnr / num_images

    print(f"Final Average SSIM: {avg_ssim}")
    print(f"Final Average PSNR: {avg_psnr}")

if __name__ == "__main__":
    main()
