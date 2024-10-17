import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from generator2 import Generator 

def infer(image_path, checkpoint_path, output_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    netG = Generator(in_channels=3).to(device)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            generated_img = netG(img_tensor).cpu()

    generated_img = generated_img.squeeze(0) 
    denormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]) 
    generated_img = denormalize(generated_img).mul(255).clamp(0, 255).byte() 

    generated_img_pil = transforms.ToPILImage()(generated_img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generated_img_pil.save(output_path)
    
    print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    infer(
        image_path="C:/Users/s_ani/Documents/Programming/deeplearning/pix2pixhd/selected_images/val/train/COCO_val2014_000000039814-Crocs_Logo-152.png", 
        checkpoint_path="C:/Users/s_ani/Downloads/netG_final.pth", 
        output_path="generated_image.png", 
        device='cuda'
    )
