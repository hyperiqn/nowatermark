import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from dataset import ImageToImageDataset
from generator import Generator 

def infer(image_path, checkpoint_path, output_path, patch_size=256, stride=128, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    img = Image.open(image_path).convert('RGB')
    dataset = ImageToImageDataset(root_A='', root_B='', patch_size=patch_size, stride=stride, transform=transform)
    patches, img_size, patch_sizes = dataset.split_into_patches(img, patch_size, stride)
    patches = torch.stack([transform(patch) for patch in patches]).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            generated_patches = netG(patches).cpu()

    generated_patches_pil = [transforms.ToPILImage()(dataset.denormalize2(patch)) for patch in generated_patches]
    reconstructed_image = dataset.reconstruct_image(generated_patches_pil, img_size, patch_sizes)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    reconstructed_image.save(output_path)
    print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    infer("input.png", "checkpoints/netG_final.pth", "output/generated_image.png", patch_size=256, stride=128, device='cuda')
