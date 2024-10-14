import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def visualize_original_and_reconstructed(original, reconstructed):
    """Visualize the original and reconstructed images side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    original_img = original[0].permute(1, 2, 0).detach().cpu().numpy()  # Convert to HWC format
    axs[0].imshow((original_img - original_img.min()) / (original_img.max() - original_img.min()))
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    # Reconstructed image
    reconstructed_img = reconstructed[0].permute(1, 2, 0).detach().cpu().numpy()  # Convert to HWC format
    axs[1].imshow((reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min()))
    axs[1].set_title("Reconstructed Image")
    axs[1].axis('off')
    
    plt.show()

def test_unfold_fold_only():
    x = torch.randn((1, 3, 512, 512))  # Test with a random larger image
    batch_size, channels, height, width = x.shape
    patch_size = 256
    stride = 128  # Overlap of 50%

    # Unfold input into patches
    patches = F.unfold(x, kernel_size=(patch_size, patch_size), stride=(stride, stride))
    
    # Reshape back to the unfolded format
    num_patches = patches.shape[-1]
    patches = patches.view(batch_size * num_patches, channels, patch_size, patch_size)

    # Fold patches back to original image
    output_patches = patches.view(batch_size, -1, num_patches)  # Reshape for fold
    output_image = F.fold(output_patches, output_size=(height, width), kernel_size=(patch_size, patch_size), stride=(stride, stride))

    # Visualize original and reconstructed image
    visualize_original_and_reconstructed(x, output_image)

# Test unfold/fold
test_unfold_fold_only()
