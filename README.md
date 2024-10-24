Watermark Removal using GANs
Author: Anirudh Swaminathan
Roll No: 220968280
Course: DSE A2

Project Overview
This project focuses on using Generative Adversarial Networks (GANs) for watermark removal from images while preserving the original image quality. Watermarks are often used for intellectual property protection, but in some cases, removing them may be necessary for improving clarity or algorithm testing. This project employs insights from various deep learning research papers to achieve effective watermark removal with minimal quality degradation.

Dataset
Dataset: CLWD (Colored Large-scale Watermark Dataset) by Yang Liu, Zhen Zhu, and Xiang Bai.
Number of Watermarks: 200 colored watermarks.
Training Data: 60,000 images (50,000 for training, 10,000 for validation).
Testing Data: 10,000 images.
Model Architecture
Generator:
Encoder: Pre-trained Segformer (Mit-B5).
Decoder: UNet.
Discriminator:
Architecture: 70x70 PatchGAN.
Loss Functions
Adversarial Loss: Binary Cross-Entropy.
Reconstruction Loss: L1 Loss.
Perceptual Loss: Pre-trained VGG19.
Evaluation Metrics
PSNR (Peak Signal-to-Noise Ratio): Target value above 30 dB.
SSIM (Structural Similarity Index): Target value above 0.85.
Results
Model	Average SSIM	Average PSNR
SegformerUNet GAN	0.81507	27.87968
Auto-encoder	0.81917	27.93482
Pix2PixHD GAN	0.94623	30.16776
While Pix2PixHD achieved the best results based on metrics, the SegformerUNet GAN was found to handle watermark removal more effectively in many cases, particularly when handling fine details.# nowatermark
