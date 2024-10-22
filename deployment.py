import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from scripts.generator3 import Generator 
from no_gan.autoencoder import AutoEncoder 

st.title("Watermark Removal: Compare GAN and Autoencoder")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])

def infer(image, model, device='cpu'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_img = model(img_tensor).cpu()

    output_img = output_img.squeeze(0)
    denormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    output_img = denormalize(output_img).mul(255).clamp(0, 255).byte()
    output_img_pil = transforms.ToPILImage()(output_img)

    return output_img_pil

def load_gan_model(checkpoint_path):
    gan_model = Generator(in_channels=3)
    gan_model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True))
    return gan_model

def load_autoencoder_model(checkpoint_path):
    autoencoder_model = AutoEncoder(in_channels=3, out_channels=3)  # Replace with your autoencoder model class
    autoencoder_model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True))
    return autoencoder_model

gan_model = load_gan_model("C:/Users/s_ani/Documents/Programming/deeplearning/pix2pixhd/generator.pth")
autoencoder_model = load_autoencoder_model("C:/Users/s_ani/Documents/Programming/deeplearning/pix2pixhd/autoencoder.pth")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', width=256)
    image = Image.open(uploaded_file).convert('RGB')

    gan_result_image = infer(image, gan_model, device='cpu')

    autoencoder_result_image = infer(image, autoencoder_model, device='cpu')

    col2, col3 = st.columns(2)


    with col2:
        st.image(gan_result_image, caption="GAN Result", width=256)
    with col3:
        st.image(autoencoder_result_image, caption="Autoencoder Result", width=256)

st.sidebar.write("Anirudh Swaminathan - 220968280 - DSE A2")
