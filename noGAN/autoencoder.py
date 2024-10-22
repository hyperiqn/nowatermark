# autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel

class SegFormerEncoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(SegFormerEncoder, self).__init__()
        self.encoder = SegformerModel.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
        self.encoder.config.output_hidden_states = True

    def forward(self, x):
        outputs = self.encoder(x)
        return outputs.hidden_states, outputs.last_hidden_state

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(UNetDecoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(320 + 256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64 + 64, 32, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, encoder_features, input_skip):
        x = self.up1(x)
        x = torch.cat([x, encoder_features[-2]], dim=1)
        x = self.up2(x)
        x = torch.cat([x, encoder_features[-3]], dim=1)
        x = self.up3(x)
        x = torch.cat([x, encoder_features[-4]], dim=1)
        x = self.up4(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.encoder = SegFormerEncoder(in_channels)
        self.decoder = UNetDecoder(in_channels=512, out_channels=out_channels)

    def forward(self, x):
        input_skip = x
        encoder_features, final_output = self.encoder(x)
        x = self.decoder(final_output, encoder_features, input_skip)
        return x

if __name__ == "__main__":
    model = Generator(in_channels=3, out_channels=3) 
    x = torch.randn((1, 3, 256, 256))
    preds = model(x)
    print(preds.shape)
    print(f"number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
