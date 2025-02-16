# discriminator.py
import torch 
import torch.nn as nn
from torch.nn.utils import spectral_norm
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, 1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x) 
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels*2, out_channels=features[0], kernel_size=3, stride=1, padding=1, padding_mode="reflect")
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(ConvBlock(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect"))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)
    print(f"number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

if __name__ == "__main__":
    test()
