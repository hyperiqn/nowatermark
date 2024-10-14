import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=False, padding_mode="reflect")
        self.in2d = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.ReLU() if use_act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.in2d(self.cnn(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.survival_prob = 0.8
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=True)

    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, x):
        out = self.block2(self.block1(x))
        return self.stochastic_depth(out) + x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, down=True, act="relu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_channels, features=64, num_residuals=9):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 7, 1, 3, bias=True, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

        self.down1 = Block(features, features*2, act="relu")
        self.down2 = Block(features*2, features*4, act="relu")
        self.down3 = Block(features*4, features*8, act="relu")
        self.down4 = Block(features*8, features*16, act="relu")
        self.residuals = nn.Sequential(*[ResidualBlock(in_channels=features*16) for _ in range(num_residuals)])
        self.up1 = Block(features*16, features*8, stride=1, act="relu")
        self.up2 = Block(features*8*2, features*4, stride=1, act="relu")
        self.up3 = Block(features*4*2, features*2, stride=1, act="relu")
        self.up4 = Block(features*2*2, features, stride=1, act="relu")
        self.final_conv = nn.Sequential(
            Block(features*2, features, stride=1, act="relu"),
            Block(features, features, stride=1, act="relu"),
            nn.Conv2d(features, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        patch_size = 128
        stride = 128
        
        patches = F.unfold(x, kernel_size=(patch_size, patch_size), stride=(stride, stride)) 
        
        num_patches = patches.shape[-1]
        patches = patches.permute(0, 2, 1).view(batch_size * num_patches, channels, patch_size, patch_size)  

        d1 = self.initial_down(patches)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        residuals = self.residuals(d5) + d5
        up1 = self.up1(F.interpolate(residuals, scale_factor=2, mode="nearest"))
        up2 = self.up2(F.interpolate(torch.cat([up1, d4], dim=1), scale_factor=2, mode="nearest"))
        up3 = self.up3(F.interpolate(torch.cat([up2, d3], dim=1), scale_factor=2, mode="nearest"))
        up4 = self.up4(F.interpolate(torch.cat([up3, d2], dim=1), scale_factor=2, mode="nearest"))
        output_patches = self.final_conv(torch.cat([up4, d1], dim=1))

        output_patches = output_patches.view(batch_size, -1, num_patches).permute(0, 2, 1)

        output_image = F.fold(output_patches, output_size=(height, width), kernel_size=(patch_size, patch_size), stride=(stride, stride))
        
        return output_image


def test():
    x = torch.randn((1, 3, 512, 512))  # Test with a larger image
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
