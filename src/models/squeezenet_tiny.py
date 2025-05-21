import torch
import torch.nn as nn
from typing import Literal

# Paper : SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE (Iandola et al., 2016)

# SqueezeNet employs three main strategies: (1) Uses 1x1 conv filters instead of 3x3 
# (9x fewer parameters), (2) Decrease the number of input channels to 3x3 filters, (3)
# Delayes downsampling and pooling in the architecture to keep large feature maps 
# - has been shown to increase accuracy (He & Sun, 2015). 

# The main component of SqueezeNet is the Fire Module which comprises of a squeeze layer
# and an expansion layer. The squeeze layer denoted by s_(1x1) includes 1x1 filters
# and the expansion layer denoted e_(1x1) and e_(3x3) includes both 1x1 and 3x3 filters
# and concatenates the result from e1 and e3 into one output. 

# General Architecute
#--------------------





# Feature map shapes [B,C,H,W], where Dout is num filters, Din is num channels
# B - Batch size
# C - Channels (same as filter in channels)
# HxW - resolution dimensions

# Filter shapes [O,I,F,F], where Dout is num filters, Din is num channels
# O - base_channels (out channels)
# I - in_channels
# FxF - kernel_size (filter size)
# Input and feature map notation: N@HxW  (6@28x28 meaning 6 feature maps 28x28)

class SqueezeLayer(nn.Module):
    def __init__(
            self,
            in_channels : int,
            base_channels : int,
            ) -> None:
        super().__init__()

        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.squeeze(x)


class ExpandLayer(nn.Module):
    def __init__(
            self,
            base_channels : int,
            kernel_size : Literal[1, 3] = 1,
            downsample : bool = False,
            ) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2

        self.expand = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=base_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(x)


class FireModule(nn.Module):
    def __init__(
            self,
            in_channels : int,
            base_channels : int,
            downsample : bool = False
            ) -> None:
        super().__init__()

        self.squeeze1x1 = SqueezeLayer(in_channels, base_channels)
        self.expand1x1 = ExpandLayer(base_channels, kernel_size=1, downsample=downsample)
        self.expand3x3 = ExpandLayer(base_channels, kernel_size=3, downsample=downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze1x1(x)
        # NOTE Due to concat along channel dim, out_channels = 2 * base_channels 
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1) 


class SqueezeNetTiny(nn.Module):
    def __init__(
            self,
            in_channels : int = 1,  # Gray Scale
            base_channels : int = 16,
            num_classes : int = 10,  # FashionMNIST
            dropout : float = 0.2,
            ) -> None:
        super().__init__()

        self.conv_stack = nn.Sequential(  # -> [B, 1, 28, 28]
            # initial conv layer
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),  # [B, 16, 28, 28]
            nn.ReLU(inplace=True),
            # -- Skip the maxpool here, keep spatial dims large early on
            FireModule(base_channels, base_channels),       # [B, 32, 28, 28]
            FireModule(base_channels*2, base_channels*2),   # [B, 64, 28, 28]
            nn.MaxPool2d(kernel_size=2, stride=2),          # [B, 64, 14, 14]
            FireModule(base_channels*4, base_channels*4),   # [B, 128, 14, 14]
            # nn.MaxPool2d(kernel_size=2, stride=2),          # [B, 128, 7, 7]
            # FireModule(base_channels*8, base_channels*8),   # [B, 256, 7, 7]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=1),  # [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128, 1, 1] -> [B, 128]
            nn.Dropout(p=dropout),
            nn.Linear(in_features=base_channels*8, out_features=num_classes),  # [B, 10]
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)


def main():
    t = torch.rand((1, 1, 28, 28))
    sqzn = SqueezeNetTiny(base_channels=8)
    t = sqzn(t)
    print(t.shape) # Output [1, 10]

if __name__=='__main__':
    main()
