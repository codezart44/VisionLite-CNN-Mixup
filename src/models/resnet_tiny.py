import torch
from torch import nn

# Paper: Deep Residual Learning for Image Recognition (He et. al, 2015)

# Instead of letting the network learn the full transformation H(x) to 
# the output y, residual networks only lets the network learn the residual
# part F(x), so that y = x + F(x). 

# This allows for optimization, especially in deeper networks as it is easier to
# learn an identity (small change F(x) ~ 0) than the full mapping.
# Gradient do not suffer from the vanishing gradient problem through 
# skip paths. 

# basic ResNet block w. skip connections
# y = x + F(x)   <- NOTE This trick makes training deeper networks much easier.
# Block: Output is x -> Conv -> BN -> ReLU -> Conv -> BN -> Add original x -> ReLU
# Stack a few of these blocks

# All Convs typically use kernel size 3, padding 1, stride 1

class _ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downsample: bool = False,
            ):
        super().__init__()

        # With stride set to 2, the dimension lengths of the image are halved. 
        stride = 2 if downsample else 1  # For downsampling

        # Optimised to compute the residual of the input x instead of the 
        # transformation of x
        self.res_stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # Used to reshape input x to same dimensions as residual output 
        # (for addition F(x) + x). Make input x congruent to residual F(x). 
        self.skip = nn.Identity()
        if downsample or out_channels != in_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        residual = self.res_stack(x)
        return self.relu(identity + residual)


# MiniResNet - A lightweight version of the original ResNet architecture. 
# Consists of:
# - Initial convolutional layer to transform input to have expected number of channels. 
# - Residual stages. Typically three. Each stage consists of two ResidualBlocks, 
#   first one doing downsampling (stride=2) and the other does not. 
#   During each stage, the number of channels doubles and the resolution
#   dimensions are halved (downsampled). Giving the network a cone-like shape when
#   visualized. Starting wide but thin and then narrowing off while becoming thicker. 
#       Stage 1. [B, 16, 28, 28] -> [B, 16, 28, 28]
#       Stage 2. [B, 16, 28, 28] -> [B, 32, 14, 14]
#       Stage 3. [B, 32, 14, 14] -> [B, 64, 7, 7]
# - Average Global Pooling layer. [B, 64, 7, 7] -> [B, 64, 1, 1]
# - Flattening layer. [B, 64, 1, 1] -> [B, 64*1*1] = [B, 64]
# - Fully connected layer. [B, 64] -> [B, 10], since there are 10 FashionMNIST classes. 

class ResNetTiny(nn.Module):
    def __init__(
            self,
            in_channels : int = 1,  # Gray Scale
            base_channels : int = 16,
            num_classes : int = 10,  # FashionMNIST
            dropout : float = 0.2,
            ):
        super().__init__()

        self.conv_stack = nn.Sequential(
            # Initial Convolutional layer
            nn.Conv2d(in_channels, base_channels, kernel_size=1, stride=1, padding=0),

            # Residual Stage 1
            _ResidualBlock(base_channels, base_channels, downsample=False),  # [B, 16, 28, 28]
            _ResidualBlock(base_channels, base_channels, downsample=False),  # [B, 16, 28, 28]

            # Residual Stage 2
            _ResidualBlock(base_channels, base_channels*2, downsample=True),   # [B, 32, 14, 14]
            _ResidualBlock(base_channels*2, base_channels*2, downsample=False),  # [B, 32, 14, 14]

            # Residual Stage 3
            _ResidualBlock(base_channels*2, base_channels*4, downsample=True),   # [B, 64, 7, 7]
            _ResidualBlock(base_channels*4, base_channels*4, downsample=False),  # [B, 64, 7, 7]
        )

        self.classifier = nn.Sequential(
            # Takes Mean along resolution dimensions (H, W)
            nn.AdaptiveAvgPool2d(output_size=1),  # [B, 64, 7, 7] -> [B, 64, 1, 1]
            nn.Flatten(),  # [B, 64, 1, 1] -> [B, 64]
            nn.Dropout(p=dropout),  # Dropout for regularisation
            nn.Linear(in_features=base_channels*4, out_features=num_classes),  # [B, 64] -> [B, 10]
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)
