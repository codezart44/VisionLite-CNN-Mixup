import torch
from torch import nn

# Paper: Gradient-Based Learning Applied to Do cument Recognition (Lecun et. al, 1998)

# General structure
# Two convolutional blocks with pooling for feature extraction
# Three fully connected (dense) layers for classification

# Conv Block 1
# Conv Layer: 6 filters, 5x5 kernel, stride 1, padding 2
# Activation: Tanh (originally, ReLU also good otherwise)
# Pooling: Average Pooling (2x2, stride 2)
# size: 28x28 -> 14x14 (from pooling)

# Conv Block 2
# Conv Layer: 16 filters, 5x5 kernel, no padding
# Activation: Tanh
# Pooling: Average Pooling (2x2, stride 2)
# size: 14x14 -> 10x10 (from conv w/o padding)
#       10x10 -> 5x5 (from pooling)

# Flatten
# [B, C, H, W] / [batch_size, channels, height, width]
# [B, 16, 5, 5] -> [B, 16*5*5] = [B, 400]

# Fully Connected
# FC1: 400 -> 120
# Activation: Tanh
# FC2: 120 -> 84
# Activation: Tanh
# FC3: 84 -> 10 (logits for 10 classes)

# NOTE 
# In the original paper Tanh was used for activation as ReLU had not been
# discovered yet. Pooling was also average instead of max. 
# One difference from the original paper is the input size, which for 
# FashionMNIST is 28x28 as opposed to the 32x32 input size used for the
# NIST hand written numbers in the original paper. 

class LeNet5(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,  # Gray Scale
            kernel_size: int = 5,  # LeNet5
            num_classes: int = 10,  # FashionMNIST
            ):
        super().__init__()
        out_channels1 = 6
        out_channels2 = 16

        self.conv_stack = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels, out_channels1, kernel_size, stride=1, padding=2),  # [B, 6, 28, 28]
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # [B, 6, 14, 14]

            # Conv Block 2
            nn.Conv2d(out_channels1, out_channels2, kernel_size, stride=1, padding=0),  # [B, 16, 10, 10], no padding
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # [B, 16, 5, 5]
        )

        out_features1 = 120
        out_features2 = 84

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 16, 5, 5] -> [B, 400]
            nn.Linear(in_features=400, out_features=out_features1),  # [B, 120]
            nn.Tanh(),
            nn.Linear(in_features=out_features1, out_features=out_features2),  # [B, 84]
            nn.Tanh(),
            nn.Linear(in_features=out_features2, out_features=num_classes),  # [B, 10] (logits)
        )
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)
