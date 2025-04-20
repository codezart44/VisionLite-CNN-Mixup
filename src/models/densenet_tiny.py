import torch
import torch.nn as nn

# Paper: Densely Connected Convolutional Networks (Huang et. al., 2018)

# DenseNet is short for Densely Connected Convolutional Network
# Unlike ResNet which adds the residual to the input, DenseNet instead
# concatenates the outputs

# out = Concat(x0, x1, ..., xn)

# Each layer receives the outputs of all previous layers as input. 

# Dense Layer:
# ------------
# - BatchNorm -> ReLU -> Conv2D (3x3)
# Outputs shape is fixed (e.g. 12 channels) (growth rate, i.e. 
# input increases by this many channels each block)
# The input to the layer is the concatenation of all previous 
# layer outputs (feature maps). Thus features are accumulated 

# Dense Block: 
# ------------
# - Sequence of dense blocks
# The output channels of each block increases linearly.
# - in_channels -> in_channels + growth_rate * L (after L layers)

# Transition Layer:
# -----------------
# Used between dense blocks to downsample and reduce the complexity
# of the features maps. Typically:
# - 1x1 Conv layer (compress number of channels, out < in)
# - AvgPool2d (reduce length of resolution dimensions)

# Global Pool + Classifier:
# -------------------------
# Apply global pooling, flatten and then map to class logits

GROWTH_RATE = 12
COMPRESSION = 4

class DenseLayer(nn.Module):
    def __init__(
            self,
            in_channels : int,
            growth_rate : int = GROWTH_RATE,
            dropout : float = 0.2,
            ):
        super().__init__()

        self.dense_stack = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_stack(x)


# Each dense layer in a dense block concatenates a new feature map
# to the original input. E.g. if the input has: 
# - in_channels = 24
# - growth_rate = 12
# - num_layers = 3      # Num dense layers in block
# => out_channels = 24+3*12 = 60. 

# The output of the last dense layer is concatenated with the last
# input and then used as input for the next dense layer. 

class DenseBlock(nn.Module):
    def __init__(
            self,
            in_channels : int,
            growth_rate : int = GROWTH_RATE,
            num_layers : int = 3,
            ):
        super().__init__()
        self.growth_rate = growth_rate

        # List of #num dense layers
        self.dense_layers = nn.ModuleList(
            DenseLayer(
                in_channels = in_channels + i*growth_rate, 
                growth_rate = growth_rate, 
                dropout = 0.2
            ) for i in range(num_layers)
        )
    
    def __repr__(self):
        return f'DenseBlock(num_layers={len(self.dense_layers)}, growth_rate={self.growth_rate})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dense in self.dense_layers:
            feat_map = dense(x)  # [B, growth_rate, H, W]
            # Concat feature maps along channel dimension.
            # Takes last input and concats new dense output to it.
            # Thus output channels grows by grow_rate each iteration.
            x = torch.cat([x, feat_map], dim=1)  # [B, in_channels+i*growth_rate, H, W]
        return x  # [B, in_channels + growth_rate * num_layers, H, B]

# The transition layer is used to compress the input between 
# dense blocks. It does so using both:
# - Conv layer: Compressing number of channels with 1x1 kernel
# - AvgPool layer: Halving length of the resolution dimensions
# [B, C, H, W] -> [B, compression*C, H/2, W/2]

class TransitionLayer(nn.Module):
    def __init__(
            self,
            in_channels : int,
            compression : int = COMPRESSION,
            ):
        super().__init__()

        self.trans_stack = nn.Sequential(
            # Compresses to half the number of channels
            # [B, C, H, W] -> [B, C/compression, H, W]
            nn.Conv2d(in_channels, int(in_channels/compression), kernel_size=1, stride=1, padding=0),
            # Compresses resolution dimesions to half
            # [B, C/compression, H, W] -> [B, C/compression, H/2, W/2]
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trans_stack(x)
    
# Tiny (shallow) Version of original DenseNet
# Consists of:
# - Initial Conv2d layer
# - Dense Block 1
# - Transition Layer 1
# - Dense Block 2
# - Transition Layer 2
# - Dense Block 3
# - Global Pooling Layer
# - Flatten Layer
# - Dropout
# - Linear Layer

class DenseNetTiny(nn.Module):
    def __init__(
            self,
            in_channels : int = 1,  # Gray Scale
            base_channels : int = 16,
            num_classes : int = 10,  # FashionMNIST
            compression : int = COMPRESSION,
            growth_rate : int = GROWTH_RATE,
            ):
        super().__init__()
        num_layers = 3

        in_channels_d1 = base_channels
        in_channels_t1 = in_channels_d1 + growth_rate * num_layers
        in_channels_d2 = int(in_channels_t1 / compression)
        in_channels_t2 = in_channels_d2 + growth_rate * num_layers
        in_channels_d3 = int(in_channels_t2 / compression)
        in_channels_lin = in_channels_d3 + growth_rate * num_layers

        self.conv_stack = nn.Sequential(
            # Initial Conv Layer
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),  # [B, 12, 28, 28]

            # Dense Block 1
            DenseBlock(in_channels_d1, growth_rate, num_layers),  # [B, 12+12*3, 28, 28] = [B, 48, 28, 28]

            # Transision Layer 1
            TransitionLayer(in_channels_t1, compression),  # [B, 48/4, 28/2, 28/2] = [B, 12, 14, 14]

            # Dense Block 2
            DenseBlock(in_channels_d2, growth_rate, num_layers),  # [B, 12+12*3, 14, 14] = [B, 48, 14, 14]

            # Transition Layer 2
            TransitionLayer(in_channels_t2, compression),  # [B, 48/4, 14/2, 14/2] = [B, 12, 7, 7]
            
            # Dense Block 3
            DenseBlock(in_channels_d3, growth_rate, num_layers)  # [B, 12+12*3, 7, 7] = [B, 48, 7, 7]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),  # [B, 48, 7, 7] -> [B, 48, 1, 1]
            nn.Flatten(),  # [B, 48, 1, 1] -> [B, 48]
            nn.Dropout(p=0.2),
            nn.Linear(in_channels_lin, num_classes)  # [B, 48] -> [B, 10]
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)
