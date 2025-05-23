import torch
import torch.nn as nn

# Paper: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (Andrew et. al., 2017)

# MobileNet makes use of Depthwise Separable Convolutions - a modern technique
# A form of factorized convolution that separates a normal convolution into
# - Depthwise convolution (3x3), filtering
# - Pointwise convolution (1x1), combining

# A normal convolutional layer does both filtering and combining in a single
# step, however a Depthwise Separable Convolution separates the two steps
# into different layers. One layer for filtering (3x3 conv) that is applied
# to each channel separately (thus depthwise), and one layer for combining 
# (1x1 conv) that is used to combine the outputs of the filter layer. 

class DWConvBlock(nn.Module):
    def __init__(
            self,
            in_channels : int,
            out_channels : int,
            downsample : bool = False,  # No downsampling
        ):
        super().__init__()
        stride = 2 if downsample else 1

        depthwise_conv = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = in_channels, 
            groups = in_channels,  # Avoids channel mixing, keeps filter outputs separate
            kernel_size = 3, 
            stride = stride,  # Potential to downsample in depthwise
            padding = 1
        )
        pointwise_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
        )
        
        self.dwsep_stack = nn.Sequential(
            depthwise_conv,
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            pointwise_conv,
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwsep_stack(x)
    
class MobileNetTiny(nn.Module):
    def __init__(
            self,
            in_channels : int = 1,  # Gray Scale
            base_channels : int = 16,
            num_classes : int = 10,  # FashionMNIST
            dropout : float = 0.2,
            ):
        super().__init__()

        self.conv_stack = nn.Sequential(
            # Initial Conv Layer
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),  # [B, 16, 28, 28]

            # DW Conv Block 1
            DWConvBlock(base_channels, 2*base_channels, downsample=False),  # [B, 32, 28, 28]

            # DW Conv Block 2
            DWConvBlock(2*base_channels, 4*base_channels, downsample=True),  # [B, 64, 14, 14]

            # DW Conv Block 3
            DWConvBlock(4*base_channels, 8*base_channels, downsample=True),  # [B, 128, 7, 7]

            # # (Optional) DW Conv Block X
            # DWConvBlock(8*base_channels, 8*base_channels, downsample=False),
        )   

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),  # [B, 128, 7, 7] -> [B, 128, 1, 1]
            nn.Flatten(),  # [B, 128, 1, 1] -> [B, 128]
            nn.Dropout(p=dropout),
            nn.Linear(in_features=8*base_channels, out_features=num_classes)  # [B, 128] -> [B, 10]
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)
