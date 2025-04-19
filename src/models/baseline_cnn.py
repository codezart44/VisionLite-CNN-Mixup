import torch
from torch import nn

class BaselineCNN(nn.Module):
    def __init__(
            self, 
            in_channels: int = 1,  # Gray Scale
            base_channels: int = 8,
            num_classes: int = 10,  # FashionMNIST
            dropout: float = 0.2,
            kernel_size: int = 3,
            ):
        super().__init__()

        self.conv_stack = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, base_channels, kernel_size, padding=1), # 256 x 256
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels),  # NOTE Apply batch normalization after activation
            nn.MaxPool2d(kernel_size=2),  # 28 -> 14

            # Block 2
            nn.Conv2d(base_channels, base_channels*2, kernel_size, padding=1), # 
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels*2),  # Normalize output, standard scale
            nn.MaxPool2d(kernel_size=2),  # 14 -> 7

            # Block 3 NOTE DISABLED DURING DEVELOPMENT
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels*4),  # Standard scale
            nn.MaxPool2d(kernel_size=2),  # 64 -> 32
        )   

        self.classifier = nn.Sequential(
            # Adaptive average pooling essentially takes the mean values of each 
            # channel layer. This compression works well since the high level features,
            # deep into the network, contains binary information (true / false) about
            # a class and using central statistics con often summarise that information
            # neatly into a single scalar signal. This is a classification task, and 
            # rather asks the question 'is this a shoe?' than 'at what pixel is the shoe at?'. 
            nn.AdaptiveAvgPool2d(output_size=1),  # [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(),  # [B, C * 1 * 1] -> [B, C], flattens all but the batch dimension
            nn.Dropout(p=dropout),
            nn.Linear(in_features=base_channels*4, out_features=num_classes),  # One for each class
            # CrossEntropyLoss handels vectorized input internally
        )
    def forward_features(self, x) -> torch.Tensor:
        return self.conv_stack(x)
    
    def forward(self, x) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)
