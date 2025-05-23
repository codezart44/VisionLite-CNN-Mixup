import torch
import torch.nn as nn

# Paper : Going deeper with convolutions (Szegedy, 2014) Google

# Inception (GoogLeNet from 2014) strives for architectural efficiency
# allowing for wider and and deeper networks while keeping computational
# cost constant. 

# GoogLeNet includes both the novel Inception Block, where inputs are processed
# in parallell by varying filter sizes, and also uses the auxiliary classifiers
# (intermediate classifiers) which both provide stronger gradient signals,
# regularizes the network and forces the network to learn meaningful features
# for classification throughout all parts of the network!

# InceptionModule
# ---------------
# Four convolutional paths to pass in parallell
# 1. 1x1 Conv
# 2. 1x1 -> 3x3 Conv
# 3. 1x1 -> 5x5 Conv
# 4. 3x3 Maxpool -> 1x1 Conv
# The four paths allows the network to capture multi-scale features


# General Structure
# -----------------
# Initial conv stack
# MaxPooling
# InceptionBlock stack (w. auxiliary classifiers)
# MaxPool
# InceptionBlock again?
# GlobalAvgPool -> Flattening -> Dropout -> FC (Softmax not neeed due to CrossEntropyLoss)


class InceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels : int,
            inception_channels : tuple[int, int, int, int] = (4,4,4,4)
            ) -> None:
        super().__init__()

        self.out_channels = sum(inception_channels)
        path1_channels = inception_channels[0]  # 1x1
        path2_channels = inception_channels[1]  # 3x3
        path3_channels = inception_channels[2]  # 5x5
        path4_channels = inception_channels[3]  # pool
        
        # 1x1 Conv
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, path1_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=path1_channels),
            nn.ReLU(inplace=True),
        )

        # 1x1 -> 3x3 Conv
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, path2_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(path2_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(path2_channels, path2_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(path2_channels),
            nn.ReLU(inplace=True),
        )

        # 1x1 -> 5x5 Conv
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, path3_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(path3_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(path3_channels, path3_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(path3_channels),
            nn.ReLU(inplace=True),
        )

        # 3x3 Maxpool -> 1x1 Conv
        self.pool3x3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, path4_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(path4_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Consists of four channels (paths)
        out1 = self.conv1x1(x)  # 1x1 Conv
        out2 = self.conv3x3(x)  # 1x1 -> 3x3 Conv
        out3 = self.conv5x5(x)  # 1x1 -> 5x5 Conv
        out4 = self.pool3x3(x)  # 3x3 Maxpool -> 1x1 Conv
        return torch.cat([out1, out2, out3, out4], dim=1)


class AuxiliaryClassifier(nn.Module):
    def __init__(
            self,
            in_channels : int,
            out_channels : int = 16, # NOTE Preferably! out_channels > num_classes
            num_classes : int = 10,  # FashionMNIST
            dropout : float = 0.2,
            ) -> None:
        super().__init__()

        self.aux_clf = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # For simplicity
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),  # Channel reduction
            nn.Flatten(),  # [B, 4, 1, 1] -> [B, 4]
            nn.Dropout(p=dropout),
            nn.Linear(out_channels, num_classes)  # [B, 10]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aux_clf(x)


class GoogLeNetTiny(nn.Module):
    def __init__(
            self,
            in_channels : int = 1,  # Gray Scale
            base_channels : int = 16,
            inception_channels : tuple[int, int, int, int] = (4,4,4,4),
            num_classes : int = 10,  # FashionMNIST
            dropout : float = 0.2,
            # downsample : bool = False,
            ) -> None:
        super().__init__()
        assert sum(inception_channels) >= num_classes, 'Sum of inceptions channels should exceed num classes. '

        self.init_conv = nn.Sequential(
            # Initial Conv(s)  [B, 1, 28, 28]
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),  # [B, 16, 28, 28]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 16, 14, 14]
        )

        # Inception Blocks
        self.inception_block1 = InceptionBlock(base_channels, inception_channels)  # [B, 16, 14, 14]
        self.inception_block2 = InceptionBlock(self.inception_block1.out_channels, inception_channels)  # [B, 16, 14, 14]
        self.inception_block3 = InceptionBlock(self.inception_block2.out_channels, inception_channels)  # [B, 16, 14, 14]

        # Auxiliary Classifier
        self.aux_clf = AuxiliaryClassifier(self.inception_block3.out_channels)  # [B, 10]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),  # [B, incep_out, 14, 14] -> [B, incep_out, 1, 1]
            nn.Flatten(),  # [B, incep_out]
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.inception_block3.out_channels, out_features=num_classes)
        )

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.init_conv(x)           # [B, 16, 14, 14]
        x = self.inception_block1(x)    # [B, 16, 14, 14]
        x = self.inception_block2(x)    # [B, 16, 14, 14]
        aux_pred = self.aux_clf(x)      # [B, 10]
        x = self.inception_block3(x)    # [B, 16, 14, 14]
        return x, aux_pred
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, aux_pred = self.forward_features(x)
        if self.training:
            return self.classifier(x), aux_pred  # y_pred, aux_pred
        return self.classifier(x)




def main():
    t = torch.rand((1, 1, 28, 28))
    # inception = InceptionBlock(in_channels=1)
    googlenet_tiny = GoogLeNetTiny(inception_channels=(4,4,4,5))
    y_pred, aux_pred = googlenet_tiny(t)
    print(y_pred.shape, y_pred)
    print(aux_pred.shape, aux_pred)

if __name__=='__main__':
    main()
