import torch
import torch.nn as nn

# Paper: AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (Dosovitskiy et. al., 2021)

# ViT stands for Vision Transformer
# Vision Transformers stand apart from CNNs in they they do not use
# convolutions but instead divide images into patches, transform them
# into vectors and then treat each patch vector as a token in a sequence,
# akin to works in an NLP context. 

# Patch Embedding Layer
# Turn an image into a sequence of patch embeddings
# [B, 1, 28, 28] -> [B, num_patches, embed_dim]
# E.g.
# - Patch size 4x4: 28x28 pixles -> 7x7 patches -> 49 patches
# - Each patch being 4x4 meaning 16 pixles. 
# - Lift patch vec to embed space of 64 dimension (richer features)

class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channels : int = 1,  # Gray Scale
            image_size : int = 28,  # FashionMNIST
            patch_size : int = 4,
            embed_dim : int = 64,
            ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size.'

        self.project = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        x = self.flatten(x)
        x = torch.permute(x, (0, 2, 1))
        return x

# The token embedding both adds the CLS token and the positional embeddings.
# After that return the full embedding sequence to the transformer. 

class TokenEmbedding(nn.Module):
    def __init__(
            self,
            num_patches : int = 49,
            embed_dim : int = 64,
            ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand((batch_size, -1, -1))
        x = torch.cat([cls_tokens, x], dim=1)  # Prepend along patch count dimension
        return x + self.pos_embed   # Pos embed broadcasted along batch dimension

# Transformer Encoding Blocks
# Broken into two steps
# 1. Multi-head Self-attention (MHSA)
# 2. Transformer Encoder Block
# In total: MHSA -> feedforward MLP -> residuals -> normalization

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            num_heads : int = 8,
            embed_dim : int = 64,
            ):
        super().__init__()
        assert embed_dim % num_heads == 0, 'Embed dim must be split evenly between heads.'
        self.num_heads = num_heads

        # Input [B, N, D] ~ [B, cls+patch_size, embed_dim]
        # Split into [Q, K, V] ~ (queries, keys, values)
        # [B, N, D] -> [B, num_heads, N, D_head]
        # Attention(Q, K, V) = softmax(QK^T / √d_k)*V
        # Concat all heads -> project back to output dimension

        # Create Q, K, V, same shape as TokenEmbeddings
        self.query = nn.Linear(embed_dim, embed_dim)  # [B, patch_size+cls, embed_dim] -> -//-
        self.key = nn.Linear(embed_dim, embed_dim)  # [B, N, D] -> [B, N, D]
        self.value = nn.Linear(embed_dim, embed_dim)  # [B, N, D] -> [B, N, D]

        self.softmax = nn.Softmax(dim=-1)  # Softmax along key dimension

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute Q, K, V matricies
        Q: torch.Tensor = self.query(x)  # Used to compute which tokens to attend to
        K: torch.Tensor = self.key(x)  # Used to decide how much attention each token receives
        V: torch.Tensor = self.value(x)  # Value representation of the tokens that are weighted and summed
        
        B, N, D = x.shape
        d_head = D // self.num_heads

        # Rehsape Q, K, V for multi head : [B, N, D] -> [B, num_heads, N, d_head]
        Q = Q.view((B, N, self.num_heads, d_head)).transpose(1, 2)
        K = K.view((B, N, self.num_heads, d_head)).transpose(1, 2)
        V = V.view((B, N, self.num_heads, d_head)).transpose(1, 2)

        # Attention(Q, K, V) = softmax(QK^T / √d_k)*V
        QK = torch.matmul(Q, torch.permute(K, (0, 1, 3, 2)))  # Swap last two dims
        weights = self.softmax(QK / (d_head**0.5))
        attention_multihead = torch.matmul(weights, V)
        attention = attention_multihead.permute(0, 2, 1, 3).reshape(B, N, D)  # Permute axes back and concat
        return self.out_proj(attention)

# Transformer Encoder Block
# Integrates MHSA with an MLP
# 1. Multi Head Self Attention (MHSA) - Draws information from neighbouring tokens
#   Where the queries get keys to get scores that are used to weight and sum values
# 2. Multi Layered Perceptron (MLP) - To learn to optimise its own state (not from
#   neighbours).
# While also utilising layer normalisations and akin to ResNet, make 
# residual connections (adding layer output to input making the network learn
# residuals instead of mapping full transformations).

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            hidden_dim : int = 128,
            num_heads : int = 8,
            embed_dim : int = 64,
            dropout : float = 0.2,
            ):
        super().__init__()

        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.msha = MultiHeadSelfAttention(
                num_heads=num_heads,
                embed_dim=embed_dim,
            )
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.msha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Final step
# ViT-Tiny Model
# The steps of the model are:
# Input Image  [B, 1, 28, 28]
#  v
# PatchEmbedding (split image into patches + linear projection) -> [B, 49, 64] ~ [B, num_patches, embed_dim]
#  v                                      E.g. Patch size = 4,  28/4 x 28/4 = 7x7 = 49
# TokenEmbedding (add CLS token + positional embedding) -> [B, 50, 64] [B, num_patches+1, embed_dim]
#  v
# TransformerEncoder (stack of TransformerEncoderBlocks), including MHSA & MLP
#  v                                      Optimise own state and draw info from neighbouring tokens
# Select CLS token output (index 0, is prefixed to patch tokens)
#  v
# MLP head (classification, 10 classes)
#  v 
# Logits output [B, 10]

# Overall steps of a Vision Transformer like this (though this is Tiny!)
# Patchify -> Tokenize -> CLS + Positional Encoding -> ...
# ... Tranformer Blocks -> CLS token readout -> Classification head

class ViTTiny(nn.Module):
    """ Vision Transformer Tiny Model """
    def __init__(
            self,
            image_size : int = 28,  # FashionMNIST
            in_channels : int = 1,  # GrayScale
            dropout : float = 0.2,
            num_classes : int = 10,  # FashionMNIST
            embed_dim : int = 64,
            num_heads : int = 8,
            hidden_dim : int = 128,
            patch_size : int = 4,  
            num_layers : int = 3,  # Number of Encoder Blocks in Encoder
            ):
        super().__init__()
        assert image_size % patch_size == 0, 'Resolution dimension must be divisible by patch size. '
        assert embed_dim % num_heads == 0, 'Embedding dimension must be divisible by number of heads. '

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            image_size=image_size, 
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.token_embed = TokenEmbedding(
            num_patches=(image_size // patch_size)**2,
            embed_dim=embed_dim,
        )
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                embed_dim=embed_dim,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.token_embed(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.norm(x)
        cls_token = x[:, 0, :]  # [B, num_patches+1, embed_dim] -> [B, embed_dim]
        return self.classifier(cls_token)  # [B, embed_dim] -> [B, num_classes]


def main():
    t = torch.rand((12, 1, 28, 28))
    # patch_embed = PatchEmbedding()
    # t = patch_embed(t)
    # token_embed = TokenEmbedding()
    # t = token_embed(t)
    # mhsa = MultiHeadSelfAttention()
    # t = mhsa(t)
    # encoder = TransformerEncoderBlock(14)
    # print(t.shape)
    # t = encoder(t)
    vit_tiny = ViTTiny()
    t = vit_tiny.forward_features(t)

if __name__=='__main__':
    main()
