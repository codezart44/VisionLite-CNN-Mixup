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
        attention = attention_multihead.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(attention)


class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,

            ):
        super().__init__()

        ...


def main():
    t = torch.rand((12, 1, 28, 28))
    patch_embed = PatchEmbedding()
    t = patch_embed(t)
    token_embed = TokenEmbedding()
    t = token_embed(t)
    mhsa = MultiHeadSelfAttention()
    t = mhsa(t)
    print(t.shape)

    

if __name__=='__main__':
    main()