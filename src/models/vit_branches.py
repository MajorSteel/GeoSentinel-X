import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
from functools import partial

class SARVisionTransformer(nn.Module):
    """
    Vision Transformer branch for SAR data (Sentinel-1).
    Typically 2 bands (VV, VH) + calculated features.
    """
    def __init__(self, in_channels: int = 4, embed_dim: int = 768, image_size: int = 256, patch_size: int = 16):
        super().__init__()
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=12,
            num_heads=12,
            hidden_dim=embed_dim,
            mlp_dim=embed_dim * 4,
            dropout=0.1,
            attention_dropout=0.1,
            num_classes=0 # Headless for foundation model
        )
        # Override the input projection to support arbitrary input channels
        self.vit.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x is [B, C, H, W]
        # Return hidden states (sequence of patches)
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        return x # Shape: [B, num_patches + 1, embed_dim]

class OpticalVisionTransformer(nn.Module):
    """
    Vision Transformer branch for Optical data (Sentinel-2).
    Typically 13 bands + spectral indices.
    """
    def __init__(self, in_channels: int = 13, embed_dim: int = 768, image_size: int = 256, patch_size: int = 16):
        super().__init__()
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=12,
            num_heads=12,
            hidden_dim=embed_dim,
            mlp_dim=embed_dim * 4,
            dropout=0.1,
            attention_dropout=0.1,
            num_classes=0
        )
        self.vit.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        return x

class ClimateMLP(nn.Module):
    """
    MLP branch for climate tabular/low-res data (ERA5).
    """
    def __init__(self, in_features: int = 4, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        # x: [B, in_features]
        out = self.proj(x) # [B, embed_dim]
        # Reshape to match sequence format [B, 1, embed_dim]
        return out.unsqueeze(1)
