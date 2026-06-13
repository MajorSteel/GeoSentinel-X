import torch
import torch.nn as nn

class SiameseChangeDetector(nn.Module):
    """
    Multi-Temporal Siamese network for Change Detection.
    Takes latent representations of T1 and T2, calculates difference, and detects change.
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        """
        num_classes: e.g., 2 (binary change/no-change) or multi-class (deforestation, urban growth)
        """
        super().__init__()
        
        # Decoder to process the concatenated/differenced features
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=16) # Scale up to original 256x256 image size

    def forward(self, features_t1, features_t2):
        """
        features_t1, features_t2: [B, num_patches, embed_dim]
        Output: [B, num_classes, H, W] Change Map
        """
        B, N, D = features_t1.shape
        H_patch = W_patch = int(N ** 0.5)
        
        # Reshape sequence to 2D grid: [B, D, H_patch, W_patch]
        ft1 = features_t1.transpose(1, 2).reshape(B, D, H_patch, W_patch)
        ft2 = features_t2.transpose(1, 2).reshape(B, D, H_patch, W_patch)
        
        # Calculate absolute difference and concatenate
        diff = torch.abs(ft1 - ft2)
        concatenated = torch.cat([ft1, diff], dim=1) # Shape: [B, D*2, H_patch, W_patch]
        
        # Decode and classify change
        out = self.decoder(concatenated)
        out = self.upsample(out)
        
        return out
