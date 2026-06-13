import torch
import torch.nn as nn
import math

class MaskGenerator:
    """
    Generates random masks for MAE (Masked Autoencoder) pretraining.
    """
    def __init__(self, input_size: int = 256, patch_size: int = 16, mask_ratio: float = 0.75):
        self.num_patches = (input_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.num_mask = int(self.mask_ratio * self.num_patches)

    def generate_mask(self, batch_size: int, device: torch.device):
        """
        Generates a binary mask where 1 indicates a masked patch and 0 indicates a visible patch.
        Returns:
            mask: [B, num_patches]
            ids_keep: [B, num_keep]
            ids_restore: [B, num_patches]
        """
        noise = torch.rand(batch_size, self.num_patches, device=device)
        
        # Sort noise to get indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first part (unmasked)
        num_keep = self.num_patches - self.num_mask
        ids_keep = ids_shuffle[:, :num_keep]
        
        # Generate binary mask
        mask = torch.ones([batch_size, self.num_patches], device=device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return mask, ids_keep, ids_restore

class MAEReconstructionHead(nn.Module):
    """
    Decoder head for MAE pretraining. Reconstructs pixel values from the latent representation.
    """
    def __init__(self, embed_dim: int = 768, decoder_embed_dim: int = 512, out_channels: int = 13, patch_size: int = 16, num_patches: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Absolute positional embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        self.decoder_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=8, batch_first=True),
            num_layers=4
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Predict pixel values for each patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * out_channels, bias=True)

    def forward(self, x, ids_restore):
        # x is the sequence of unmasked patches [B, N_keep + 1, embed_dim]
        # Map to decoder dim
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # exclude class token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Prepend class token and add pos embed
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        
        # Apply transformer
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        
        # Predict pixels (remove class token)
        x = self.decoder_pred(x[:, 1:, :])
        return x # [B, num_patches, patch_size**2 * out_channels]

def mae_loss(imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, patch_size: int = 16):
    """
    Computes MSE loss only on masked patches.
    imgs: [B, C, H, W] raw images
    pred: [B, num_patches, patch_size**2 * C] predicted patches
    mask: [B, num_patches] 1 for masked, 0 for kept
    """
    # Patchify target image
    B, C, H, W = imgs.shape
    p = patch_size
    h, w = H // p, W // p
    
    target = imgs.reshape(B, C, h, p, w, p)
    target = target.permute(0, 2, 4, 3, 5, 1) # [B, h, w, p, p, C]
    target = target.reshape(B, h * w, p**2 * C) # [B, num_patches, p**2 * C]
    
    # Compute per-patch loss
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1) # [B, num_patches]
    
    # Mask loss
    loss = (loss * mask).sum() / mask.sum()
    return loss
