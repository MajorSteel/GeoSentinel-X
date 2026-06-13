import torch
import torch.nn as nn
from .vit_branches import SARVisionTransformer, OpticalVisionTransformer, ClimateMLP

class MultiModalCrossAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key, value):
        attn_out, _ = self.cross_attn(query, key, value)
        out1 = self.norm1(query + attn_out)
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_out)
        return out2

class GeoSentinelFusionModel(nn.Module):
    """
    Multi-modal Fusion Foundation Model integrating SAR, Optical, and Climate branches.
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        
        # Branch Encoders
        self.sar_encoder = SARVisionTransformer(in_channels=4, embed_dim=embed_dim)
        self.opt_encoder = OpticalVisionTransformer(in_channels=13, embed_dim=embed_dim)
        self.climate_encoder = ClimateMLP(in_features=4, embed_dim=embed_dim)
        
        # Cross-Attention Fusion modules
        # Fuse SAR into Optical representations
        self.sar_to_opt_fusion = MultiModalCrossAttention(embed_dim=embed_dim)
        # Fuse Climate into fused Optical-SAR representations
        self.climate_fusion = MultiModalCrossAttention(embed_dim=embed_dim)
        
        # Temporal integration for multi-temporal forecasting (Stage 4/8)
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, batch_first=True),
            num_layers=6
        )

    def forward(self, sar, opt, climate, time_steps=None):
        """
        sar: [B, C_sar, H, W]
        opt: [B, C_opt, H, W]
        climate: [B, C_clim]
        """
        # 1. Encode single modalities
        sar_feat = self.sar_encoder(sar)     # [B, N_patches+1, embed_dim]
        opt_feat = self.opt_encoder(opt)     # [B, N_patches+1, embed_dim]
        clim_feat = self.climate_encoder(climate) # [B, 1, embed_dim]
        
        # 2. Cross-Modal Fusion
        # Query: Optical, Key/Value: SAR
        fused_spatial = self.sar_to_opt_fusion(query=opt_feat, key=sar_feat, value=sar_feat)
        
        # Query: Spatial Fused, Key/Value: Climate (broadcasted across patches)
        clim_feat_expanded = clim_feat.expand(-1, fused_spatial.size(1), -1)
        fused_all = self.climate_fusion(query=fused_spatial, key=clim_feat_expanded, value=clim_feat_expanded)
        
        # 3. Temporal Modeling (If sequence provided)
        if time_steps is not None:
            # Assuming fused_all is reshaped or processed for temporal sequence
            # For simplicity, passing directly to temporal transformer here
            latent = self.temporal_transformer(fused_all)
            return latent
            
        return fused_all
