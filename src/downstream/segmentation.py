import torch
import torch.nn as nn

class SegFormerDecoderHead(nn.Module):
    """
    Simplified SegFormer-style Decoder for semantic segmentation.
    Uses features extracted from the Foundation Model.
    """
    def __init__(self, in_channels_list: list = [768, 768, 768, 768], decoder_dim: int = 256, num_classes: int = 12):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.decoder_dim = decoder_dim
        
        # MLPs to project different feature pyramid levels to the same dim
        self.linear_c = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, decoder_dim),
                nn.LayerNorm(decoder_dim)
            ) for in_channels in in_channels_list
        ])
        
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(in_channels_list), decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features: list):
        # features is a list of tensors from different ViT blocks
        # e.g. [ [B, C1, H/4, W/4], [B, C2, H/8, W/8], ... ]
        outs = []
        for i, x in enumerate(features):
            B, C, H, W = x.shape
            
            # Reshape for linear layer: [B, C, H, W] -> [B, H*W, C]
            x_flat = x.flatten(2).transpose(1, 2)
            
            # Project to decoder_dim
            x_proj = self.linear_c[i](x_flat)
            
            # Reshape back to image grid: [B, H*W, decoder_dim] -> [B, decoder_dim, H, W]
            x_proj = x_proj.transpose(1, 2).reshape(B, self.decoder_dim, H, W)
            
            # Upsample to common resolution (e.g. H/4, W/4 - usually the highest res feature map)
            x_upsampled = nn.functional.interpolate(x_proj, size=features[0].shape[2:], mode='bilinear', align_corners=False)
            outs.append(x_upsampled)
            
        # Concatenate all upsampled feature maps
        out = torch.cat(outs, dim=1)
        
        # Fuse and classify
        out = self.linear_fuse(out)
        out = self.dropout(out)
        out = self.classifier(out)
        
        # Final upsample to original image size (assumes 4x upsampling needed)
        out = nn.functional.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        return out
