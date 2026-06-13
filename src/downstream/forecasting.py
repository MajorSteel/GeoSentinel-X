import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM Cell for Spatio-temporal forecasting.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class LULCForecaster(nn.Module):
    """
    Predicts future land cover maps (e.g., T+1) given historical time steps.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convlstm = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=3)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=16)

    def forward(self, seq_features):
        """
        seq_features: List of temporal features [B, num_patches, embed_dim] spanning T0, T1, ...
        """
        B, N, D = seq_features[0].shape
        H_patch = W_patch = int(N ** 0.5)
        
        # Initialize hidden states
        h = torch.zeros(B, self.hidden_dim, H_patch, W_patch, device=seq_features[0].device)
        c = torch.zeros(B, self.hidden_dim, H_patch, W_patch, device=seq_features[0].device)
        
        # Iterate through time steps
        for features_t in seq_features:
            # Reshape 1D sequence to 2D grid
            x_t = features_t.transpose(1, 2).reshape(B, D, H_patch, W_patch)
            h, c = self.convlstm(x_t, (h, c))
            
        # Decode the final hidden state to predict T+1 map
        predicted_map = self.decoder(h)
        predicted_map = self.upsample(predicted_map)
        
        return predicted_map
