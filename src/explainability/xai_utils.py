import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for Vision Transformers.
    Highlights regions in the input image responsible for a particular prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook into the target layer (e.g., last Block in ViT)
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        """
        Generate CAM for a specific class index.
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=-1)
            
        score = logits[:, class_idx].squeeze()
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Calculate weights based on gradients
        gradients = self.gradients.mean(dim=(2, 3), keepdim=True) # Assuming spatial dims
        activations = self.activations
        
        cam = F.relu((gradients * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().detach().cpu().numpy()

# SHAP wrapper would ideally utilize the SHAP library's DeepExplainer or GradientExplainer
# Note: For Transformers, specific SHAP partition explainers are usually required.
