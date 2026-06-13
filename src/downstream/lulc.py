import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class LULC_ClassificationHead(nn.Module):
    """
    Deep Learning head for Land Use Land Cover (LULC) Classification.
    Takes the fused foundation model latent representation and outputs class logits.
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 12):
        super().__init__()
        # Using a Multi-Layer Perceptron (MLP) mapping latent feature vector to LULC classes.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [B, embed_dim] (usually the class token or global average pool of patches)
        return self.classifier(x)

class ML_Baselines:
    """
    Machine learning baselines for pixel-based LULC.
    These operate directly on multi-spectral features rather than deep latents.
    """
    @staticmethod
    def get_random_forest():
        return RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, class_weight='balanced')

    @staticmethod
    def get_xgboost():
        return xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, tree_method='hist')

    @staticmethod
    def get_lightgbm():
        return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1, class_weight='balanced')
        
    @staticmethod
    def get_catboost():
        return CatBoostClassifier(iterations=200, depth=8, learning_rate=0.1, task_type="GPU", verbose=False)
