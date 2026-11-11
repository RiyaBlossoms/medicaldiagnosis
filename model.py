"""
Lightweight Hybrid EfficientNet + Vision Transformer model
FAST DEMO VERSION
"""

import torch
import torch.nn as nn
import timm

from config import Config


class FusionBlock(nn.Module):
    def __init__(self, cnn_dim, vit_dim, hidden, dropout):
        super().__init__()

        self.proj_cnn = nn.Linear(cnn_dim, hidden)
        self.proj_vit = nn.Linear(vit_dim, hidden)

        self.attn = nn.MultiheadAttention(
            hidden,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, cnn_feat, vit_feat):

        q = self.proj_cnn(cnn_feat).unsqueeze(1)
        kv = self.proj_vit(vit_feat).unsqueeze(1)

        attn_out, _ = self.attn(q, kv, kv)

        x = self.norm1(q + attn_out).squeeze(1)
        x = self.norm2(x + self.ffn(x))

        return x


class HybridMedicalModel(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        print("LOADING EFFICIENTNET...")

        # LIGHTER EfficientNet
        self.efficientnet = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        cnn_dim = self.efficientnet.num_features

        print("LOADING VIT...")

        # LIGHTER ViT
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=0
        )

        vit_dim = self.vit.num_features

        print("FREEZING VIT...")

        # Freeze ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        print("CREATING FUSION BLOCK...")

        self.fusion = FusionBlock(
            cnn_dim,
            vit_dim,
            cfg.fusion_hidden_dim,
            cfg.dropout
        )

        print("CREATING CLASSIFIER...")

        self.classifier = nn.Sequential(
            nn.Linear(cfg.fusion_hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, cfg.num_classes())
        )

        print("MODEL READY")

    def forward(self, x):

        cnn_feat = self.efficientnet(x)

        with torch.no_grad():
            vit_feat = self.vit(x)

        fused = self.fusion(cnn_feat, vit_feat)

        output = self.classifier(fused)

        return output

    def get_cnn_features(self, x):
        return self.efficientnet.forward_features(x)