import torch.nn as nn
import timm


def build_model(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B3 with a single-logit binary head (for BCEWithLogitsLoss)."""
    return timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=1)
