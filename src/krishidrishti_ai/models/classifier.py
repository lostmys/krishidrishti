from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    efficientnet_b0,
    mobilenet_v3_small,
    vit_b_16,
)


SUPPORTED_MODELS = ("mobilenet_v3_small", "efficientnet_b0", "convnext_tiny", "vit_b_16")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build an ImageNet-initialized classifier for a controlled experiment."""
    if name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model '{name}'. Choose one of: {', '.join(SUPPORTED_MODELS)}")
    return model
