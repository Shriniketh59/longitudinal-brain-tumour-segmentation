"""VGG16 slice classifier: ImageNet-pretrained backbone, new 4-way head."""

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

CLASS_NAMES = ("glioma", "meningioma", "no_tumor", "pituitary")


def build_vgg16_classifier(num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> nn.Module:
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg16(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_vgg16_classifier(path: str, device: torch.device) -> nn.Module:
    model = build_vgg16_classifier(pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
