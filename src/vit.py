import torch.nn as nn
import timm


def build_vit(
    num_classes: int = 6,
    hidden_dropout: float = 0.25,
    attention_dropout: float = 0.1,
    pretrained: bool = True,
) -> nn.Module:
    """Load ViT-B/16 384px from timm and adapt for single-channel grayscale input.

    Following Matlala et al. 2024:
    - Pre-trained on ImageNet-21k
    - 12 hidden layers, 12 attention heads
    - Patch embedding modified for 1-channel input
    """
    model = timm.create_model(
        "vit_base_patch16_384",
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=1,
        drop_rate=hidden_dropout,
        attn_drop_rate=attention_dropout,
    )
    return model
