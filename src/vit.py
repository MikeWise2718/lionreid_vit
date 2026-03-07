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


def build_vit_ablation(variant: str, num_classes: int = 6) -> nn.Module:
    """Build ViT ablation variants.

    Variants:
        B1 - Reduce transformer layers from 12 to 8
        B2 - Remove positional encoding
        B3 - Increase patch size from 16x16 to 32x32
        B4 - Reduce attention heads from 12 to 6
    """
    if variant == "B1":
        model = timm.create_model(
            "vit_base_patch16_384",
            pretrained=False,
            num_classes=num_classes,
            in_chans=1,
            drop_rate=0.25,
            attn_drop_rate=0.1,
            depth=8,
        )
    elif variant == "B2":
        model = build_vit(num_classes=num_classes)
        # Zero out positional embeddings
        model.pos_embed.data.zero_()
        model.pos_embed.requires_grad = False
        return model
    elif variant == "B3":
        model = timm.create_model(
            "vit_base_patch32_384",
            pretrained=True,
            num_classes=num_classes,
            in_chans=1,
            drop_rate=0.25,
            attn_drop_rate=0.1,
        )
    elif variant == "B4":
        model = timm.create_model(
            "vit_base_patch16_384",
            pretrained=False,
            num_classes=num_classes,
            in_chans=1,
            drop_rate=0.25,
            attn_drop_rate=0.1,
            num_heads=6,
        )
    else:
        raise ValueError(f"Unknown ViT ablation variant: {variant}")
    return model
