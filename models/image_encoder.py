"""
Image encoder (Ba et al. ICCV 2015 Sec 3.2 & 3.3), extended with multiple backbones.

fc branch gv(·):
  - vgg19      : VGG-19 fc1 (4096-d) → 4096-gv_hidden-k
  - densenet121: global avg pool (1024-d) → 1024-gv_hidden-k
  - resnet50   : global avg pool (2048-d) → 2048-gv_hidden-k

conv branch g'_v(·): supported for all three backbones.
  feature map → K' filters 3×3; spatial resolution 14×14 for all.
  - vgg19:       conv_feature_layer ("conv5_3" 512×14×14, "conv4_3" 512×28×28, "pool5" 512×7×7)
  - densenet121: after denseblock3 (1024×14×14), conv_feature_layer ignored
  - resnet50:    after layer3 (1024×14×14), conv_feature_layer ignored

All pretrained weights are frozen (no fine-tuning), matching the paper spirit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    VGG19_Weights,
    DenseNet121_Weights,
    ResNet50_Weights,
)

# VGG-19 conv feature layer → slice index into vgg.features.children()
CONV_FEATURE_SLICE = {
    "pool5": None,   # full feature extractor → 512×7×7
    "conv5_3": -3,   # after conv5_3+ReLU, before conv5_4 → 512×14×14
    "conv4_3": -12,  # after conv4_3+ReLU, before conv4_4 → 512×28×28
}

_BACKBONE_CHOICES = ("vgg19", "densenet121", "resnet50")


class ImageEncoder(nn.Module):
    """Image encoder supporting VGG19 / DenseNet121 / ResNet50 backbones.

    The fc branch is available for all three backbones.
    The conv branch (forward_conv_feature) is only available for backbone='vgg19'.
    """

    def __init__(
        self,
        output_dim: int = 50,
        gv_hidden: int = 300,
        conv_channels: int = 5,
        conv_feature_layer: str = "conv5_3",
        backbone: str = "vgg19",
    ):
        """
        Args:
            output_dim: Output dimension k for joint embedding.
            gv_hidden: Hidden dimension for the fc projection branch.
            conv_channels: Number of predicted conv filters K' (VGG-19 only).
            conv_feature_layer: VGG-19 layer for conv branch ('conv5_3', 'conv4_3', 'pool5').
            backbone: One of 'vgg19', 'densenet121', 'resnet50'.
        """
        super().__init__()
        self.conv_channels = conv_channels
        self.conv_feature_layer = conv_feature_layer.lower()
        self.backbone = backbone.lower()

        if self.conv_feature_layer not in CONV_FEATURE_SLICE:
            raise ValueError(
                f"conv_feature_layer must be one of {list(CONV_FEATURE_SLICE)}; "
                f"got {conv_feature_layer!r}."
            )
        if self.backbone not in _BACKBONE_CHOICES:
            raise ValueError(
                f"backbone must be one of {_BACKBONE_CHOICES}; got {backbone!r}."
            )

        # ----------------------------
        # VGG-19
        # ----------------------------
        if self.backbone == "vgg19":
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

            # fc branch: features → 512×7×7 (25088) → fc1 4096-d
            self.features = vgg.features
            self.fc1 = nn.Sequential(
                vgg.classifier[0],  # Linear(25088 → 4096)
                vgg.classifier[1],  # ReLU
            )
            self.projection = nn.Sequential(
                nn.Linear(4096, gv_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gv_hidden, output_dim),
            )

            # conv branch
            children = list(vgg.features.children())
            slice_idx = CONV_FEATURE_SLICE[self.conv_feature_layer]
            self.features_conv = (
                nn.Sequential(*children)
                if slice_idx is None
                else nn.Sequential(*children[:slice_idx])
            )
            self.conv_reduce = nn.Conv2d(512, conv_channels, kernel_size=3, padding=1)

            # freeze pretrained weights
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.fc1.parameters():
                p.requires_grad = False
            for p in self.features_conv.parameters():
                p.requires_grad = False

        # ----------------------------
        # DenseNet-121
        # ----------------------------
        elif self.backbone == "densenet121":
            dense = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.features = dense.features  # output: [B, 1024, H, W] (before relu+pool)
            self.projection = nn.Sequential(
                nn.Linear(1024, gv_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gv_hidden, output_dim),
            )
            for p in self.features.parameters():
                p.requires_grad = False

            # conv branch: extract after denseblock3 → [B, 1024, 14×14]
            # DenseNet features children (0-indexed):
            #   0:conv0 1:norm0 2:relu0 3:pool0 4:denseblock1 5:transition1
            #   6:denseblock2 7:transition2 8:denseblock3 ...
            dense_children = list(dense.features.children())
            self.features_conv = nn.Sequential(*dense_children[:9])  # through denseblock3
            self.conv_reduce = nn.Conv2d(1024, conv_channels, kernel_size=3, padding=1)
            for p in self.features_conv.parameters():
                p.requires_grad = False
            self.fc1 = None

        # ----------------------------
        # ResNet-50
        # ----------------------------
        elif self.backbone == "resnet50":
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Remove final FC; keep avgpool → output: [B, 2048, 1, 1]
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.projection = nn.Sequential(
                nn.Linear(2048, gv_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gv_hidden, output_dim),
            )
            for p in self.features.parameters():
                p.requires_grad = False

            # conv branch: extract after layer3 → [B, 1024, 14×14]
            # ResNet children (0-indexed):
            #   0:conv1 1:bn1 2:relu 3:maxpool 4:layer1 5:layer2 6:layer3 7:layer4 ...
            resnet_children = list(resnet.children())
            self.features_conv = nn.Sequential(*resnet_children[:7])  # through layer3
            self.conv_reduce = nn.Conv2d(1024, conv_channels, kernel_size=3, padding=1)
            for p in self.features_conv.parameters():
                p.requires_grad = False
            self.fc1 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fc branch embeddings from images.

        Args:
            x: Input images [B, 3, 224, 224].

        Returns:
            Embeddings [B, k].
        """
        if self.backbone == "vgg19":
            with torch.no_grad():
                x = self.features(x)        # [B, 512, 7, 7]
                flat = torch.flatten(x, 1)  # [B, 25088]
                fc1_out = self.fc1(flat)    # [B, 4096]
            return self.projection(fc1_out)

        elif self.backbone == "densenet121":
            with torch.no_grad():
                x = self.features(x)                        # [B, 1024, H, W]
                x = F.relu(x, inplace=False)
                x = F.adaptive_avg_pool2d(x, (1, 1))       # [B, 1024, 1, 1]
                x = torch.flatten(x, 1)                    # [B, 1024]
            return self.projection(x)

        elif self.backbone == "resnet50":
            with torch.no_grad():
                x = self.features(x)        # [B, 2048, 1, 1]
                x = torch.flatten(x, 1)    # [B, 2048]
            return self.projection(x)

        else:
            raise RuntimeError(f"Unsupported backbone in forward: {self.backbone}")

    def forward_conv_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Compute conv branch features from images.

        Args:
            x: Input images [B, 3, 224, 224].

        Returns:
            Conv features [B, K', H, W]:
              vgg19:       pool5→7×7, conv5_3→14×14, conv4_3→28×28
              densenet121: denseblock3→14×14
              resnet50:    layer3→14×14

        Raises:
            ValueError: If features_conv is not available (should not happen with
                        the current three supported backbones).
        """
        if self.features_conv is None:
            raise ValueError(
                f"forward_conv_feature is not available for backbone={self.backbone!r}."
            )
        with torch.no_grad():
            x = self.features_conv(x)
        return self.conv_reduce(x)
