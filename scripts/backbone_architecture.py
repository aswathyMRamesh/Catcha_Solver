import torch
import torch.nn as nn
from torchvision import models

__all__ = [
    "ResNet18Backbone",
    "VGG16Backbone",
    "InceptionV1Backbone",
    "height_pool_to_sequence",
]

# Small helpers

def _make_resnet18_no_weights():
    # Newer torchvision: weights=None; older: pretrained=False
    try:
        return models.resnet18(weights=None)
    except TypeError:
        return models.resnet18(pretrained=False)

def _make_vgg16_no_weights():
    try:
        return models.vgg16(weights=None)
    except TypeError:
        return models.vgg16(pretrained=False)

def _make_googlenet_no_weights():
    # We still disable aux heads; no weights used.
    try:
        return models.googlenet(weights=None, aux_logits=False)
    except TypeError:
        return models.googlenet(pretrained=False, aux_logits=False)

def height_pool_to_sequence(x: torch.Tensor) -> torch.Tensor:
    """
    Convert CNN feature map (B,C,H,W) to sequence (T,B,C) by pooling H->1 and
    transposing to time-major.
    """
    x = nn.functional.adaptive_avg_pool2d(x, (1, None))  # (B,C,1,W)
    x = x.squeeze(2)                                      # (B,C,W)
    x = x.permute(2, 0, 1).contiguous()                  # (T,B,C)
    return x

class _Project(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)

# Backbones (no weights)

class ResNet18Backbone(nn.Module):
    """
    ResNet-18 feature extractor with effective downsample ≈ /16.
    We de-stride layer4 to avoid /32. Use height_pool_to_sequence() to get (T,B,C).
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        backbone = _make_resnet18_no_weights()

        # CHANGE: first conv accept grayscale (1->64) 
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.stem = nn.Sequential(
            backbone.conv1,  # /2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # /4
        )
        self.layer1 = backbone.layer1      # /4
        self.layer2 = backbone.layer2      # /8
        self.layer3 = backbone.layer3      # /16
        self.layer4 = backbone.layer4      # default /32

        # keep /16 by removing stride in the first block of layer4
        self.layer4[0].conv1.stride = (1, 1)
        if self.layer4[0].downsample is not None:
            self.layer4[0].downsample[0].stride = (1, 1)

        self.proj = _Project(512, out_channels)
        self.stride = 16  # horizontal stride wrt input width

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)  # (B,C,H',W')
        return x

class VGG16Backbone(nn.Module):
    """
    VGG-16 backbone with downsample ≈ /16 (drop the last pool).
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        backbone = _make_vgg16_no_weights()
        feats = list(backbone.features.children())

        # CHANGE: first conv accept grayscale (1->64)
        if isinstance(feats[0], nn.Conv2d) and feats[0].in_channels != 1:
            conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)
            nn.init.kaiming_normal_(conv0.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv0.bias)
            feats[0] = conv0
        
        # keep up to pool4; this gives /16 instead of /32
        last_pool_idx = max(i for i, m in enumerate(feats) if isinstance(m, nn.MaxPool2d))
        feats = feats[:last_pool_idx]  # up to the 4th pool
        self.features = nn.Sequential(*feats)
        self.proj = _Project(512, out_channels)
        self.stride = 16

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x

class InceptionV1Backbone(nn.Module):
    """
    GoogLeNet (Inception v1) backbone with effective downsample ≈ /16.
    We stop after inception4e (do not apply maxpool4).
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        net = _make_googlenet_no_weights()

        # CHANGE: first conv accept grayscale (1->64)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
        
        self.conv1 = net.conv1
        self.maxpool1 = net.maxpool1
        self.conv2 = net.conv2
        self.conv3 = net.conv3
        self.maxpool2 = net.maxpool2

        self.inception3a = net.inception3a
        self.inception3b = net.inception3b
        self.maxpool3 = net.maxpool3

        self.inception4a = net.inception4a
        self.inception4b = net.inception4b
        self.inception4c = net.inception4c
        self.inception4d = net.inception4d
        self.inception4e = net.inception4e
        # Not applying net.maxpool4 to keep /16

        # After inception4e output channels = 832
        self.proj = _Project(832, out_channels)
        self.stride = 16

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.proj(x)
        return x
