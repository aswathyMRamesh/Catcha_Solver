import torch
import torch.nn as nn

from backbone_architecture import (
    ResNet18Backbone, VGG16Backbone, InceptionV1Backbone, height_pool_to_sequence
)
from rnn_architecture import BiLSTMEncoder, AdditiveAttention1D
from refine_architecture import AdaptiveRefiner, STNNewRST, AdaptivePlusSTNNew

__all__ = [
    "CRNN_ResNet18_LSTM",
    "CRNN_VGG16_LSTM",
    "CRNN_InceptionV1_LSTM",
    "CRNN_Adaptive_ResNet_LSTM",
    "CRNN_STN_ResNet_LSTM",
    "CRNN_Adaptive_STN_ResNet_LSTM",
]
import torch
import torch.nn as nn
import torch.nn.init as init


__all__ = [
    "CRNN_ResNet18_LSTM",
    "CRNN_VGG16_LSTM",
    "CRNN_InceptionV1_LSTM",
    "CRNN_Adaptive_ResNet_LSTM",
    "CRNN_STN_ResNet_LSTM",
    "CRNN_Adaptive_STN_ResNet_LSTM",
    "VanillaCRNN",
]


# Simple vanilla CNN backbone (5-6 conv layers)

class VanillaCNNBackbone(nn.Module):
    """
    A compact CRNN-friendly CNN with 6 conv layers.
    Downsampling produces ~16x stride along width to form a reasonable T.
    Input: (B, C=1 or 3, H, W)
    Output: (B, C_out=feat_dim, H', W')
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 256):
        super().__init__()
        # Block 1: 1/2 in H & W
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # /2 width, /2 height
        )
        # Block 2: 1/4 in H & W
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # /4 width, /4 height
        )
        # Block 3: keep width/height
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Block 4: reduce height only (helps textlines) -> H smaller, W unchanged
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # /2 height
        )
        # Block 5: stride along width (x2) -> ~ /8 total width
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=(1, 2), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Block 6: another width stride (x2) -> ~ /16 total width
        self.block6 = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=3, padding=1, stride=(1, 2), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x


# VanillaCRNN (from-scratch + Xavier init)

class VanillaCRNN(nn.Module):
    """
    Minimal CRNN assembled here (no external backbone/refiners).
    Components:
      - 6-layer CNN (VanillaCNNBackbone)
      - BiLSTM encoder
      - (optional) Additive attention head
    Everything is initialized from scratch with Xavier (Glorot) initialization.

    Returns: (logits_seq[T,B,K], logits_attn[B,K], feats[T,B,2H], attn_weights[B,T])
    """
    def __init__(self, num_classes: int,
                 in_channels: int = 1,
                 feat_dim: int = 256,
                 rnn_hidden: int = 256,
                 rnn_layers: int = 2,
                 dropout: float = 0.1,
                 use_attention: bool = True):
        super().__init__()
        # CNN backbone
        self.backbone = VanillaCNNBackbone(in_channels=in_channels, out_channels=feat_dim)

        # to-sequence helper expects C(features)=feat_dim
        self.enc = BiLSTMEncoder(input_size=feat_dim, hidden_size=rnn_hidden,
                                 num_layers=rnn_layers, dropout=dropout)

        self.classifier_seq = nn.Linear(2 * rnn_hidden, num_classes)

        self.use_attention = use_attention
        if use_attention:
            self.attn = AdditiveAttention1D(dim=2 * rnn_hidden, attn_dim=128)
            self.classifier_attn = nn.Linear(2 * rnn_hidden, num_classes)
        else:
            self.attn = None
            self.classifier_attn = None

        # Xavier init for everything (CNN, LSTM, Linear, BN)
        self._init_weights_xavier_()

    @staticmethod
    def expected_T(width: int, stride: int = 16) -> int:
        # Matches the ~x16 width stride used in VanillaCNNBackbone
        return (width + stride - 1) // stride

    def _init_weights_xavier_(self):
        """
        Apply Xavier/Glorot initialization across the model:
          - Conv/Linear: xavier_uniform_ + bias zeros
          - BatchNorm: weight=1, bias=0
          - LSTM weights (ih/hh for each layer & direction): xavier_uniform_, biases zeros
        """
        # Generic init for Conv/Linear/BN
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # LSTM-specific: BiLSTMEncoder typically owns an nn.LSTM in .lstm (or similar).
        # We try common attribute names
        
        lstm = getattr(self.enc, "lstm", None)
        if isinstance(lstm, nn.LSTM):
            for name, param in lstm.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x: torch.Tensor):
        # CNN backbone -> (B,C,H',W')
        feat = self.backbone(x)

        # To sequence (T,B,C)
        seq = height_pool_to_sequence(feat)  # uses your existing utility

        # BiLSTM
        feats = self.enc(seq)                    # (T,B,2H)
        logits_seq = self.classifier_seq(feats)  # (T,B,K)

        if self.use_attention:
            context, attn_w = self.attn(feats)       # context: (B,2H), attn_w: (B,T)
            logits_attn = self.classifier_attn(context)
        else:
            B, T = seq.size(1), seq.size(0)
            attn_w = torch.zeros(B, T, device=x.device)
            logits_attn = torch.zeros(B, logits_seq.size(-1), device=x.device)

        return logits_seq, logits_attn, feats, attn_w


class _CRNNCommon(nn.Module):
    """
    Shared CRNN scaffold:
      optional image_refiner (pre-backbone) -> backbone -> seq -> BiLSTM -> logits
    Returns: (logits_seq[T,B,K], logits_attn[B,K], feats[T,B,2H], attn_weights[B,T])
    """
    def __init__(self, backbone: nn.Module, num_classes: int,
                 rnn_hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.1,
                 use_attention: bool = True,
                 image_refiner: nn.Module = None):
        super().__init__()
        self.backbone = backbone
        self.enc = BiLSTMEncoder(input_size=256, hidden_size=rnn_hidden,
                                 num_layers=rnn_layers, dropout=dropout)
        self.classifier_seq = nn.Linear(2 * rnn_hidden, num_classes)

        self.use_attention = use_attention
        if use_attention:
            self.attn = AdditiveAttention1D(dim=2 * rnn_hidden, attn_dim=128)
            self.classifier_attn = nn.Linear(2 * rnn_hidden, num_classes)
        else:
            self.attn = None
            self.classifier_attn = None

        # image-level refiner (may return x or (x, theta))
        self.image_refiner = image_refiner

    @staticmethod
    def expected_T(width: int, stride: int = 16) -> int:
        return (width + stride - 1) // stride

    def _apply_image_refiner(self, x: torch.Tensor) -> torch.Tensor:
        if self.image_refiner is None:
            return x
        out = self.image_refiner(x)
        # Support both: refiner(x) -> x'  OR  refiner(x) -> (x', theta)
        if isinstance(out, (tuple, list)):
            x = out[0]
        else:
            x = out
        return x

    def forward(self, x):
        # optional pre-backbone refinement
        x = self._apply_image_refiner(x)

        # backbone: (B,C,H',W')
        feat = self.backbone(x)

        # to sequence (T,B,C)
        seq = height_pool_to_sequence(feat)

        # BiLSTM
        feats = self.enc(seq)                    # (T,B,2H)
        logits_seq = self.classifier_seq(feats)  # (T,B,K)

        if self.use_attention:
            context, attn_w = self.attn(feats)
            logits_attn = self.classifier_attn(context)  # (B,K)
        else:
            B = seq.size(1)
            T = seq.size(0)
            attn_w = torch.zeros(B, T, device=x.device)
            logits_attn = torch.zeros(B, logits_seq.size(-1), device=x.device)

        return logits_seq, logits_attn, feats, attn_w


# Concrete variants

# A) resnet18 + LSTM
class CRNN_ResNet18_LSTM(_CRNNCommon):
    def __init__(self, num_classes: int, feat_dim: int = 256,
                 rnn_hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.1,
                 pretrained_backbone: bool = False):
        # change: backbones are 1-ch
        backbone = ResNet18Backbone(out_channels=feat_dim)
        super().__init__(backbone, num_classes, rnn_hidden, rnn_layers, dropout, use_attention=True)

# B) VGG-16 + LSTM
class CRNN_VGG16_LSTM(_CRNNCommon):
    def __init__(self, num_classes: int, feat_dim: int = 256,
                 rnn_hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.1,
                 pretrained_backbone: bool = False):
        backbone = VGG16Backbone(out_channels=feat_dim)
        super().__init__(backbone, num_classes, rnn_hidden, rnn_layers, dropout, use_attention=True)

# C) Inception v1 + LSTM
class CRNN_InceptionV1_LSTM(_CRNNCommon):
    def __init__(self, num_classes: int, feat_dim: int = 256,
                 rnn_hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.1,
                 pretrained_backbone: bool = False):
        backbone = InceptionV1Backbone(out_channels=feat_dim)
        super().__init__(backbone, num_classes, rnn_hidden, rnn_layers, dropout, use_attention=True)

# D) adaptive + resnet + LSTM  (pre-backbone AFFN)
class CRNN_Adaptive_ResNet_LSTM(_CRNNCommon):
    def __init__(self, num_classes: int, feat_dim: int = 256,
                 rnn_hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.1,
                 pretrained_backbone: bool = False, in_channels: int = 1,
                 affn_depth: int = 3, affn_kernel: int = 5, affn_growth: int = 4, affn_alpha_init: float = 0.7):
        backbone = ResNet18Backbone(out_channels=feat_dim)
        image_refiner = AdaptiveRefiner(
            in_channels=in_channels, depth=affn_depth, kernel_size=affn_kernel,
            growth=affn_growth, alpha_init=affn_alpha_init
        )
        super().__init__(backbone, num_classes, rnn_hidden, rnn_layers, dropout,
                         use_attention=True, image_refiner=image_refiner)

# E) new STN + resnet + LSTM  (pre-backbone constrained STN)
class CRNN_STN_ResNet_LSTM(_CRNNCommon):
    def __init__(self, num_classes: int, feat_dim: int = 256,
                 rnn_hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.1,
                 pretrained_backbone: bool = False, in_channels: int = 1,
                 stn_loc_channels: int = 32, stn_tx_lim: float = 0.5, stn_ty_lim: float = 0.5,
                 stn_s_min: float = 0.8, stn_s_max: float = 2.0):
        backbone = ResNet18Backbone(out_channels=feat_dim)
        image_refiner = STNNewRST(
            in_channels=in_channels, loc_channels=stn_loc_channels,
            tx_lim=stn_tx_lim, ty_lim=stn_ty_lim, s_min=stn_s_min, s_max=stn_s_max
        )
        super().__init__(backbone, num_classes, rnn_hidden, rnn_layers, dropout,
                         use_attention=True, image_refiner=image_refiner)

# F) adaptive + new STN + resnet + LSTM  (AFFN -> STNNewRST -> backbone)
class CRNN_Adaptive_STN_ResNet_LSTM(_CRNNCommon):
    def __init__(self, num_classes: int, feat_dim: int = 256,
                 rnn_hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.1,
                 pretrained_backbone: bool = False, in_channels: int = 1,
                 affn_depth: int = 3, affn_kernel: int = 5, affn_growth: int = 4, affn_alpha_init: float = 0.7,
                 stn_loc_channels: int = 32, stn_tx_lim: float = 0.5, stn_ty_lim: float = 0.5,
                 stn_s_min: float = 0.8, stn_s_max: float = 2.0):
        backbone = ResNet18Backbone(out_channels=feat_dim)
        image_refiner = AdaptivePlusSTNNew(
            in_channels=in_channels,
            affn_depth=affn_depth, affn_kernel=affn_kernel, affn_growth=affn_growth, affn_alpha_init=affn_alpha_init,
            stn_loc_channels=stn_loc_channels, stn_tx_lim=stn_tx_lim, stn_ty_lim=stn_ty_lim,
            stn_s_min=stn_s_min, stn_s_max=stn_s_max
        )
        super().__init__(backbone, num_classes, rnn_hidden, rnn_layers, dropout,
                         use_attention=True, image_refiner=image_refiner)
