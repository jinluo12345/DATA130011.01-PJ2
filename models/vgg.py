"""
VGG
"""
import numpy as np
from torch import nn
import sys
sys.path.append('../')
sys.path.append('./')
from utils.nn import init_weights_
import torch

# ## Models implementation
def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()

    return parameters_n


class VGG_A(nn.Module):
    """VGG_A model

    size of Linear layers is smaller since input assumed to be 32x32x3, instead of
    224x224x3
    """

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


class VGG_A_Light(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        '''
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        '''
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        # x = self.stage5(x)
        x = self.classifier(x.view(-1, 32 * 8 * 8))
        return x


class VGG_A_Dropout(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

# ────────────────────────────────
# 1. VGG_Large (adapted VGG-16 for 32×32 inputs)
# ────────────────────────────────
class VGG_Large(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        # Block 1: 2×(conv 3×3, 64) → maxpool
        self.block1 = nn.Sequential(
            nn.Conv2d(inp_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,   64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2: 2×(conv 3×3, 128) → maxpool
        self.block2 = nn.Sequential(
            nn.Conv2d(64,  128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 3: 3×(conv 3×3, 256) → maxpool
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 4: 3×(conv 3×3, 512) → maxpool
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 5: 3×(conv 3×3, 512) → maxpool → final feature size 512×1×1
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Classifier: 512→512→512→num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.block1(x)   # → 64×16×16
        x = self.block2(x)   # → 128×8×8
        x = self.block3(x)   # → 256×4×4
        x = self.block4(x)   # → 512×2×2
        x = self.block5(x)   # → 512×1×1
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)


class VGG_ResBlock(nn.Module):

    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1, bias=False)
        if C_in != C_out:
            self.downsample = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
        else:
            self.downsample = None

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu_out(out)
        return out

class VGG_Res(nn.Module):
    """
    VGG-like architecture with residual links (after each block).
    We mimic VGG16’s overall shape but insert a residual connection
    within each “block” of convolutions.
    """
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        # Block1: 2×(conv3×3→ReLU) with residual, channels 3→64→64 → maxpool
        self.block1 = nn.Sequential(
            VGG_ResBlock(inp_ch, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block2: 2×(conv3×3→ReLU) with residual, channels 64→128→128 → maxpool
        self.block2 = nn.Sequential(
            VGG_ResBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block3: 3×(conv3×3→ReLU) with residual inside, channels 128→256→256; three times, then maxpool
        self.block3 = nn.Sequential(
            VGG_ResBlock(128, 256),
            VGG_ResBlock(256, 256),
            VGG_ResBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block4: 3×(conv3×3→ReLU) with residual, channels 256→512→512; three times → maxpool
        self.block4 = nn.Sequential(
            VGG_ResBlock(256, 512),
            VGG_ResBlock(512, 512),
            VGG_ResBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block5: 3×(conv3×3→ReLU) with residual, channels 512→512→512; three times → maxpool
        self.block5 = nn.Sequential(
            VGG_ResBlock(512, 512),
            VGG_ResBlock(512, 512),
            VGG_ResBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # At this point, spatial size is 1×1; number of channels = 512
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.block1(x)   # → 64×16×16
        x = self.block2(x)   # → 128×8×8
        x = self.block3(x)   # → 256×4×4
        x = self.block4(x)   # → 512×2×2
        x = self.block5(x)   # → 512×1×1
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class VGG_A_Sigmoid(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        act = nn.Sigmoid()
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1), act, nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), act, nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1), act,
            nn.Conv2d(256,256,3,padding=1), act, nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,padding=1), act,
            nn.Conv2d(512,512,3,padding=1), act, nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,3,padding=1), act,
            nn.Conv2d(512,512,3,padding=1), act, nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,512), act,
            nn.Linear(512,512), act,
            nn.Linear(512,num_classes)
        )
        if init_weights:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VGG_A_Tanh(VGG_A_Sigmoid):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__(inp_ch, num_classes, init_weights)
        # 用 tanh 替换 sigmoid
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Sigmoid):
                self.features[i] = nn.Tanh()
        for i, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Sigmoid):
                self.classifier[i] = nn.Tanh()

class VGG_A_BN(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch,   out_channels=64,  kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64,       out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            nn.Conv2d(in_channels=128,      out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256,      out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256,      out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512,      out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 5
            nn.Conv2d(in_channels=512,      out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512,      out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)

class VGG_Huge(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        # ───────────────────────────────────────────────────
        # Block1: 2×(conv3×3, 128) → maxpool
        # 原 VGG_Large block1 是 64→64；这里翻倍为 128→128
        # ───────────────────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(inp_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,   128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出： 128 × 16 × 16
        )

        # ───────────────────────────────────────────────────
        # Block2: 2×(conv3×3, 256) → maxpool
        # 原 VGG_Large block2 是 128→128；翻倍为 256→256
        # ───────────────────────────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(128,  256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,  256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出： 256 × 8 × 8
        )

        # ───────────────────────────────────────────────────
        # Block3: 3×(conv3×3, 512) → maxpool
        # 原 VGG_Large block3 是 256→256→256；翻倍为 512→512→512
        # ───────────────────────────────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出： 512 × 4 × 4
        )

        # ───────────────────────────────────────────────────
        # Block4: 3×(conv3×3, 1024) → maxpool
        # 原 VGG_Large block4 是 512→512→512；翻倍为 1024→1024→1024
        # ───────────────────────────────────────────────────
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出： 1024 × 2 × 2
        )

        # ───────────────────────────────────────────────────
        # Block5: 3×(conv3×3, 1024) → maxpool
        # 原 VGG_Large block5 是 512→512→512；翻倍为 1024→1024→1024
        # ───────────────────────────────────────────────────
        self.block5 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出： 1024 × 1 × 1
        )

        # ───────────────────────────────────────────────────
        # Classifier：Linear(1024 → 1024) → ReLU → Linear(1024 → 1024) → ReLU → Linear(1024 → num_classes)
        # 与 VGG_Large 的 512→512→512→num_classes 相比，这里都翻倍为 1024
        # ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

        # 对所有卷积和全连接层做与 VGG_Large 类似的初始化
        self._init_weights()

    def forward(self, x):
        x = self.block1(x)   # → 128×16×16
        x = self.block2(x)   # → 256×8×8
        x = self.block3(x)   # → 512×4×4
        x = self.block4(x)   # → 1024×2×2
        x = self.block5(x)   # → 1024×1×1
        x = x.view(x.size(0), -1)           # 展平成 (batch_size, 1024)
        x = self.classifier(x)              # → (batch_size, num_classes)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1×1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3×3 conv (possibly strided)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1×1 conv to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50 adapted for CIFAR-10 (32×32 inputs)."""

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        # Initial conv modified for CIFAR: 3×3, stride=1, no maxpool
        self.conv1 = nn.Conv2d(inp_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Layers configuration: [3, 4, 6, 3]
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        if init_weights:
            self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        # Check if downsampling is needed (channel or spatial)
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers = []
        # First block may downsample
        layers.append(Bottleneck(in_channels, out_channels, stride, downsample))
        in_c = out_channels * Bottleneck.expansion
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_c, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)   # → 64×32×32
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # → 256×32×32
        x = self.layer2(x)  # → 512×16×16
        x = self.layer3(x)  # → 1024×8×8
        x = self.layer4(x)  # → 2048×4×4

        x = self.avgpool(x) # → 2048×1×1
        x = torch.flatten(x, 1)
        x = self.fc(x)      # → num_classes
        return x


# ---------- ConvNeXt-Large for CIFAR-sized inputs ----------

class ConvNeXtBlock(nn.Module):
    """A single ConvNeXt block."""
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw_conv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        # Input: (B, C, H, W) → depthwise conv
        shortcut = x
        x = self.dw_conv(x)  # → (B, C, H, W)
        # Permute for LayerNorm: (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Feed-forward
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        # Back to channels-first
        x = x.permute(0, 3, 1, 2)
        return x + shortcut


class ConvNeXtStage(nn.Module):
    """ConvNeXt stage: a downsampling (if needed) followed by repeated blocks."""
    def __init__(self, in_dim, out_dim, depth, downsample=True):
        super().__init__()
        self.downsample = None
        if downsample:
            # Use a 2×2 stride-2 conv to halve spatial dimensions and adjust channels
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                nn.LayerNorm(out_dim, eps=1e-6, elementwise_affine=True)
            )
        else:
            # If no downsampling (first stage), just adjust LayerNorm style
            self.downsample = nn.LayerNorm(in_dim, eps=1e-6, elementwise_affine=True)
        # Create sequential ConvNeXt blocks
        blocks = []
        for _ in range(depth):
            blocks.append(ConvNeXtBlock(out_dim))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if isinstance(self.downsample, nn.Sequential):
            x = self.downsample[0](x)  # Conv2d
            # Permute to (B, H, W, C) for LayerNorm, then back
            x = x.permute(0, 2, 3, 1)
            x = self.downsample[1](x)
            x = x.permute(0, 3, 1, 2)
        else:
            # First stage: apply LayerNorm directly on channels-last view
            x = x.permute(0, 2, 3, 1)
            x = self.downsample(x)
            x = x.permute(0, 3, 1, 2)
        x = self.blocks(x)
        return x


class ConvNeXtLarge(nn.Module):
    """ConvNeXt-Large adapted for CIFAR-10 (32×32 inputs)."""
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()
        # Stem: conv4×4, stride=4 → reduces 32×32 to 8×8
        self.stem = nn.Conv2d(inp_ch, 192, kernel_size=4, stride=4, padding=0)

        # Stages configurations for Large: depths=[3,3,27,3], dims=[192,384,768,1536]
        depths = [3, 3, 27, 3]
        dims = [192, 384, 768, 1536]

        # Stage 1: no downsample because stem already reduced to 8×8
        self.stage1 = ConvNeXtStage(dims[0], dims[0], depth=depths[0], downsample=False)
        # Stage 2: 8×8 → 4×4
        self.stage2 = ConvNeXtStage(dims[0], dims[1], depth=depths[1], downsample=True)
        # Stage 3: 4×4 → 2×2
        self.stage3 = ConvNeXtStage(dims[1], dims[2], depth=depths[2], downsample=True)
        # Stage 4: 2×2 → 1×1
        self.stage4 = ConvNeXtStage(dims[2], dims[3], depth=depths[3], downsample=True)

        # Final norm (LayerNorm on channels-last)
        self.norm = nn.LayerNorm(dims[3], eps=1e-6)

        # Classifier head
        self.head = nn.Linear(dims[3], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Reuse Kaiming init for depthwise and pointwise convs
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.stem(x)        # → (B, 192, 8, 8)
        x = self.stage1(x)      # → (B, 192, 8, 8)
        x = self.stage2(x)      # → (B, 384, 4, 4)
        x = self.stage3(x)      # → (B, 768, 2, 2)
        x = self.stage4(x)      # → (B, 1536, 1, 1)
        x = x.permute(0, 2, 3, 1)   # → (B, 1, 1, 1536)
        x = self.norm(x)           # LayerNorm over last dimension
        x = x.view(x.size(0), -1)   # → (B, 1536)
        x = self.head(x)           # → (B, num_classes)
        return x

class ResNet101(nn.Module):
    """ResNet-101 adapted for CIFAR-10 (32×32 inputs)."""
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        # Initial conv and bn for CIFAR
        self.conv1 = nn.Conv2d(inp_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Layers config: [3, 4, 23, 3]
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, blocks=23, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

        # Global pooling and fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        if init_weights:
            self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride, downsample))
        in_c = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_c, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Example usage in __main__ to check parameter counts:
if __name__ == '__main__':
    print("=== Parameter Counts (approximate) ===")
    for model_cls in [VGG_A, VGG_A_Light, VGG_A_Dropout, VGG_Res, VGG_Large,
                      VGG_A_Sigmoid, VGG_A_Tanh, VGG_A_BN, VGG_Huge,
                      ResNet50, ConvNeXtLarge]:
        model = model_cls()
        print(f"{model_cls.__name__}: {get_number_of_parameters(model)} params")