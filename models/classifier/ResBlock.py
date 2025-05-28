import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAttention1D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(SEAttention1D, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (Batch, Channels, 1) 형태로 평균 풀링
            nn.Conv1d(in_channels, out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // reduction, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.se(x) * x

class SEAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(SEAttention2D, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels // reduction, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.se(x) * x
        return x

class BottleneckBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8, stride=1, attention=True):
        super(BottleneckBlock1D, self).__init__()

        # skip connection(지름길) 경로가 필요할 경우(채널 수 변경 or stride != 1)
        self.change = None
        if (in_channels != out_channels or stride != 1):
            self.change = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm1d(out_channels)
            )

        # 메인 경로
        self.left = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm1d(out_channels)
        )

        # Squeeze-and-Excitation Attention
        if attention:
            self.attention = SEAttention1D(out_channels, out_channels, reduction=reduction)
        else:
            self.attention = nn.Identity()  # attention을 끄고 싶으면 Identity로 대체

    def forward(self, x):
        identity = x
        out = self.left(x)
        out = self.attention(out)

        if self.change is not None:
            identity = self.change(identity)

        out = out + identity
        out = F.relu(out)
        return out

class ResBlock1D(nn.Module):
    """
    blocks: 이 블록 안에 몇 개의 BottleneckBlock1D(또는 BasicBlock 등)를 쌓을지 결정
    """
    def __init__(self, in_channels, out_channels, blocks=1, block_type="BottleneckBlock1D",
                 reduction=8, stride=1, attention=True):
        super(ResBlock1D, self).__init__()
        layers = []

        # 첫 번째 블록(채널이 바뀌거나 stride가 필요할 수 있으므로)
        if blocks > 0:
            layers.append(eval(block_type)(in_channels, out_channels, reduction, stride, attention=attention))

        # 이후 블록(채널/stride 동일)
        for _ in range(blocks - 1):
            layers.append(eval(block_type)(out_channels, out_channels, reduction, 1, attention=attention))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class BottleneckBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction, stride, attention=None):
        super(BottleneckBlock2D, self).__init__()

        self.change = None
        if (in_channels != out_channels or stride != 1):
            self.change = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                          stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      stride=stride, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        self.attention = SEAttention2D(in_channels=out_channels, out_channels=out_channels, reduction=reduction)
        

    def forward(self, x):
        identity = x
        x = self.left(x)
        x = self.attention(x)

        if self.change is not None:
            identity = self.change(identity)

        x += identity
        x = F.relu(x)
        return x


class ResBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, blocks=1, block_type="BottleneckBlock2D", reduction=8, stride=1, attention=None):
        super(ResBlock2D, self).__init__()

        layers = [eval(block_type)(in_channels, out_channels, reduction, stride, attention=attention)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = eval(block_type)(out_channels, out_channels, reduction, 1, attention=attention)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
