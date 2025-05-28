import torch.nn as nn


class ConvINRelu(nn.Module):
	"""
	A sequence of Convolution, Instance Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride):
		super(ConvINRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
			nn.InstanceNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class ConvBlock(nn.Module):
	'''
	Network that composed by layers of ConvINRelu
	'''

	def __init__(self, in_channels, out_channels, blocks=1, stride=1):
		super(ConvBlock, self).__init__()

		layers = [ConvINRelu(in_channels, out_channels, stride)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvINRelu(out_channels, out_channels, 1)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

class ConvINRelu1D(nn.Module):
    """
    A sequence of Convolution(1D), InstanceNorm(1D), and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvINRelu1D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm1d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ConvBlock1D(nn.Module):
    """
    Network composed of multiple ConvINRelu1D layers.
    - blocks: how many ConvINRelu1D layers to stack
    - stride: only the first layer uses this stride
    """
    def __init__(self, in_channels, out_channels, blocks=1, stride=1):
        super(ConvBlock1D, self).__init__()

        layers = []
        # 첫 번째 레이어 (stride 적용)
        if blocks > 0:
            layers.append(ConvINRelu1D(channels_in=in_channels, channels_out=out_channels, stride=stride))

        # 이후 레이어 (stride=1)
        for _ in range(blocks - 1):
            layers.append(ConvINRelu1D(channels_in=out_channels, channels_out=out_channels, stride=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)