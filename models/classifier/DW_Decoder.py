from . import *


class Decoder2D(nn.Module):

    def __init__(self, blocks=2, message_length = 128, attention=None):
        super(Decoder2D, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down2D(16, 32, blocks=blocks)
        self.down2 = Down2D(32, 64, blocks=blocks)
        self.down3 = Down2D(64, 128, blocks=blocks)

        self.down4 = Down2D(128, 256, blocks=blocks)
        self.down5 = Down2D(256, 512, blocks=blocks)

        self.up4 = UP2D(512, 256)
        self.att4 = ResBlock2D(256 * 2, 256, blocks=blocks, attention=attention)

        self.up3 = UP2D(256, 128)
        self.att3 = ResBlock2D(128 * 2, 128, blocks=blocks, attention=attention)

        self.up2 = UP2D(128, 64)
        self.att2 = ResBlock2D(64 * 2, 64, blocks=blocks, attention=attention)

        self.up1 = UP2D(64, 32)
        self.att1 = ResBlock2D(32 * 2, 32, blocks=blocks, attention=attention)

        self.up0 = UP2D(32, 16)
        self.att0 = ResBlock2D(16 * 2, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16, 2, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        self.message_layer = nn.Linear(message_length * message_length, message_length)
        
        self.audio_layer = nn.Sequential(
            nn.Linear(message_length * message_length, audio_length),
            nn.Tanh(),
            nn.Linear(audio_length, audio_length)
        )
        '''
        self.message_layer = nn.Linear(2*message_length*message_length, 2*message_length*4)
        self.message_length = message_length



    def forward(self, x):
        _, _, H, W = x.shape

        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u4 = self.up4(d5)
        u4 = torch.cat((d4, u4), dim=1)
        u4 = self.att4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat((d3, u3), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat((d2, u2), dim=1)
        u2 = self.att2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat((d1, u1), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        u0 = torch.cat((d0, u0), dim=1)
        u0 = self.att0(u0)

        residual = self.Conv_1x1(u0) #BT*2*512*512

        message = F.interpolate(residual, size=(self.message_length, self.message_length),
                                                           mode='nearest') #BT*2*128*128
        message = message.view(message.shape[0], -1) #BT, 2*128*128
        message = self.message_layer(message)     
        message = message.view(message.shape[0], 2, self.message_length, -1)                       

        return message

class Decoder1D(nn.Module):
    """
    Video_Decoder(2D Conv) -> 1D Conv 기반의 U-Net 스타일 Audio_Decoder로 변환 예시.
    입력 x: (Batch, Channel, Time)
    """
    def __init__(self, 
                 in_channels=1,   # 오디오 채널 수 (모노=1, 스테레오=2 등)
                 blocks=2, 
                 attention=None   # "se" 등으로 설정 가능 (None 이면 사용 안 함)
                 ):
        super(Decoder1D, self).__init__()

        # 임의로 출력 채널 수를 16부터 시작 (원하시는 구조에 맞게 조정 가능)
        self.conv1 = ConvBlock1D(in_channels, 16, blocks=blocks)
        self.down1 = Down1D(16, 32, blocks=blocks)
        self.down2 = Down1D(32, 64, blocks=blocks)
        self.down3 = Down1D(64, 128, blocks=blocks)
        self.down4 = Down1D(128, 256, blocks=blocks)

        self.up3 = Up1D(256, 128)
        self.att3 = ResBlock1D(256, 128, blocks=blocks, attention=attention)

        self.up2 = Up1D(128, 64)
        self.att2 = ResBlock1D(128, 64, blocks=blocks, attention=attention)

        self.up1 = Up1D(64, 32)
        self.att1 = ResBlock1D(64, 32, blocks=blocks, attention=attention)

        self.up0 = Up1D(32, 16)
        self.att0 = ResBlock1D(32, 16, blocks=blocks, attention=attention)

        self.out_conv = nn.Conv1d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, C, T)
        """
        # ---- Encoding (Down) ----
        d0 = self.conv1(x)   # (B,16,T)
        d1 = self.down1(d0)  # (B,32,T/2)
        d2 = self.down2(d1)  # (B,64,T/4)
        d3 = self.down3(d2)  # (B,128,T/8)
        d4 = self.down4(d3)  # (B,256,T/16)

        u3 = self.up3(d4)    # (B,128,T/8)
        u3 = torch.cat((d3, u3), dim=1)  # (B,128+128=256,T/8)
        u3 = self.att3(u3)

        u2 = self.up2(u3)    # (B,64,T/4)
        u2 = torch.cat((d2, u2), dim=1)  # (B,64+64=128,T/4)
        u2 = self.att2(u2)

        u1 = self.up1(u2)    # (B,32,T/2)
        u1 = torch.cat((d1, u1), dim=1)  # (B,32+32=64,T/2)
        u1 = self.att1(u1)

        u0 = self.up0(u1)    # (B,16,T)
        u0 = torch.cat((d0, u0), dim=1)  # (B,16+16=32,T)
        u0 = self.att0(u0)

        # ---- Output ----
        out = self.out_conv(u0)  # (B,1,T)
        out = self.sigmoid(out)

        return out


class Down1D(nn.Module):
    """
    stride=2를 이용한 downsampling 후, 추가적인 ConvBlock(블록 반복)을 수행.
    """
    def __init__(self, in_channels, out_channels, blocks=1):
        super(Down1D, self).__init__()
        self.layer = nn.Sequential(
            ConvBlock1D(in_channels, in_channels, blocks=1, stride=2),  # downsampling
            ConvBlock1D(in_channels, out_channels, blocks=blocks, stride=1)
        )

    def forward(self, x):
        return self.layer(x)

class Up1D(nn.Module):
    """
    2배 Upsampling (F.interpolate) 후, ConvBlock(블록 반복)을 수행.
    """
    def __init__(self, in_channels, out_channels, blocks=1):
        super(Up1D, self).__init__()
        self.conv = ConvBlock1D(in_channels, out_channels, blocks=blocks, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 시간축 2배
        return self.conv(x)


class Down2D(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(Down2D, self).__init__()
        self.layer = torch.nn.Sequential(
            ConvBlock(in_channels, in_channels, stride=2),
            ConvBlock(in_channels, out_channels, blocks=blocks)
        )

    def forward(self, x):
        return self.layer(x)


class UP2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP2D, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)