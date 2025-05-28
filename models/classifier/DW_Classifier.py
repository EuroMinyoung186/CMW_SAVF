from . import *
from .DW_Encoder import Encoder2D
from .DW_Decoder import Decoder2D

class DW_EncoderDecoder(nn.Module):

    def __init__(self):
        super(DW_EncoderDecoder, self).__init__()
        self.encoder = Encoder2D()
        self.decoder = Decoder2D(attention = 'se')


    def forward(self, video, audio=None, execute_type = 'encoder'):
        if execute_type == 'encoder':
            out = self.encoder(video, audio)
        elif execute_type == 'decoder':
            out = self.decoder(video)
        else:
            print('Wrong Execute Type !!!')

        return out