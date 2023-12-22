import torch
import torch.nn as nn

from unet_decoder import UNetDecoder
from unet_encoder import UNetEncoder


class UNet(nn.Module):
    def __init__(self, in_channels=11, out_channels=1, input_height=512, input_width=512):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width

        # Initialize the U-Net encoder and decoder
        self.encoder = UNetEncoder(in_channels=in_channels, input_height=input_height, input_width=input_width)
        self.decoder = UNetDecoder(out_channels_sequence=self.encoder.out_channels_sequence, out_channels=out_channels)

    def forward(self, x):
        # Contracting Path (Encoder)
        encoder_features = self.encoder(x)

        # Expansive Path (Decoder)
        out = self.decoder(encoder_features, encoder_features)

        # Assert the final output dimensions
        assert out.shape == (x.shape[0], self.out_channels, self.input_height, self.input_width), \
            f"Expected output shape ({x.shape[0]}, {self.out_channels}, {self.input_height}, {self.input_width}), got {out.shape}"

        return out
