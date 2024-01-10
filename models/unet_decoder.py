import torch
from torch import nn


class UNetDecoder(nn.Module):
    def __init__(self, out_channels_sequence, out_channels):
        super(UNetDecoder, self).__init__()

        self.out_channels_sequence = out_channels_sequence

        self.dec_conv4 = self.expand_block(out_channels_sequence[4], out_channels_sequence[3])
        self.dec_conv3 = self.expand_block(out_channels_sequence[3] * 2, out_channels_sequence[2])
        self.dec_conv2 = self.expand_block(out_channels_sequence[2] * 2, out_channels_sequence[1])
        self.dec_conv1 = self.expand_block(out_channels_sequence[1] * 2, out_channels_sequence[0])
        self.dec_conv0 = self.expand_block(out_channels_sequence[0] * 2, out_channels_sequence[0])

        # Final output layer to map to the desired number of output channels
        self.final_conv = nn.Conv2d(out_channels_sequence[0], out_channels, kernel_size=1)

    def expand_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU()
        )
        return block

    def forward(self, encoder_features):
        d = encoder_features[4]

        d4 = self.dec_conv4(d)
        d3 = self.dec_conv3(torch.cat([d4, encoder_features[3]], 1))  # Skip connection from enc_conv3 output
        d2 = self.dec_conv2(torch.cat([d3, encoder_features[2]], 1))  # Skip connection from enc_conv2 output
        d1 = self.dec_conv1(torch.cat([d2, encoder_features[1]], 1))  # Skip connection from enc_conv1 output
        d0 = self.dec_conv0(torch.cat([d1, encoder_features[0]], 1))  # Skip connection from enc_conv0 output

        # Final output layer
        out = self.final_conv(d0)

        return out
