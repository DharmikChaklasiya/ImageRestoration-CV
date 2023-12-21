import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=11, out_channels=1, input_height=512, input_width=512):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width

        # Define the number of channels at each stage of the U-Net
        self.out_channels_sequence = [16, 32, 64, 128, 256]

        # Contracting Path (Encoder)
        self.enc_conv0 = self.contract_block(in_channels, self.out_channels_sequence[0])
        self.enc_conv1 = self.contract_block(self.out_channels_sequence[0], self.out_channels_sequence[1])
        self.enc_conv2 = self.contract_block(self.out_channels_sequence[1], self.out_channels_sequence[2])
        self.enc_conv3 = self.contract_block(self.out_channels_sequence[2], self.out_channels_sequence[3])
        self.bottleneck = self.contract_block(self.out_channels_sequence[3], self.out_channels_sequence[4])

        # Expansive Path (Decoder)
        self.dec_conv4 = self.expand_block(self.out_channels_sequence[4], self.out_channels_sequence[3])
        self.dec_conv3 = self.expand_block(self.out_channels_sequence[3] * 2, self.out_channels_sequence[2])
        self.dec_conv2 = self.expand_block(self.out_channels_sequence[2] * 2, self.out_channels_sequence[1])
        self.dec_conv1 = self.expand_block(self.out_channels_sequence[1] * 2, self.out_channels_sequence[0])
        self.dec_conv0 = self.expand_block(self.out_channels_sequence[0] * 2, self.out_channels_sequence[0])

        # Final output layer to map to the desired number of output channels
        self.final_conv = nn.Conv2d(self.out_channels_sequence[0], out_channels, kernel_size=1)

    def contract_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

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

    def forward(self, x):
        # Check initial input dimensions
        assert x.shape[1] == self.in_channels, f"Expected input with {self.in_channels} channels, got {x.shape[1]}"
        assert x.shape[2] == self.input_height and x.shape[3] == self.input_width, \
            f"Expected input with dimensions ({self.input_height}, {self.input_width}), got ({x.shape[2]}, {x.shape[3]})"

        # Contracting Path
        c0 = self.enc_conv0(x)
        c1 = self.enc_conv1(c0)
        c2 = self.enc_conv2(c1)
        c3 = self.enc_conv3(c2)
        c4 = self.bottleneck(c3)

        # Expansive Path
        d4 = self.dec_conv4(c4)
        d3 = self.dec_conv3(torch.cat([d4, c3], 1))  # skip connection: concatenate along the channels axis (N, C, H, W)
        d2 = self.dec_conv2(torch.cat([d3, c2], 1))
        d1 = self.dec_conv1(torch.cat([d2, c1], 1))
        d0 = self.dec_conv0(torch.cat([d1, c0], 1))

        # Final output layer
        out = self.final_conv(d0)

        # Assert the final output dimensions
        assert out.shape == (x.shape[0], self.out_channels, self.input_height, self.input_width), \
            f"Expected output shape ({x.shape[0]}, {self.out_channels}, {self.input_height}, {self.input_width}), got {out.shape}"

        return out

    def up(self, x):
        """ Upsampling function to align the feature maps for concatenation """
        return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


# Instantiate the UNet model with the correct input and output channels
model = UNet(in_channels=11, out_channels=1)
print(model)  # Print the model architecture
