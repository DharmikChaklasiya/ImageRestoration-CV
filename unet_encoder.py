import torch.nn as nn


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, input_height, input_width, out_channels_sequence=None, pool_kernel_size=2, pool_stride=2):
        super(UNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.input_width = input_width
        self.input_height = input_height
        self.pool_stride = pool_stride
        self.pool_kernel_size = pool_kernel_size
        self.out_channels_sequence = [16, 32, 64, 128, 256] if out_channels_sequence is None else out_channels_sequence

        self.enc_conv0 = self.contract_block(in_channels, self.out_channels_sequence[0])
        self.enc_conv1 = self.contract_block(self.out_channels_sequence[0], self.out_channels_sequence[1])
        self.enc_conv2 = self.contract_block(self.out_channels_sequence[1], self.out_channels_sequence[2])
        self.enc_conv3 = self.contract_block(self.out_channels_sequence[2], self.out_channels_sequence[3])
        self.bottleneck = self.contract_block(self.out_channels_sequence[3], self.out_channels_sequence[4])
        self.downsampling_factor = self.calculate_downsampling_factor()

    def calculate_downsampling_factor(self):
        # Assuming each contracting block has a max pooling layer that downsamples the image
        return self.pool_stride ** len(self.out_channels_sequence)

    def contract_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def forward(self, x):
        # Check initial input dimensions
        assert x.shape[1] == self.in_channels, f"Expected input with {self.in_channels} channels, got {x.shape[1]}"
        assert x.shape[2] == self.input_height and x.shape[3] == self.input_width, \
            f"Expected input with dimensions ({self.input_height}, {self.input_width}), got ({x.shape[2]}, {x.shape[3]})"

        features = []

        for stage in [self.enc_conv0, self.enc_conv1, self.enc_conv2, self.enc_conv3, self.bottleneck]:
            x = stage(x)
            features.append(x)

        return features
