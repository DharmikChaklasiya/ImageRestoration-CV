import torch
from torch import nn
import torch.nn.functional as f

from unet_encoder import UNetEncoder


class FCConfig:
    def __init__(self, fc1_out, fc2_out, output_size):
        self.fc1_out = fc1_out
        self.fc2_out = fc2_out
        self.output_size = output_size


class PosePredictionModel(nn.Module):
    def __init__(self, encoder: UNetEncoder, fcconfig: FCConfig = FCConfig(512, 128, 7)):
        super(PosePredictionModel, self).__init__()
        self.encoder = encoder  # U-Net's encoder

        output_height = self.encoder.input_height // self.encoder.downsampling_factor
        output_width = self.encoder.input_width // self.encoder.downsampling_factor

        final_out_channels = self.encoder.out_channels_sequence[-1]
        encoder_output_features = final_out_channels * output_height * output_width

        # Additional layers for pose estimation
        self.fc1 = nn.Linear(encoder_output_features, fcconfig.fc1_out)
        self.fc2 = nn.Linear(fcconfig.fc1_out, fcconfig.fc2_out)
        self.fc3 = nn.Linear(fcconfig.fc2_out, fcconfig.output_size)

    def forward(self, x):
        x = self.encoder(x)[-1]  # encoder bottleneck features
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)  # Output pose parameters
        return x
