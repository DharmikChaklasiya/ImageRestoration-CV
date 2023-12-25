# Import necessary libraries
from typing import Dict

import torch

from base_model_training import DatasetPartMetaInfo
from image_loader import load_input_image_parts
from unet_architecture import UNet

from pytorch_msssim import SSIM

from unet_inner_model_training import train_model_on_one_batch

ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)


def ssim_based_loss(output, target):
    # Calculate SSIM loss
    loss = 1 - ssim_loss(output, target)  # 1 - SSIM since we want to minimize the loss
    return loss


print(f"Using pytorch version: {torch.__version__}")
print(f"Using pytorch cuda version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

all_parts = ["Part1", "Part1 2", "Part1 3", "Part2", "Part2 2", "Part2 3"]

model = UNet(in_channels=10).to(device)

dataset_parts: Dict[str, DatasetPartMetaInfo] = {}

for part in all_parts:
    all_image_groups, image_group_map = load_input_image_parts([part])
    dataset_parts[part] = DatasetPartMetaInfo(part, all_image_groups, image_group_map)

num_super_batches = 10

for i in range(1, num_super_batches + 1):
    super_batch_info = f"Super-Batch: {i}/{num_super_batches}"
    print(f"Running the model in super-batches - {super_batch_info}")
    for part, dataset_metainfo in dataset_parts.items():
        train_model_on_one_batch(dataset_metainfo, model, device, super_batch_info)

print("Training complete - printing results.")

for part, dataset_metainfo in dataset_parts.items():
    print(dataset_metainfo.to_json())
