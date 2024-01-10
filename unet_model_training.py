# Import necessary libraries

import torch

from base_model_training import load_dataset_infos, load_model_and_history
from models.unet_architecture import UNet

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

dataset_parts = load_dataset_infos(all_parts)

num_super_batches = 10

model_file_name = "unet_model.pth"

load_model_and_history(model, model_file_name)

for i in range(1, num_super_batches + 1):
    super_batch_info = f"Super-Batch: {i}/{num_super_batches}"
    print(f"Running the model in super-batches - {super_batch_info}")
    for part, dataset_metainfo in dataset_parts.items():
        train_model_on_one_batch(dataset_metainfo, model, device, super_batch_info, model_file_name)

print("Training complete - printing results.")

for part, dataset_metainfo in dataset_parts.items():
    print(dataset_metainfo.to_json())
