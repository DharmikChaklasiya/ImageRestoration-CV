from typing import Dict

import torch

from image_loader import load_input_image_parts
from base_model_training import DatasetPartMetaInfo
from poseprediction_architecture import PosePredictionModel, FCConfig
from poseprediction_inner_model_training import train_model_on_one_batch
from unet_encoder import UNetEncoder

print(f"Using pytorch version: {torch.__version__}")
print(f"Using pytorch cuda version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

all_parts = ["Part1", "Part1 2", "Part1 3", "Part2", "Part2 2", "Part2 3"]

model = PosePredictionModel(encoder=UNetEncoder(in_channels=10, input_width=512, input_height=512),
                            fcconfig=FCConfig(512, 128, 2)).to(device)

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
