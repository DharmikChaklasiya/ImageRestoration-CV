import torch

from base_model_training import load_dataset_infos, load_model_optionally
from models.poseprediction_architecture import PosePredictionModel, FCConfig
from poseprediction_inner_model_training import train_model_on_one_batch
from models.unet_encoder import UNetEncoder

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
                            fcconfig=FCConfig(512, 128, 4)).to(device)

dataset_parts = load_dataset_infos(all_parts)

num_super_batches = 10

model_file_name = "pose_pred_model.pth"

load_model_optionally(model, model_file_name)

for i in range(1, num_super_batches + 1):
    super_batch_info = f"Super-Batch: {i}/{num_super_batches}"
    print(f"Running the model in super-batches - {super_batch_info}")
    for part, dataset_metainfo in dataset_parts.items():
        train_model_on_one_batch(dataset_metainfo, model, device, super_batch_info, model_file_name)

print("Training complete - printing results.")

for part, dataset_metainfo in dataset_parts.items():
    print(dataset_metainfo.to_json())
