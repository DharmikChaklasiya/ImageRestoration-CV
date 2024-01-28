# Import necessary libraries
from typing import Dict

import torch
import wandb
from pytorch_msssim import SSIM

from base_model_training import load_dataset_infos, load_model_run_summary, DatasetPartMetaInfo, \
    ModelRunSummary, wandb_api_key, find_existing_run
from models.restormer import Restormer
from restormer_inner_model_training import train_model_on_one_batch

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

model = Restormer().to(device)

dataset_parts: Dict[str, DatasetPartMetaInfo] = load_dataset_infos(all_parts)

num_super_batches = 10

wandb.login(key=wandb_api_key)

def wandbinit(part_name: str):
    run_name = "Restormer-augm-loss-" + part_name + "-2024-01-19"
    project_name = "ws23-d7-computervision"

    current_run = find_existing_run(project_name, run_name)

    wandb.init(
        resume=current_run.id if current_run is not None else None,
        project=project_name,
        config={
            "learning_rate": 0.0001,
            "architecture": "restormer-augm-loss",
            "dataset": "computervision-simulation-results-5k-images-" + part_name,
            "epochs": 100,
            "part_name": part_name,
            "batch_size": 1,
            "model": "restormer_model_augm_loss.pth"
        },
        name=run_name
    )


model_run_summary: ModelRunSummary = load_model_run_summary("2024-01-19-restormer-augm-loss", model)

model_run_summary.train_model_on_all_batches(dataset_parts, num_super_batches, wandbinit, train_model_on_one_batch)
