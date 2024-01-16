# Import necessary libraries
from typing import Dict

import torch
import wandb
from pytorch_msssim import SSIM

from base_model_training import load_dataset_infos, load_model_optionally, load_model_run_summary, DatasetPartMetaInfo, \
    ModelRunSummary
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

wandb.login(key='e84f9aa9585ed75083b2923b94f68b05c287fe7e')


def wandbinit(part_name: str):
    run_name = "Restormer-" + part_name + "-2024-01-10"
    project_name = "ws23-d7-computervision"

    api = wandb.Api()
    runs = api.runs(f"{project_name}")
    current_run = None
    for run in runs:
        if run.name == run_name:
            current_run = run
            break

    if current_run:
        print("Wand will resume run : "+current_run.id)
    else:
        print("No existing run found in this project fitting this run name - so we are starting a new run from scratch")

    wandb.init(
        resume=current_run.id if current_run is not None else None,
        project=project_name,
        config={
            "learning_rate": 0.0001,
            "architecture": "restormer",
            "dataset": "computervision-simulation-results-5k-images-" + part_name,
            "epochs": 100,
            "part_name": part_name,
            "batch_size": 1,
            "model": "restormer_model.pth"
        },
        name=run_name
    )


model_run_summary: ModelRunSummary = load_model_run_summary("2024-01-10-restormer-base-architecture", model)

model_run_summary.train_model_on_all_batches(dataset_parts, num_super_batches, wandbinit, train_model_on_one_batch)
