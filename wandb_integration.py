import json
import wandb
from datetime import datetime, timedelta

wandb.login(key='e84f9aa9585ed75083b2923b94f68b05c287fe7e')

all_parts = ["Part1", "Part1 2", "Part1 3", "Part2", "Part2 2", "Part2 3"]

for part_name in all_parts:
    wandb.init(
        # set the wandb project where this run will be logged
        project="ws23-d7-computervision",
        config={
            "learning_rate": 0.0001,
            "architecture": "restormer",
            "dataset": "computervision-simulation-results-5k-images-" + part_name,
            "epochs": 100,
            "part_name": part_name,
            "batch_size": 1,
            "model": "restormer_model.pth"
        },
        name="Restormer-" + part_name + "-2024-01-10"
    )

    file_path = "repo_of_good_outcomes/2024-01-10-restormer-base-architecture/dataset_infos/" + part_name + "_dataset_info.json"

    # Read and parse the JSON file
    with open(file_path, 'r') as file:
        loss_data = json.load(file)

    initial_timestamp = datetime(2024, 1, 7)

    for model_name, histories in loss_data["loss_histories"].items():
        for epoch, (loss, val_loss) in enumerate(zip(histories["training_losses"], histories["validation_losses"])):
            # Use current time or replace with actual timestamp if available
            # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_timestamp = initial_timestamp + timedelta(minutes=20 * epoch)

            timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")

            wandb.log({"epoch": epoch,
                       "loss": loss,
                       "val_loss": val_loss,
                       "timestamp": timestamp_str})

    wandb.finish()
