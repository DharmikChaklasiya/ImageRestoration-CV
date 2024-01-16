import json
import os

# Define the directory path where the JSON files are located
dir_path = "dataset_infos"

# Define the list of part names
part_names = ["Part1", "Part1 2", "Part1 3", "Part2", "Part2 2", "Part2 3"]

# Define the threshold for significant loss difference
threshold = 20

# Initialize the current part name and epoch
current_part_name = None
current_epoch = None

model_run = {
    "current_part_name": current_part_name,
    "current_epoch": current_epoch,
    "loss_histories": {}
}

# Loop through each part name
for part_name in part_names:
    # Define the path to the JSON file
    json_path = os.path.join(dir_path, f"{part_name}_dataset_info.json")

    # Check if the JSON file exists
    if os.path.exists(json_path):
        # Load the JSON data
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract the loss histories
        loss_histories = data["loss_histories"]

        # Loop through each loss history
        for model_name, loss_history in loss_histories.items():
            # Add the loss history to the model run dictionary
            model_run["loss_histories"][part_name] = loss_history


# Loop through each part name
for part_name in part_names:

    loss_history = model_run["loss_histories"][part_name]

    training_losses = loss_history["training_losses"]
    validation_losses = loss_history["validation_losses"]

    if current_part_name is None:
        # Set the current part name
        current_part_name = part_name

        # Set the current epoch
        current_epoch = len(training_losses) - 1

    else:
        # Calculate the difference in length between the current part and the new part
        current_length = len(model_run["loss_histories"][current_part_name]["training_losses"])
        new_length = len(training_losses)
        length_difference = abs(current_length - new_length)

        # Check if the length difference is significant
        if length_difference > threshold:
            # Set the new current part name
            current_part_name = part_name

            # Set the new current epoch
            current_epoch = len(training_losses) - 1

model_run["current_part_name"] = current_part_name
model_run["current_epoch"] = current_epoch

# Write the model run dictionary to a new JSON file
with open("model_run_summary.json", "w") as f:
    json.dump(model_run, f, indent=4)
