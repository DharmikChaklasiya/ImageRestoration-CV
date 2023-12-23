# Import necessary libraries
import os
import random
import webbrowser
from math import inf
from typing import Dict

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from tqdm import tqdm

from base_model_training import Phase
from image_loader import get_image_group_map, get_sorted_image_groups, ImageTensorGroup, PosePredictionLabelDataset
from performance_visualization import LossHistory, update_report_with_losses, ImagePerformance, LabelAndPrediction, \
    update_report_samples_for_epoch
from poseprediction_architecture import PosePredictionModel, FCConfig
from unet_encoder import UNetEncoder

print(f"Using pytorch version: {torch.__version__}")
print(f"Using pytorch cuda version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

sorted_image_groups = get_sorted_image_groups()

sorted_image_tensor_groups = []

image_tensor_group_map: Dict[str, ImageTensorGroup] = {}

focal_stack_indices = [0, 15, 30]

for img_group in tqdm(sorted_image_groups, desc="Preloading images..."):
    image_tensor_group = ImageTensorGroup(img_group, focal_stack_indices)
    image_tensor_group.load_images()
    image_tensor_group_map[img_group.formatted_image_index] = image_tensor_group
    sorted_image_tensor_groups.append(image_tensor_group)

pose_prediction_label_dataset = PosePredictionLabelDataset(sorted_image_tensor_groups)

train_size = int(0.8 * len(pose_prediction_label_dataset))  # 80% for training
val_size = len(pose_prediction_label_dataset) - train_size  # 20% for validation

# Split the dataset
train_dataset, val_dataset = random_split(pose_prediction_label_dataset, [train_size, val_size])

# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

image_group_map = get_image_group_map()

num_samples_for_eval = 100
all_indices = list(range(len(pose_prediction_label_dataset)))
random.shuffle(all_indices)
selected_indices = all_indices[:num_samples_for_eval]
eval_subset = Subset(pose_prediction_label_dataset, selected_indices)
eval_dataloader = torch.utils.data.DataLoader(eval_subset, batch_size=1, shuffle=False)

model = PosePredictionModel(encoder=UNetEncoder(in_channels=3, input_width=512, input_height=512),
                            fcconfig=FCConfig(512, 128, 1)).to(device)

loss_function = torch.nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.00005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Number of training epochs
num_epochs = 1000

html_file_path = 'model_run_pose_pred.html'

loss_history = LossHistory()


def normalize_x_and_y_labels():
    # call this if max/min values of x and y need to be found
    global dataloader_with_progress, i, batch, inputs, pose_prediction_labels, img_group_index
    dataloader_with_progress = tqdm(
        DataLoader(PosePredictionLabelDataset(sorted_image_tensor_groups), batch_size=4, shuffle=False),
        desc="Emit all values")
    max_x = 0
    max_y = 0
    min_x = inf
    min_y = inf
    for i, batch in enumerate(dataloader_with_progress):
        inputs, pose_prediction_labels, img_group_index = batch
        x_values = pose_prediction_labels[:, -2]
        y_values = pose_prediction_labels[:, -1]

        max_x = max(max_x, x_values.max().item())
        max_y = max(max_y, y_values.max().item())

        min_x = min(min_x, x_values.min().item())
        min_y = min(min_y, y_values.min().item())
    print(f"Max X: {max_x}, Max Y: {max_y}, Min X: {min_x}, Min Y: {min_y}")


# normalize_x_and_y_labels()

def evaluate_rows_print_images_to_report():
    global inputs, pose_prediction_labels, outputs, loss_history, i
    model.eval()
    performance = []
    with torch.no_grad():

        eval_dataloader_with_progress = tqdm(eval_dataloader, desc="Processing Evaluation")

        for inputs, pose_prediction_labels, img_group_indices in eval_dataloader_with_progress:
            inputs, pose_prediction_labels = inputs.to(device), pose_prediction_labels.to(device)
            outputs = model(inputs)

            for pose_pred_label, out in zip(pose_prediction_labels, outputs):
                loss = loss_function(out, pose_pred_label)
                first_img_group_index = img_group_indices[0]
                gt = image_tensor_group_map[first_img_group_index].ground_truth_tensor
                performance.append(
                    ImagePerformance(loss.item(), gt,
                                     LabelAndPrediction(pose_pred_label, out), image_group_map[first_img_group_index]))

    performance.sort(key=lambda x: x.metric)
    ranks = ["1st-Best", "2nd-Best", "3rd-Best", "3rd-Worst", "2nd-Worst", "1st-Worst"]
    top_and_bottom_images = []
    for i, img_perf in enumerate(performance[:3] + performance[-3:]):
        img_perf.rank = ranks[i]
        top_and_bottom_images.append(img_perf)
    update_report_samples_for_epoch(epoch + 1, top_and_bottom_images, html_file_path)
    if epoch == 0:
        webbrowser.open('file://' + os.path.realpath(html_file_path))


# Training loop
for epoch in range(num_epochs):
    model.train()

    dataloader_with_progress = tqdm(train_loader, desc="Processing Batches")

    for i, batch in enumerate(dataloader_with_progress):
        inputs, pose_prediction_labels, img_group_index = batch

        inputs, pose_prediction_labels = inputs.to(device), pose_prediction_labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_function(outputs, pose_prediction_labels)

        try:
            loss.backward()
        except RuntimeError as e:
            print(f"Error in epoch {epoch}, batch {i}")
            print(f"Inputs type: {inputs.dtype}, Labels type: {pose_prediction_labels.dtype}")
            print(f"Model outputs type: {outputs.dtype}, Loss value type: {loss.dtype}")
            raise e

        optimizer.step()

        loss_history.add_current_running_loss(i, loss.item(), Phase.TRAINING)
        current_loss = loss.item()

        dataloader_with_progress.set_description(
            f"Epoch {epoch + 1}/{num_epochs} - Batch {i + 1}/{len(train_loader)} "
            f"Processing {img_group_index}, Loss: {current_loss:.4f}, "
            f"Avg loss so far: {loss_history.current_avg_train_loss:.4f}"
        )

        if i % 20 == 0 or i == len(train_loader) - 1:
            update_report_with_losses(epoch + 1, loss_history, html_file_path)

    epoch_loss = loss_history.running_loss / len(train_loader)

    model.eval()

    val_dataloader_with_progress = tqdm(val_loader, desc="Processing Validation")

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader_with_progress):
            inputs, pose_prediction_labels, img_group_index = batch

            inputs, pose_prediction_labels = inputs.to(device), pose_prediction_labels.to(device)

            outputs = model(inputs)

            loss = loss_function(outputs, pose_prediction_labels)
            loss_history.add_current_running_loss(i, loss.item(), Phase.VALIDATION)

            if i % 20 == 0 or i == len(val_loader) - 1:
                update_report_with_losses(epoch + 1, loss_history, html_file_path)

    avg_val_loss = (loss_history.running_loss / len(val_loader))

    loss_history.add_loss(epoch_loss, avg_val_loss)

    scheduler.step()

    evaluate_rows_print_images_to_report()

    print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

print("Training complete")
