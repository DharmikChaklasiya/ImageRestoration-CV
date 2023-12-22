# Import necessary libraries
import random
from math import inf

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from tqdm import tqdm

from base_model_training import Phase
from image_loader import get_image_group_map, get_sorted_image_groups, ImageTensorGroup, PosePredictionLabelDataset
from performance_visualization import LossHistory, update_report_with_losses
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

focal_stack_indices = [0, 15, 30]

for img_group in tqdm(sorted_image_groups, desc="Preloading images..."):
    image_tensor_group = ImageTensorGroup(img_group, focal_stack_indices)
    image_tensor_group.load_images()
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

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Consider starting with a lower lr, e.g., 0.0001
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # Adjust step_size and gamma as needed

# Number of training epochs
num_epochs = 1000

html_file_path = 'model_run.html'

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

    print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

print("Training complete")
