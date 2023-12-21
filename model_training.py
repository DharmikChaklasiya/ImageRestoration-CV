# Import necessary libraries
import os
import webbrowser
import random

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from image_loader import get_dataset, get_image_group_map
from performance_visualization import update_html_for_epoch, ImagePerformance
from unet_architecture import UNet

from torch.utils.data import DataLoader, random_split

print(f"Using pytorch version: {torch.__version__}")
print(f"Using pytorch cuda version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Define the ratio or number of items for the split
train_size = int(0.8 * len(get_dataset()))  # 80% for training
val_size = len(get_dataset()) - train_size  # 20% for validation

# Split the dataset
train_dataset, val_dataset = random_split(get_dataset(), [train_size, val_size])

# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

image_group_map = get_image_group_map()

num_samples_for_eval = 100
all_indices = list(range(len(get_dataset())))
random.shuffle(all_indices)
selected_indices = all_indices[:num_samples_for_eval]
eval_subset = Subset(get_dataset(), selected_indices)
eval_dataloader = torch.utils.data.DataLoader(eval_subset, batch_size=1, shuffle=False)

model = UNet(in_channels=3).to(device)

# Assuming you have already defined 'model', 'dataloader', and 'dataset'
# Define your loss function and optimizer
loss_function = F.mse_loss  # or any other appropriate loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # learning rate

# Number of training epochs
num_epochs = 1000


def evaluate_rows_print_images_to_report():
    global inputs, ground_truth, outputs, loss, i
    model.eval()
    performance = []
    with torch.no_grad():

        eval_dataloader_with_progress = tqdm(eval_dataloader, desc="Processing Evaluation")

        for inputs, ground_truth, img_group_indices in eval_dataloader_with_progress:
            inputs, ground_truth = inputs.to(device), ground_truth.to(device)
            outputs = model(inputs)

            for gt, out in zip(ground_truth, outputs):
                loss = loss_function(outputs, ground_truth)
                performance.append(ImagePerformance(loss.item(), gt, out, image_group_map[img_group_indices[0]]))
    performance.sort(key=lambda x: x.metric)
    ranks = ["1st-Best", "2nd-Best", "3rd-Best", "3rd-Worst", "2nd-Worst", "1st-Worst"]
    top_and_bottom_images = []
    for i, img_perf in enumerate(performance[:3] + performance[-3:]):
        img_perf.rank = ranks[i]
        top_and_bottom_images.append(img_perf)
    html_file_path = 'model_run.html'
    update_html_for_epoch(epoch + 1, top_and_bottom_images, html_file_path)
    if epoch == 0:
        webbrowser.open('file://' + os.path.realpath(html_file_path))


# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    dataloader_with_progress = tqdm(train_loader, desc="Processing Batches")

    for i, batch in enumerate(dataloader_with_progress):
        inputs, ground_truth, img_group = batch

        inputs, ground_truth = inputs.to(device), ground_truth.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_function(outputs, ground_truth)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        current_loss = loss.item()

        dataloader_with_progress.set_description(
            f"Epoch {epoch + 1}/{num_epochs} - Batch {i + 1}/{len(train_loader)} Processing {img_group}, Loss: {current_loss:.4f}, Avg loss so far: {running_loss / (i + 1):.4f}"
        )

    # Print average loss for the epoch
    epoch_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0

    val_dataloader_with_progress = tqdm(val_loader, desc="Processing Validation")

    with torch.no_grad():
        for inputs, ground_truth, img_group in val_dataloader_with_progress:
            inputs, ground_truth = inputs.to(device), ground_truth.to(device)
            outputs = model(inputs)

            loss = loss_function(outputs, ground_truth)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    evaluate_rows_print_images_to_report()

    print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

print("Training complete")
