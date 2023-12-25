import base64
import json
import random
from enum import Enum
from io import BytesIO
from math import inf
from typing import List, Dict, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from LFR.python.image_group import ImageGroup
from image_loader import PosePredictionLabelDataset


class Phase(Enum):
    TRAINING = "training"
    VALIDATION = "validation"


class LossHistory:
    def __init__(self):
        self.running_loss = 0.0
        self.current_avg_val_loss = None
        self.current_avg_train_loss = None
        self.training_losses = []
        self.validation_losses = []

    def to_json(self):
        return {
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses
        }

    def add_loss(self, training_loss, validation_loss):
        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)
        self.running_loss = 0.0
        self.current_avg_val_loss = None
        self.current_avg_train_loss = None

    def get_loss_plot_base64(self):
        plt.figure()
        epochs = np.arange(1, len(self.training_losses) + 1)
        plt.plot(epochs, self.training_losses, label='Training Loss')
        plt.plot(epochs, self.validation_losses, label='Validation Loss')

        current_epoch = len(epochs) + 1

        # Plot current average losses if available
        if self.current_avg_train_loss is not None:
            plt.scatter(current_epoch, self.current_avg_train_loss, color='blue', marker='x',
                        label='Current Avg Training Loss')

        if self.current_avg_val_loss is not None:
            plt.scatter(current_epoch, self.current_avg_val_loss, color='orange', marker='x',
                        label='Current Avg Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        max_loss = 0
        if self.training_losses and self.validation_losses:
            max_loss = max(max(self.training_losses), max(self.validation_losses))
        if self.current_avg_val_loss:
            max_loss = max(max_loss, self.current_avg_val_loss)
        if self.current_avg_train_loss:
            max_loss = max(max_loss, self.current_avg_train_loss)
        plt.ylim(bottom=0, top=max_loss + 0.1 * max_loss)

        plt.xticks(epochs)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()  # Close the figure to free memory
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        return base64.b64encode(image_png).decode('utf-8')

    def add_current_running_loss(self, batch_num, running_loss, phase):
        if phase == Phase.TRAINING:
            if batch_num == 0:
                self.running_loss = 0.0
            self.running_loss += running_loss
            self.current_avg_train_loss = self.running_loss / (batch_num + 1)
        elif phase == Phase.VALIDATION:
            if batch_num == 0:
                self.running_loss = 0.0
            self.running_loss += running_loss
            self.current_avg_val_loss = self.running_loss / (batch_num + 1)
        else:
            raise ValueError("Unknown phase : " + phase)


class DatasetPartMetaInfo:
    def __init__(self, part_name: str, all_image_groups: List[ImageGroup], image_group_map: Dict[str, ImageGroup]):
        assert part_name is not None
        self.part_name = part_name
        self.image_group_map = image_group_map
        self.total_size = len(all_image_groups)
        self.val_ratio = 0.1
        self.num_samples_for_eval = 100

        self.train_indices = []
        self.val_indices = []
        self.eval_indices = []
        self.loss_history = LossHistory()

        self._create_indices()

    def to_json(self):
        data = {
            "part_name": self.part_name,
            "total_size": self.total_size,
            "loss_history": self.loss_history.to_json()
        }
        return json.dumps(data, indent=4)

    def _create_indices(self):
        all_indices = list(range(self.total_size))
        random.shuffle(all_indices)

        val_size = int(self.total_size * self.val_ratio)
        train_size = self.total_size - val_size

        self.train_indices = all_indices[:train_size]
        self.val_indices = all_indices[train_size:]
        self.eval_indices = random.sample(self.val_indices, self.num_samples_for_eval)

    def create_dataloaders(self, dataset: torch.utils.data.Dataset, train_batch_size=4, val_batch_size=4,
                           eval_batch_size=1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_subset = Subset(dataset, self.train_indices)
        val_subset = Subset(dataset, self.val_indices)
        eval_subset = Subset(dataset, self.eval_indices)

        train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)
        eval_loader = DataLoader(eval_subset, batch_size=eval_batch_size, shuffle=False)

        return train_loader, val_loader, eval_loader


def normalize_x_and_y_labels(sorted_img_tensor_grps):
    # call this if max/min values of x and y need to be found
    norm_dataloader_wp = tqdm(
        DataLoader(PosePredictionLabelDataset(sorted_img_tensor_grps), batch_size=4, shuffle=False),
        desc="Emit all values")
    max_x = 0
    max_y = 0
    min_x = inf
    min_y = inf
    for i, batch in enumerate(norm_dataloader_wp):
        inputs, pose_prediction_labels, img_group_index = batch
        x_values = pose_prediction_labels[:, -2]
        y_values = pose_prediction_labels[:, -1]

        max_x = max(max_x, x_values.max().item())
        max_y = max(max_y, y_values.max().item())

        min_x = min(min_x, x_values.min().item())
        min_y = min(min_y, y_values.min().item())
    print(f"Max X: {max_x}, Max Y: {max_y}, Min X: {min_x}, Min Y: {min_y}")
