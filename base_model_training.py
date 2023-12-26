import base64
import random
from enum import Enum
from io import BytesIO
from math import inf
from typing import List, Dict, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from image_group import ImageGroup
from image_loader import PosePredictionLabelDataset


class Phase(Enum):
    TRAINING = "training"
    VALIDATION = "validation"


class CurrentRunningLoss(BaseModel):
    running_loss: float = 0.0
    current_avg_val_loss: float = float('inf')
    current_avg_train_loss: float = float('inf')

    def reset(self):
        self.running_loss = 0.0
        self.current_avg_val_loss: float = float('inf')
        self.current_avg_train_loss: float = float('inf')

    def has_avg_train_loss(self):
        return self.current_avg_train_loss < float('inf')

    def has_avg_val_loss(self):
        return self.current_avg_val_loss < float('inf')

    def add_running_loss(self, batch_num, running_loss, phase):
        if phase == Phase.TRAINING:
            if batch_num == 0:
                self.reset()
            self.running_loss += running_loss
            self.current_avg_train_loss = self.running_loss / (batch_num + 1)
        elif phase == Phase.VALIDATION:
            if batch_num == 0:
                self.reset()
            self.running_loss += running_loss
            self.current_avg_val_loss = self.running_loss / (batch_num + 1)
        else:
            raise ValueError("Unknown phase : " + phase)


class LossHistory(BaseModel):
    training_losses: List[float] = []
    validation_losses: List[float] = []
    min_val_loss: float = float('inf')
    current_running_loss: CurrentRunningLoss = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.validation_losses:
            self.min_val_loss = min(self.validation_losses)

    def add_loss(self, training_loss, validation_loss) -> bool:
        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)
        self.current_running_loss = CurrentRunningLoss()
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            return True  # Indicate that the model should be saved
        return False

    def get_loss_plot_base64(self):
        plt.figure()
        epochs = np.arange(1, len(self.training_losses) + 1)
        plt.plot(epochs, self.training_losses, label='Training Loss')
        plt.plot(epochs, self.validation_losses, label='Validation Loss')

        current_epoch = len(epochs) + 1

        # Plot current average losses if available
        has_avg_train_loss = self.current_running_loss and self.current_running_loss.has_avg_train_loss()
        has_avg_val_loss = self.current_running_loss and self.current_running_loss.has_avg_val_loss()
        if has_avg_train_loss:
            plt.scatter(current_epoch, self.current_running_loss.current_avg_train_loss, color='blue', marker='x',
                        label='Current Avg Training Loss')

        if has_avg_val_loss:
            plt.scatter(current_epoch, self.current_running_loss.current_avg_val_loss, color='orange', marker='x',
                        label='Current Avg Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        max_loss = 0
        if self.training_losses and self.validation_losses:
            max_loss = max(max(self.training_losses), max(self.validation_losses))
        if has_avg_val_loss:
            max_loss = max(max_loss, self.current_running_loss.current_avg_val_loss)
        if has_avg_train_loss:
            max_loss = max(max_loss, self.current_running_loss.current_avg_train_loss)
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
        if self.current_running_loss is None:
            self.current_running_loss = CurrentRunningLoss()
        self.current_running_loss.add_running_loss(batch_num, running_loss, phase)


class DatasetPartMetaInfo(BaseModel):
    part_name: str
    val_ratio: float = 0.2
    num_samples_for_eval: int = 100
    train_indices: List[str] = []
    val_indices: List[str] = []
    eval_indices: List[str] = []
    all_indices: List[str] = []
    base_output_path: str
    loss_history: LossHistory = LossHistory()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_indices()

    def _create_indices(self):
        if self.val_indices and len(self.val_indices) > 0:
            # already initialized / loaded from json, we may not redo the randomization split
            return

        print(f"Attention, we are doing the random split in train/validation data! this should only happen once per {self.part_name}")

        random.shuffle(self.all_indices)

        total_size = len(self.all_indices)

        val_size = int(total_size * self.val_ratio)
        train_size = total_size - val_size

        self.train_indices = self.all_indices[:train_size]
        self.val_indices = self.all_indices[train_size:]
        self.eval_indices = random.sample(self.val_indices, self.num_samples_for_eval)

    def create_dataloaders(self, dataset: torch.utils.data.Dataset, train_batch_size=4, val_batch_size=4,
                           eval_batch_size=1) -> Tuple[DataLoader, DataLoader, DataLoader]:

        image_group_map = {}

        for formatted_image_index in self.all_indices:
            img_group = ImageGroup(formatted_image_index=formatted_image_index)
            img_group.initialize_output_only(self.base_output_path, False)
            image_group_map[formatted_image_index] = img_group

        train_data = [dataset[self.image_group_map[index]] for index in self.train_indices]
        val_data = [dataset[self.image_group_map[index]] for index in self.val_indices]
        eval_data = [dataset[self.image_group_map[index]] for index in self.eval_indices]

        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False)

        return train_loader, val_loader, eval_loader


def save_model_and_history(model, loss_history, filename):
    state = {
        'model_state_dict': model.state_dict(),
        'loss_history': {
            'training_losses': loss_history.training_losses,
            'validation_losses': loss_history.validation_losses
        }
    }
    torch.save(state, filename)


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
