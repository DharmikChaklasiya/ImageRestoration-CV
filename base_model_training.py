import base64
import json
import os
import random
from enum import Enum
from io import BytesIO
from math import inf
from typing import List, Dict, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from pydantic import BaseModel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from image_group import ImageGroup
from image_loader import PosePredictionLabelDataset, load_input_image_parts


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
    base_output_path: str
    loss_histories: Dict[str, LossHistory] = {}
    train_indices: List[str] = []
    val_indices: List[str] = []
    eval_indices: List[str] = []
    all_indices: List[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_indices()

    def _create_indices(self):
        if self.val_indices and len(self.val_indices) > 0:
            # already initialized / loaded from json, we may not redo the randomization split
            return

        print(
            f"Attention, we are doing the random split in train/validation data! this should only happen once per {self.part_name}")

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

        image_formatted_to_numerical_index_map = {}

        for i, (_, _, formatted_image_index) in enumerate(dataset):
            formatted_image_index = image_group_map[formatted_image_index].formatted_image_index
            image_formatted_to_numerical_index_map[formatted_image_index] = i

        num_train_indices = [image_formatted_to_numerical_index_map[idx] for idx in self.train_indices]
        num_val_indices = [image_formatted_to_numerical_index_map[idx] for idx in self.val_indices]
        num_eval_indices = [image_formatted_to_numerical_index_map[idx] for idx in self.eval_indices]

        train_subset = Subset(dataset, num_train_indices)
        val_subset = Subset(dataset, num_val_indices)
        eval_subset = Subset(dataset, num_eval_indices)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
        eval_loader = DataLoader(eval_subset, batch_size=1, shuffle=False)

        return train_loader, val_loader, eval_loader

    def get_loss_history(self, model_file_name):
        return self.loss_histories.setdefault(model_file_name, LossHistory())


def save_model_and_history(model, loss_history, filename):
    state = {
        'model_state_dict': model.state_dict(),
        'loss_history': {
            'training_losses': loss_history.training_losses,
            'validation_losses': loss_history.validation_losses
        }
    }
    torch.save(state, filename)


def load_model_and_history(model, filename):
    try:
        state = torch.load(filename)
        model.load_state_dict(state['model_state_dict'])

        print(f"\nLoaded model from file {filename}.")

        loss_history_data = state['loss_history']
        loss_history = LossHistory()
        loss_history.training_losses = loss_history_data['training_losses']
        loss_history.validation_losses = loss_history_data['validation_losses']

        return loss_history
    except FileNotFoundError:
        print(f"File {filename} not found. Loading a new model and an empty loss history.")
        return LossHistory()


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


def load_dataset_infos(all_parts):
    ds_parts: Dict[str, DatasetPartMetaInfo] = {}
    for part in all_parts:
        # Define the file path for the saved data
        file_path = get_file_path(part)

        if os.path.exists(file_path):
            print(f"Load the existing DatasetPartMetaInfo from the file {file_path}")
            with open(file_path, 'r') as file:
                data = json.load(file)
                dataset_part_info = DatasetPartMetaInfo.parse_obj(data)
                ds_parts[part] = dataset_part_info
        else:
            print(f"Create new DatasetPartMetaInfo and save it to file {file_path}")
            all_image_groups, image_group_map = load_input_image_parts([part])
            dataset_part_info = DatasetPartMetaInfo(part_name=part,
                                                    all_indices=[img_group.formatted_image_index for img_group in
                                                                 all_image_groups],
                                                    base_output_path=all_image_groups[0].base_path)

            save_datasetpart_metainfo(dataset_part_info)

            ds_parts[part] = dataset_part_info

    return ds_parts


def save_datasetpart_metainfo(dataset_part_info):
    file_path = get_file_path(dataset_part_info.part_name)

    with open(file_path, 'w') as file:
        json.dump(dataset_part_info.dict(), file, indent=4)


def get_file_path(part_name: str):
    file_path = f"dataset_infos/{part_name}_dataset_info.json"
    return file_path
