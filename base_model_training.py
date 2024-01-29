import base64
import json
import math
import os
import random
from datetime import datetime
from enum import Enum
from io import BytesIO
from math import inf
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from image_group import ImageGroup
from image_loader import PosePredictionLabelDataset, ImageTensorGroup
from parameter_file_parser import BoundingBox, ImageDimension


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
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_samples_for_eval: int = 100
    base_output_path: str
    train_indices: List[str] = []
    val_indices: List[str] = []
    test_indices: List[str] = []
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
        test_size = int(total_size * self.test_ratio)
        train_size = total_size - val_size - test_size

        self.train_indices = self.all_indices[:train_size]
        self.val_indices = self.all_indices[train_size:train_size + val_size]
        self.test_indices = self.all_indices[train_size + val_size:]
        self.eval_indices = random.sample(self.val_indices, self.num_samples_for_eval)

    def create_dataloaders(self, dataset: torch.utils.data.Dataset, train_batch_size=4, val_batch_size=4,
                           test_batch_size=1, eval_batch_size=1) -> tuple[
            DataLoader[Any], DataLoader[Any], DataLoader[Any], DataLoader[Any]]:

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
        num_test_indices = [image_formatted_to_numerical_index_map[idx] for idx in self.test_indices]
        num_eval_indices = [image_formatted_to_numerical_index_map[idx] for idx in self.eval_indices]

        train_subset = Subset(dataset, num_train_indices)
        val_subset = Subset(dataset, num_val_indices)
        test_subset = Subset(dataset, num_test_indices)
        eval_subset = Subset(dataset, num_eval_indices)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)
        eval_loader = DataLoader(eval_subset, batch_size=eval_batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, eval_loader

    def limit_dataset_size(self, n: int):
        """
        Limit the dataset to the first n elements while respecting the train/validation split.
        """
        assert n > 0, "n must be greater than 0"
        assert n <= len(self.all_indices), "n is greater than the size of the dataset"

        # Calculate the number of training and validation samples based on the split ratio
        val_size = int(n * self.val_ratio)
        train_size = n - val_size

        # Adjust train_indices, val_indices, and all_indices
        self.train_indices = self.train_indices[:train_size]
        self.val_indices = self.val_indices[:val_size]
        self.all_indices = self.train_indices + self.val_indices

        # Adjust eval_indices
        if len(self.val_indices) < self.num_samples_for_eval:
            self.eval_indices = self.val_indices.copy()
        else:
            self.eval_indices = random.sample(self.val_indices, self.num_samples_for_eval)

    def limit_validation_dataset_size(self, n: int):
        """
        Limit the dataset to the first n elements while respecting the train/validation split.
        """
        assert n > 0, "n must be greater than 0"
        assert n <= len(self.val_indices), "n is greater than the size of the dataset"
        self.val_indices = self.val_indices[:n]


class StoredLossHistory(BaseModel):
    training_losses: list[float] = []
    validation_losses: list[float] = []


class ModelRunSummary(BaseModel):
    current_part_name: str
    current_epoch: int
    current_best_model: str
    html_summary: str
    loss_histories: dict[str, StoredLossHistory]
    model: nn.Module = None
    device: str = None
    model_run_root_path: str = None
    model_run_name: str = None
    model_run_summary_file_name: str = None

    def __init__(self,
                 current_part_name: str,
                 current_epoch: int,
                 current_best_model: str,
                 html_summary: str,
                 loss_histories: dict[str, StoredLossHistory],
                 **kwargs):
        super().__init__(current_part_name=current_part_name,
                         current_epoch=current_epoch,
                         current_best_model=current_best_model,
                         html_summary=html_summary,
                         loss_histories=loss_histories,
                         **kwargs)

    class Config:
        arbitrary_types_allowed = True
        # exclude fields from serialization
        fields = {
            'model': {'exclude': True, 'arbitrary_types_allowed': True},
            'device': {'exclude': True},
            'model_run_root_path': {'exclude': True},
            'model_run_name': {'exclude': True},
            'model_run_summary_file_name': {'exclude': True}
        }

    def get_loss_history(self) -> LossHistory:
        loss_history = LossHistory()
        stored_loss_history = self.loss_histories.setdefault(self.current_part_name, StoredLossHistory())
        loss_history.training_losses = stored_loss_history.training_losses
        loss_history.validation_losses = stored_loss_history.validation_losses
        loss_history.min_val_loss = min(loss_history.validation_losses) if len(
            loss_history.validation_losses) > 0 else math.inf
        return loss_history

    def get_html_summary_path(self):
        return os.path.join(self.model_run_root_path, self.model_run_name, self.html_summary)

    def get_model_file_name(self):
        return os.path.join(self.model_run_root_path, self.model_run_name, self.current_best_model)

    def train_model_on_all_batches(self, dataset_parts, num_super_batches, wandbinit, train_model_on_one_batch):

        if self.current_part_name is not None:
            print(f"Restarting from {self.current_part_name}")
            start_from_current_part = True
        else:
            start_from_current_part = False

        for i in range(1, num_super_batches + 1):
            super_batch_info = f"Super-Batch: {i}/{num_super_batches}"
            print(f"Running the model in super-batches - {super_batch_info}")
            for part, dataset_metainfo in dataset_parts.items():
                if start_from_current_part:
                    if part == self.current_part_name:
                        start_from_current_part = False
                    else:
                        continue
                self.current_part_name = part
                wandbinit(part)
                train_model_on_one_batch(dataset_metainfo, self, super_batch_info)
                wandb.finish()

        print("Training complete - printing results.")

        for part, dataset_metainfo in dataset_parts.items():
            print(dataset_metainfo.json())

    def update(self, loss_history):
        self.loss_histories[self.current_part_name].training_losses = loss_history.training_losses
        self.loss_histories[self.current_part_name].validation_losses = loss_history.validation_losses
        file_path = os.path.join(self.model_run_root_path, self.model_run_name, self.model_run_summary_file_name)

        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=4)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        wandb.log({"epoch": len(loss_history.training_losses),
                   "loss": loss_history.training_losses[-1],
                   "val_loss": loss_history.validation_losses[-1],
                   "timestamp": timestamp_str})


def load_model_run_summary(model_run_name: str, model: nn.Module):
    model_run_root_path = 'model-runs'
    file_name_of_summary = 'model_run_summary.json'
    summary_file_path = os.path.join(model_run_root_path, model_run_name, file_name_of_summary)

    model_run_summary: ModelRunSummary

    try:
        with open(summary_file_path) as f:
            data = json.load(f)

        model_run_summary = ModelRunSummary.parse_obj(data)
        model_file_path = os.path.join(model_run_root_path, model_run_name, model_run_summary.current_best_model)
        load_model_optionally(model, model_file_path)

        model_run_summary.model = model
        model_run_summary.device = next(model.parameters()).device
    except FileNotFoundError as e:
        print(
            f"Model run summary file {summary_file_path} not found: {e}. Creating a new model run with an empty loss history.")
        model_run_summary = ModelRunSummary(current_part_name='Part1',
                                            current_epoch=0,
                                            current_best_model=model_run_name + ".pth",
                                            html_summary=model_run_name + ".html", loss_histories={})
        model_run_summary.model = model
        model_run_summary.device = next(model.parameters()).device

    model_run_summary.model_run_root_path = model_run_root_path
    model_run_summary.model_run_name = model_run_name
    model_run_summary.model_run_summary_file_name = file_name_of_summary
    os.makedirs(model_run_summary.model_run_root_path, exist_ok=True)
    os.makedirs(os.path.join(model_run_summary.model_run_root_path, model_run_summary.model_run_name), exist_ok=True)
    return model_run_summary


def save_model(model_run_summary: ModelRunSummary):
    state = {
        'model_state_dict': model_run_summary.model.state_dict()
    }
    torch.save(state, model_run_summary.get_model_file_name())

    artifact = wandb.Artifact(model_run_summary.current_best_model, type="model_dict")
    artifact.add_file(model_run_summary.get_model_file_name())
    wandb.log_artifact(artifact)


def load_model_optionally(model: nn.Module, filename: str):
    try:
        state = torch.load(filename)
        model.load_state_dict(state['model_state_dict'])

        print(f"\nLoaded model from file {filename}.")
    except FileNotFoundError:
        print(f"File {filename} not found. Loading a new model")


def load_model(model, filename):
    try:
        state = torch.load(filename)
        model.load_state_dict(state['model_state_dict'])

        print(f"\nLoaded model from file {filename}.")

    except FileNotFoundError as e:
        raise ValueError(
            f"File {filename} not found, but we need to have a pretrained model for the integration.") from e


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

        file_path = get_file_path(part)

        if os.path.exists(file_path):
            print(f"Load the existing DatasetPartMetaInfo from the file {file_path}")
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    dataset_part_info = DatasetPartMetaInfo.parse_obj(data)
                    ds_parts[part] = dataset_part_info
                except Exception as e:
                    raise ValueError("Error while parsing file: " + file_path) from e
        else:
            print(f"Create new DatasetPartMetaInfo and save it to file {file_path}")
            all_image_groups, image_group_map = load_input_image_parts(part)
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


def preload_images_from_drive(batch_part: DatasetPartMetaInfo, sorted_image_groups: List[ImageGroup], super_batch_info,
                              focal_stack_indices: List[int] = None) -> (
        List[ImageTensorGroup], Dict[str, ImageTensorGroup]):
    sorted_image_tensor_groups = []
    image_tensor_group_map: Dict[str, ImageTensorGroup] = {}
    focal_stack_indices: List[int] = \
        [0, 1, 2, 3, 7, 10, 15, 20, 25, 30] if focal_stack_indices is None else focal_stack_indices

    sorted_img_groups_with_progress = tqdm(sorted_image_groups, desc="Preloading images for batches")
    filtered_image_groups = []

    batch_part_all_indices = set(batch_part.all_indices)

    for i, img_group in enumerate(sorted_img_groups_with_progress):
        if img_group.formatted_image_index in batch_part_all_indices:
            image_tensor_group = ImageTensorGroup(img_group, focal_stack_indices)
            image_tensor_group.load_images()
            image_tensor_group_map[img_group.formatted_image_index] = image_tensor_group
            sorted_image_tensor_groups.append(image_tensor_group)
        else:
            filtered_image_groups.append(img_group)

        sorted_img_groups_with_progress.set_description(
            f"{super_batch_info}-part:{batch_part.part_name}-Preloading images for batches: {i}/{len(sorted_image_groups)} ")

    if len(filtered_image_groups) > 0:
        print(f"\nWe filtered image groups, because they were on the drive, "
              f"but not in the batch definition, amount : {len(filtered_image_groups)}")

    return sorted_image_tensor_groups, image_tensor_group_map


def load_input_image_parts(part: DatasetPartMetaInfo) -> Tuple[List[ImageGroup], Dict[str, ImageGroup]]:
    all_image_groups: List[ImageGroup] = []
    image_group_map: Dict[str, ImageGroup] = {}

    root_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "computervision", "integrals"))

    print(f"Will load parts: {part.part_name} data from root dir: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name == part.part_name:
                sub_folder_path = os.path.join(root, dir_name)

                all_indices = set(part.all_indices)

                filtered_indices = []

                for formatted_image_index in os.listdir(sub_folder_path):
                    if formatted_image_index in all_indices:
                        img_group = ImageGroup(formatted_image_index=formatted_image_index)
                        img_group.initialize_output_only(sub_folder_path, True)
                        if img_group.valid:
                            all_image_groups.append(img_group)
                            image_group_map[img_group.formatted_image_index] = img_group
                        else:
                            print("Invalid image : " + img_group.base_path)
                    else:
                        filtered_indices.append(formatted_image_index)

                if len(filtered_indices) > 0:
                    print(
                        f"\nFiltered out indices because they are not part of the current batch definition, amount : {len(filtered_indices)}")

    all_image_groups = sorted(all_image_groups, key=lambda img_group1: int(img_group1.formatted_image_index))

    return all_image_groups, image_group_map


class AccentedLoss:
    def __init__(self, normal_loss, alpha: float = 0.9):
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.normal_loss = normal_loss
        self.alpha: float = alpha

    def calculate_combined_loss(self, bboxes: List[BoundingBox], predictions, targets):
        masks = torch.zeros_like(predictions)
        for i in range(predictions.size(0)):
            x_min, y_min, x_max, y_max = bboxes[i].correct_the_box(ImageDimension(512, 512))
            masks[i, :, y_min:y_max, x_min:x_max] = 1

        masked_predictions = predictions * masks
        masked_targets = targets * masks

        loss = self.mse_loss(masked_predictions, masked_targets)

        accented_loss = loss.sum() / masks.sum()

        normal_loss = self.normal_loss(predictions, targets)

        return (1 - self.alpha) * normal_loss + self.alpha * accented_loss


wandb_api_key = 'e84f9aa9585ed75083b2923b94f68b05c287fe7e'


def find_existing_run(project_name, run_name):
    api = wandb.Api()
    runs = api.runs(f"{project_name}")
    current_run = None
    for run in runs:
        if run.name == run_name:
            current_run = run
            break
    if current_run:
        print("Wand will resume run : " + current_run.id)
    else:
        print("No existing run found in this project fitting this run name - so we are starting a new run from scratch")
    return current_run
