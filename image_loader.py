import os
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from image_group import ImageGroup
from parameter_file_parser import process_content, PersonShape, calculate_bounding_box


class ImageTensorGroup:
    def __init__(self, image_group: ImageGroup, focal_stack_indices):
        self.pose_prediction_labels = None
        self.ground_truth_tensor = None
        self.image_tensors = []
        self.image_group = image_group
        self.focal_stack_indices = focal_stack_indices  # Indices of the images in the focal stack to load
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.shape_mapping = {PersonShape.LAYING: [1, 0, 0, 0], PersonShape.SITTING: [0, 1, 0, 0],
                              PersonShape.IDLE: [0, 0, 1, 0], PersonShape.NO_PERSON: [0, 0, 0, 1]}

    def load_images(self):
        self.image_tensors = []
        for index in self.focal_stack_indices:
            image_path = self.image_group.output_image_name(index)
            image = Image.open(image_path).convert('L')
            self.image_tensors.append(self.transform(image))

        try:
            opened_image = Image.open(self.image_group.ground_truth_file).convert('L')
        except Exception as e:
            raise ValueError(f"Error while loading image from: {self.image_group.ground_truth_file}") from e
        self.ground_truth_tensor = self.transform(opened_image)

        shape_encoded, x_min, x_max, y_min, y_max = self.parse_param_file(self.image_group.parameter_file)

        self.pose_prediction_labels = torch.tensor([x_min, x_max, y_min, y_max])  # torch.tensor(shape_encoded + [x, y]) we start slowly -
        # predicting everything sadly doesn't work at all

    def get_images(self):
        stacked_images = self.images_to_tensors(self.image_tensors, self.image_group.formatted_image_index)
        return stacked_images, self.ground_truth_tensor, self.image_group.formatted_image_index

    def get_images_with_pose_prediction_labels(self):
        stacked_images = self.images_to_tensors(self.image_tensors, self.image_group.formatted_image_index)
        return stacked_images, self.pose_prediction_labels, self.image_group.formatted_image_index

    def parse_param_file(self, parameter_file):
        shape_encoded, x, y, rotz = process_content(parameter_file)

        x_min, y_min, x_max, y_max = calculate_bounding_box(shape_encoded, x, y, rotz)

        try:
            shape_encoded = self.shape_mapping[shape_encoded]
        except KeyError as e:
            raise ValueError(
                f"Shape encoding '{shape_encoded}' not found in shape mapping for file: {parameter_file}") from e

        return shape_encoded, x_min, y_min, x_max, y_max

    def images_to_tensors(self, image_tensors, image_index):
        stacked_images = torch.cat(image_tensors, dim=0)  # Shape: (focal_stack_indices, 512, 512)
        focal_stack_height = len(self.focal_stack_indices)
        assert stacked_images.shape[0] == focal_stack_height, (
            "Issue while processing image_group : {} - dimension 1 should be {}, but is {}"
            .format(image_index, focal_stack_height, stacked_images.shape[0]))
        assert stacked_images.shape[1] == 512, ("Issue while processing image_group : {} - dimension 2 should be 512, "
                                                "but is {}").format(image_index,
                                                                    stacked_images.shape[1])
        assert stacked_images.shape[2] == 512, ("Issue while processing image_group : {} - dimension 3 should be 512, "
                                                "but is {}").format(image_index,
                                                                    stacked_images.shape[2])
        return stacked_images


class GroundTruthLabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_tensor_groups: List[ImageTensorGroup]):
        self.image_tensor_groups = image_tensor_groups

    def __getitem__(self, index):
        img_tensor_group = self.image_tensor_groups[index]
        return img_tensor_group.get_images()

    def __len__(self):
        return len(self.image_tensor_groups)


class PosePredictionLabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_tensor_groups: List[ImageTensorGroup]):
        self.image_tensor_groups = image_tensor_groups

    def __getitem__(self, index):
        img_tensor_group: ImageTensorGroup = self.image_tensor_groups[index]
        return img_tensor_group.get_images_with_pose_prediction_labels()

    def __len__(self):
        return len(self.image_tensor_groups)


def load_input_image_parts(parts: List[str]) -> Tuple[List[ImageGroup], Dict[str, ImageGroup]]:
    all_image_groups: List[ImageGroup] = []
    image_group_map: Dict[str, ImageGroup] = {}

    root_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "computervision", "integrals"))

    print(f"Will load parts: {parts} data from root dir: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name in parts:
                sub_folder_path = os.path.join(root, dir_name)

                for formatted_image_index in os.listdir(sub_folder_path):
                    img_group = ImageGroup(formatted_image_index=formatted_image_index)
                    img_group.initialize_output_only(sub_folder_path, True)
                    if img_group.valid:
                        all_image_groups.append(img_group)
                        image_group_map[img_group.formatted_image_index] = img_group
                    else:
                        print("Invalid image : " + img_group.base_path)

    all_image_groups = sorted(all_image_groups, key=lambda img_group1: int(img_group1.formatted_image_index))

    return all_image_groups, image_group_map

