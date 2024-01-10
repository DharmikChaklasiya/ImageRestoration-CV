import random
from typing import List

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
            transforms.ToTensor()  # this doesn't seem to help because ToTensor already normalizes, but Ivan will
            # investigate... transforms.Normalize(mean=[0.5], std=[0.5])
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

        shape_encoded, x_min, y_min, x_max, y_max = self.parse_param_file(self.image_group.parameter_file)

        self.pose_prediction_labels = torch.tensor(
            [x_min, y_min, x_max, y_max])  # torch.tensor(shape_encoded + [x, y]) we start slowly -
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


class ImageCropper:
    def __init__(self, divisor=8):
        self.divisor = divisor

    def _random_divisible_integer(self, min_value, max_value):
        # Adjust the range to ensure it's within divisible bounds
        min_value = min_value + (self.divisor - min_value % self.divisor) % self.divisor
        max_value = max_value - max_value % self.divisor
        if min_value > max_value:
            raise ValueError("No divisible value (divisable by : {}) in the specified range : {}, {}".format(self.divisor, min_value, max_value))
        return random.choice(range(min_value, max_value + 1, self.divisor))

    def _new_random_width(self, min_width, max_width):
        return self._random_divisible_integer(min_width, max_width)

    def _new_random_height(self, min_height, max_height):
        return self._random_divisible_integer(min_height, max_height)

    def _new_random_x(self, min_x, max_x):
        return random.randint(min_x, max_x)

    def _new_random_y(self, min_y, max_y):
        return random.randint(min_y, max_y)

    def randomly_crop_image(self, img_shape, x_min, y_min, x_max, y_max, new_bbox_width=None, new_bbox_height=None):
        _, H, W = img_shape

        assert x_min >= 0, "x_min : {} must be greater 0".format(x_min)
        assert y_min >= 0, "y_min : {} must be greater 0".format(y_min)
        assert x_min < x_max, "x_min : {} must be smaller than x_max : {}".format(x_min, x_max)
        assert y_min < y_max, "y_min : {} must be smaller than y_max : {}".format(y_min, y_max)
        assert x_max <= W, "x_max : {} must be smaller {}".format(x_max, W)
        assert y_max <= H, "y_max : {} must be smaller {}".format(y_max, H)

        # Check if the bounding box is the entire image
        if x_min == 0 and y_min == 0 and x_max >= W - 1 and y_max >= H - 1:
            bbox_width = new_bbox_width if new_bbox_width is not None else self._new_random_width(50, 100)
            bbox_height = new_bbox_height if new_bbox_height is not None else self._new_random_height(50, 100)
            x_start = self._new_random_x(0, W - bbox_width)
            y_start = self._new_random_y(0, H - bbox_height)
            x_end = x_start + bbox_width
            y_end = y_start + bbox_height
        else:
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            crop_max_width = min(W, bbox_width * 2)
            crop_max_height = min(H, bbox_height * 2)
            crop_width = new_bbox_width if new_bbox_width is not None else self._new_random_width(bbox_width, crop_max_width)
            crop_height = new_bbox_height if new_bbox_height is not None else self._new_random_height(bbox_height, crop_max_height)

            min_x_start = max(0, x_min - (crop_width - bbox_width))
            max_x_start = min(W - crop_width, x_min)
            x_start = self._new_random_x(min_x_start, max_x_start)

            min_y_start = max(0, y_min - (crop_height - bbox_height))
            max_y_start = min(H - crop_height, y_min)
            y_start = self._new_random_y(min_y_start, max_y_start)

            x_end = min(W, x_start + crop_width)
            y_end = min(H, y_start + crop_height)

            assert x_start <= x_min, "Cropping box starts after the initial bounding box on X-axis : {}, {}".format(
                x_start, x_min)
            assert y_start <= y_min, "Cropping box starts after the initial bounding box on Y-axis : {}, {}".format(
                y_start, y_min)
            assert x_end >= x_max, "Cropping box ends before the initial bounding box on X-axis : {}, {}".format(x_end,
                                                                                                                 x_max)
            assert y_end >= y_max, "Cropping box ends before the initial bounding box on Y-axis : {}, {}".format(y_end,
                                                                                                                 y_max)

            assert crop_width >= bbox_width, "Cropping box width is smaller than the initial bounding box width"
            assert crop_height >= bbox_height, "Cropping box height is smaller than the initial bounding box height"
            # assert crop_width <= 2 * bbox_width, "Cropping box width is more than twice the initial bounding box width"
            # assert crop_height <= 2 * bbox_height, "Cropping box height is more than twice the initial bounding box height"

        return x_start, y_start, x_end, y_end
